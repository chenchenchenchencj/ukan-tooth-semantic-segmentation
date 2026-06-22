
import argparse, csv, json, os, random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

import archs
import losses


def init_dist():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank(); world = dist.get_world_size(); local = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local)
        return True, rank, world, local
    return False, 0, 1, 0


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class ToothSemanticDataset(Dataset):
    def __init__(self, data_dir, ids, h=320, w=640, train=False):
        self.data_dir = Path(data_dir); self.ids = ids; self.h = h; self.w = w; self.train = train
        self.img_dir = self.data_dir / 'busi' / 'images'
        self.mask_dir = self.data_dir / 'busi' / 'masks' / '0'
        if not self.mask_dir.exists(): self.mask_dir = self.data_dir / 'busi' / 'masks'
    def __len__(self): return len(self.ids)
    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img = cv2.imread(str(self.img_dir / f'{img_id}.png'), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.mask_dir / f'{img_id}_mask.png'), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        if self.train:
            # Dental panoramics are left-right symmetric enough for horizontal flip,
            # but vertical flip is anatomically invalid and hurts generalization.
            if random.random() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1]); mask = np.ascontiguousarray(mask[:, ::-1])
            if random.random() < 0.70:
                hh, ww = img.shape[:2]
                angle = random.uniform(-7.0, 7.0)
                scale = random.uniform(0.94, 1.06)
                tx = random.uniform(-0.025, 0.025) * ww
                ty = random.uniform(-0.020, 0.020) * hh
                M = cv2.getRotationMatrix2D((ww * 0.5, hh * 0.5), angle, scale)
                M[0, 2] += tx; M[1, 2] += ty
                img = cv2.warpAffine(img, M, (ww, hh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                mask = cv2.warpAffine(mask, M, (ww, hh), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            if random.random() < 0.65:
                alpha = random.uniform(0.82, 1.18); beta = random.uniform(-14, 14)
                img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
            if random.random() < 0.30:
                gamma = random.uniform(0.85, 1.20)
                lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
                img = cv2.LUT(img, lut)
            if random.random() < 0.18:
                noise = np.random.normal(0, random.uniform(2.0, 6.0), img.shape).astype(np.float32)
                img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        img = img.astype(np.float32) / 255.0
        img = (img - np.array([0.485,0.456,0.406], np.float32)) / np.array([0.229,0.224,0.225], np.float32)
        mask = (mask > 0).astype(np.float32)[None]
        return torch.from_numpy(img.transpose(2,0,1)), torch.from_numpy(mask), img_id


def make_split(data_dir, split_file, seed=20260515):
    split_file = Path(split_file)
    if split_file.exists(): return json.loads(split_file.read_text(encoding='utf-8'))
    ids = sorted([p.stem for p in (Path(data_dir)/'busi'/'images').glob('*.png')])
    rng = random.Random(seed); rng.shuffle(ids)
    n=len(ids); n_val=int(round(n*0.1)); n_test=int(round(n*0.1))
    split={'train':ids[:n-n_val-n_test], 'val':ids[n-n_val-n_test:n-n_test], 'test':ids[n-n_test:]}
    split_file.parent.mkdir(parents=True, exist_ok=True)
    split_file.write_text(json.dumps(split, indent=2, ensure_ascii=False), encoding='utf-8')
    return split


def metric_sums(logits, target):
    pred = (torch.sigmoid(logits) > 0.5).float()
    dims=(1,2,3); inter=(pred*target).sum(dims); ps=pred.sum(dims); ts=target.sum(dims)
    dice=(2*inter+1e-6)/(ps+ts+1e-6); iou=(inter+1e-6)/(ps+ts-inter+1e-6)
    prec=(inter+1e-6)/(ps+1e-6); rec=(inter+1e-6)/(ts+1e-6)
    return torch.stack([dice.sum(), iou.sum(), prec.sum(), rec.sum(), torch.tensor(float(logits.size(0)), device=logits.device)])


def run_epoch(model, loader, criterion, opt, scaler, device, train, rank, distributed):
    model.train(train)
    loss_sum=torch.tensor(0., device=device); count=torch.tensor(0., device=device); metric_sum=torch.zeros(5, device=device)
    iterator = loader
    if rank == 0 and tqdm is not None:
        iterator = tqdm(loader, dynamic_ncols=True, leave=False)
    for img, mask, _ in iterator:
        img=img.to(device, non_blocking=True); mask=mask.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            with torch.amp.autocast('cuda', enabled=device.type=='cuda'):
                out=model(img); loss=criterion(out, mask)
            if train:
                opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); scaler.step(opt); scaler.update()
        bs=torch.tensor(float(img.size(0)), device=device)
        loss_sum += loss.detach() * bs; count += bs; metric_sum += metric_sums(out.detach(), mask)
        if rank == 0 and tqdm is not None:
            iterator.set_postfix(loss=float(loss_sum/count), dice=float(metric_sum[0]/metric_sum[4]))
    packed=torch.cat([loss_sum.view(1), count.view(1), metric_sum])
    if distributed: dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    loss_avg=float(packed[0]/packed[1]); denom=packed[6].clamp_min(1)
    return {'loss':loss_avg, 'dice':float(packed[2]/denom), 'iou':float(packed[3]/denom), 'precision':float(packed[4]/denom), 'recall':float(packed[5]/denom)}


@torch.no_grad()
def export_visuals(model, loader, out_dir, device, limit=16):
    out_dir=Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True); model.eval(); count=0
    mean=np.array([0.485,0.456,0.406]); std=np.array([0.229,0.224,0.225])
    for img, mask, ids in loader:
        img=img.to(device); mask=mask.to(device); pred=(torch.sigmoid(model(img))>0.5).float()
        for b in range(img.size(0)):
            if count >= limit: return
            arr=img[b].cpu().numpy().transpose(1,2,0); arr=np.clip((arr*std+mean)*255,0,255).astype(np.uint8)
            gt=(mask[b,0].cpu().numpy()*255).astype(np.uint8); pr=(pred[b,0].cpu().numpy()*255).astype(np.uint8)
            cv2.imwrite(str(out_dir/f'{ids[b]}__image.jpg'), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out_dir/f'{ids[b]}__gt.png'), gt); cv2.imwrite(str(out_dir/f'{ids[b]}__pred.png'), pr)
            ov=arr.copy(); ov[gt>0]=(0.5*ov[gt>0]+np.array([0,255,0])*0.5).astype(np.uint8); ov[pr>0]=(0.5*ov[pr>0]+np.array([255,0,0])*0.5).astype(np.uint8)
            cv2.imwrite(str(out_dir/f'{ids[b]}__overlap.jpg'), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR)); count+=1


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--arch', default='UKAN'); ap.add_argument('--loss', default='BCEDiceLoss'); ap.add_argument('--name', required=True)
    ap.add_argument('--data_dir', default='inputs'); ap.add_argument('--out_dir', default='outputs_semantic_ddp')
    ap.add_argument('--split_file', default='splits/tooth_semantic_80_10_10_seed20260515.json')
    ap.add_argument('--epochs', type=int, default=160); ap.add_argument('--patience', type=int, default=25)
    ap.add_argument('--batch_size', type=int, default=12); ap.add_argument('--input_h', type=int, default=320); ap.add_argument('--input_w', type=int, default=640)
    ap.add_argument('--lr', type=float, default=1e-4); ap.add_argument('--kan_lr', type=float, default=1e-3); ap.add_argument('--seed', type=int, default=20260515)
    ap.add_argument('--input_list', default='128,160,256'); ap.add_argument('--no_kan', action='store_true')
    ap.add_argument('--workers', type=int, default=6)
    args=ap.parse_args(); distributed,rank,world,local=init_dist(); seed_all(args.seed+rank)
    device=torch.device('cuda', local) if torch.cuda.is_available() else torch.device('cpu')
    if rank == 0: split=make_split(args.data_dir,args.split_file,args.seed)
    if distributed: dist.barrier()
    split=make_split(args.data_dir,args.split_file,args.seed)
    out_dir=Path(args.out_dir)/args.name
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir/'config.json').write_text(json.dumps(vars(args),indent=2,ensure_ascii=False), encoding='utf-8')
        (out_dir/'split_counts.json').write_text(json.dumps({k:len(v) for k,v in split.items()},indent=2), encoding='utf-8')
    embed=[int(x) for x in args.input_list.split(',')]
    model=archs.__dict__[args.arch](1,3,False,embed_dims=embed,no_kan=args.no_kan).to(device)
    if distributed: model=DDP(model, device_ids=[local], output_device=local)
    criterion=losses.__dict__[args.loss]().to(device)
    param_groups=[]
    for n,p in (model.module if hasattr(model,'module') else model).named_parameters():
        lr=args.kan_lr if ('layer' in n.lower() and 'fc' in n.lower()) else args.lr
        param_groups.append({'params':p,'lr':lr})
    opt=torch.optim.AdamW(param_groups, weight_decay=1e-4); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs,eta_min=args.lr*0.05)
    train_ds=ToothSemanticDataset(args.data_dir,split['train'],args.input_h,args.input_w,True); val_ds=ToothSemanticDataset(args.data_dir,split['val'],args.input_h,args.input_w,False); test_ds=ToothSemanticDataset(args.data_dir,split['test'],args.input_h,args.input_w,False)
    train_sampler=DistributedSampler(train_ds, shuffle=True) if distributed else None; val_sampler=DistributedSampler(val_ds, shuffle=False) if distributed else None; test_sampler=DistributedSampler(test_ds, shuffle=False) if distributed else None
    train_loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=train_sampler is None,sampler=train_sampler,num_workers=args.workers,pin_memory=True,drop_last=True,persistent_workers=args.workers>0)
    val_loader=DataLoader(val_ds,batch_size=args.batch_size,shuffle=False,sampler=val_sampler,num_workers=args.workers,pin_memory=True,persistent_workers=args.workers>0)
    test_loader=DataLoader(test_ds,batch_size=args.batch_size,shuffle=False,sampler=test_sampler,num_workers=args.workers,pin_memory=True,persistent_workers=args.workers>0)
    scaler=torch.amp.GradScaler('cuda', enabled=device.type=='cuda')
    if rank == 0: print(f'Experiment {args.name}: arch={args.arch} loss={args.loss} world={world} batch={args.batch_size} params={sum(p.numel() for p in (model.module if hasattr(model,"module") else model).parameters())}', flush=True)
    best=-1; bad=0; rows=[]
    for ep in range(1,args.epochs+1):
        if train_sampler is not None: train_sampler.set_epoch(ep)
        tr=run_epoch(model, train_loader, criterion, opt, scaler, device, True, rank, distributed)
        va=run_epoch(model, val_loader, criterion, opt, scaler, device, False, rank, distributed)
        sched.step(); stop=False
        if rank == 0:
            row={'epoch':ep, **{f'train_{k}':v for k,v in tr.items()}, **{f'val_{k}':v for k,v in va.items()}, 'lr':sched.get_last_lr()[0]}
            rows.append(row)
            with open(out_dir/'log.csv','w',newline='') as f:
                wr=csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            state=(model.module if hasattr(model,'module') else model).state_dict(); torch.save(state, out_dir/'latest.pth')
            if va['dice'] > best:
                best=va['dice']; bad=0; torch.save(state, out_dir/'best.pth'); print(f'=> best val dice {best:.5f}', flush=True)
            else:
                bad += 1
            stop = bad >= args.patience
            if stop: print(f'=> early stop patience={args.patience}', flush=True)
        if distributed:
            flag=torch.tensor([1 if stop else 0],device=device); dist.broadcast(flag,src=0); stop=bool(flag.item())
        if stop: break
    if distributed: dist.barrier()
    # Evaluate the real best checkpoint, not the final/latest weights.
    base=model.module if hasattr(model,'module') else model
    if rank == 0:
        base.load_state_dict(torch.load(out_dir/'best.pth', map_location=device)); base.to(device)
    if distributed: dist.barrier()
    test_metrics=run_epoch(model, test_loader, criterion, opt, scaler, device, False, rank, distributed)
    if rank == 0:
        (out_dir/'test_metrics.json').write_text(json.dumps(test_metrics,indent=2), encoding='utf-8')
        vis_loader=DataLoader(test_ds,batch_size=args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=True,persistent_workers=args.workers>0)
        export_visuals(base, vis_loader, out_dir/'visualizations_individual', device, 16)
        print('TEST_BEST', json.dumps(test_metrics, ensure_ascii=False), flush=True)
    if distributed: dist.destroy_process_group()

if __name__=='__main__': main()
