
import argparse, csv, json, os, random, time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import archs
import losses


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class ToothSemanticDataset(Dataset):
    def __init__(self, data_dir, ids, h=320, w=640, train=False):
        self.data_dir = Path(data_dir)
        self.ids = ids
        self.h = h; self.w = w; self.train = train
        self.img_dir = self.data_dir / 'busi' / 'images'
        self.mask_dir = self.data_dir / 'busi' / 'masks' / '0'
        if not self.mask_dir.exists():
            self.mask_dir = self.data_dir / 'busi' / 'masks'

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img = cv2.imread(str(self.img_dir / f'{img_id}.png'), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.mask_dir / f'{img_id}_mask.png'), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        if self.train:
            if random.random() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1]); mask = np.ascontiguousarray(mask[:, ::-1])
            if random.random() < 0.5:
                img = np.ascontiguousarray(img[::-1, :]); mask = np.ascontiguousarray(mask[::-1, :])
            if random.random() < 0.25:
                alpha = 0.85 + random.random() * 0.3
                beta = random.randint(-12, 12)
                img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        mask = (mask > 0).astype(np.float32)[None]
        img = img.transpose(2,0,1)
        return torch.from_numpy(img), torch.from_numpy(mask), img_id


def make_split(data_dir, split_file, seed=20260515):
    data_dir = Path(data_dir)
    split_file = Path(split_file)
    if split_file.exists():
        return json.loads(split_file.read_text(encoding='utf-8'))
    ids = sorted([p.stem for p in (data_dir/'busi'/'images').glob('*.png')])
    rng = random.Random(seed); rng.shuffle(ids)
    n = len(ids); n_val = int(round(n * 0.10)); n_test = int(round(n * 0.10))
    split = {'train': ids[:n-n_val-n_test], 'val': ids[n-n_val-n_test:n-n_test], 'test': ids[n-n_test:]}
    split_file.parent.mkdir(parents=True, exist_ok=True)
    split_file.write_text(json.dumps(split, indent=2, ensure_ascii=False), encoding='utf-8')
    return split


def dice_iou_from_logits(logits, target, thr=0.5):
    prob = torch.sigmoid(logits)
    pred = (prob > thr).float()
    dims = (1,2,3)
    inter = (pred * target).sum(dims)
    ps = pred.sum(dims); ts = target.sum(dims)
    dice = ((2*inter + 1e-6) / (ps + ts + 1e-6)).mean().item()
    iou = ((inter + 1e-6) / (ps + ts - inter + 1e-6)).mean().item()
    prec = ((inter + 1e-6) / (ps + 1e-6)).mean().item()
    rec = ((inter + 1e-6) / (ts + 1e-6)).mean().item()
    return dice, iou, prec, rec


def run_epoch(model, loader, criterion, opt, scaler, device, train):
    model.train(train)
    total_loss = total_dice = total_iou = total_p = total_r = n = 0
    pbar = tqdm(loader, dynamic_ncols=True, leave=False)
    for img, mask, _ in pbar:
        img = img.to(device, non_blocking=True); mask = mask.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                out = model(img)
                loss = criterion(out, mask)
            if train:
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt); scaler.update()
        d,i,p,r = dice_iou_from_logits(out.detach(), mask)
        bs = img.size(0); n += bs
        total_loss += float(loss.detach()) * bs; total_dice += d * bs; total_iou += i * bs; total_p += p * bs; total_r += r * bs
        pbar.set_postfix(loss=total_loss/n, dice=total_dice/n, iou=total_iou/n)
    return {'loss': total_loss/n, 'dice': total_dice/n, 'iou': total_iou/n, 'precision': total_p/n, 'recall': total_r/n}


@torch.no_grad()
def export_visuals(model, loader, out_dir, device, limit=12):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    model.eval(); count = 0
    for img, mask, ids in loader:
        img = img.to(device); mask = mask.to(device)
        pred = (torch.sigmoid(model(img)) > 0.5).float()
        for b in range(img.size(0)):
            if count >= limit: return
            # denormalize
            arr = img[b].detach().cpu().numpy().transpose(1,2,0)
            mean = np.array([0.485,0.456,0.406]); std=np.array([0.229,0.224,0.225])
            arr = np.clip((arr*std+mean)*255,0,255).astype(np.uint8)
            gt = (mask[b,0].detach().cpu().numpy()*255).astype(np.uint8)
            pr = (pred[b,0].detach().cpu().numpy()*255).astype(np.uint8)
            cv2.imwrite(str(out_dir / f'{ids[b]}__image.jpg'), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out_dir / f'{ids[b]}__gt.png'), gt)
            cv2.imwrite(str(out_dir / f'{ids[b]}__pred.png'), pr)
            overlap = arr.copy(); overlap[gt>0] = (0.5*overlap[gt>0] + np.array([0,255,0])*0.5).astype(np.uint8); overlap[pr>0] = (0.5*overlap[pr>0] + np.array([255,0,0])*0.5).astype(np.uint8)
            cv2.imwrite(str(out_dir / f'{ids[b]}__overlap.jpg'), cv2.cvtColor(overlap, cv2.COLOR_RGB2BGR))
            count += 1


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--arch', default='UKAN')
    ap.add_argument('--loss', default='BCEDiceLoss')
    ap.add_argument('--name', required=True)
    ap.add_argument('--data_dir', default='inputs')
    ap.add_argument('--out_dir', default='outputs_semantic_batch')
    ap.add_argument('--split_file', default='splits/tooth_semantic_80_10_10_seed20260515.json')
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--input_h', type=int, default=320)
    ap.add_argument('--input_w', type=int, default=640)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--kan_lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=20260515)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--no_kan', action='store_true')
    ap.add_argument('--input_list', default='128,160,256')
    args=ap.parse_args()
    seed_all(args.seed)
    out_dir=Path(args.out_dir)/args.name; out_dir.mkdir(parents=True, exist_ok=True)
    split=make_split(args.data_dir, args.split_file, args.seed)
    (out_dir/'config.json').write_text(json.dumps(vars(args),indent=2,ensure_ascii=False), encoding='utf-8')
    (out_dir/'split_counts.json').write_text(json.dumps({k:len(v) for k,v in split.items()},indent=2), encoding='utf-8')
    embed=[int(x) for x in args.input_list.split(',')]
    device=torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model=archs.__dict__[args.arch](1,3,False,embed_dims=embed,no_kan=args.no_kan).to(device)
    criterion=losses.__dict__[args.loss]().to(device)
    param_groups=[]
    for n,p in model.named_parameters():
        lr=args.kan_lr if ('layer' in n.lower() and 'fc' in n.lower()) else args.lr
        param_groups.append({'params':p,'lr':lr})
    opt=torch.optim.AdamW(param_groups, weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr*0.05)
    train_ds=ToothSemanticDataset(args.data_dir, split['train'], args.input_h, args.input_w, True)
    val_ds=ToothSemanticDataset(args.data_dir, split['val'], args.input_h, args.input_w, False)
    test_ds=ToothSemanticDataset(args.data_dir, split['test'], args.input_h, args.input_w, False)
    train_loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,num_workers=4,pin_memory=True,drop_last=True)
    val_loader=DataLoader(val_ds,batch_size=args.batch_size,shuffle=False,num_workers=4,pin_memory=True)
    test_loader=DataLoader(test_ds,batch_size=args.batch_size,shuffle=False,num_workers=4,pin_memory=True)
    scaler=torch.cuda.amp.GradScaler(enabled=device.type=='cuda')
    rows=[]; best=-1
    print(f'Experiment {args.name}: arch={args.arch} loss={args.loss} split={len(split["train"])}/{len(split["val"])}/{len(split["test"])} params={sum(p.numel() for p in model.parameters())}')
    for ep in range(1,args.epochs+1):
        tr=run_epoch(model, train_loader, criterion, opt, scaler, device, True)
        va=run_epoch(model, val_loader, criterion, opt, scaler, device, False)
        sched.step()
        row={'epoch':ep, **{f'train_{k}':v for k,v in tr.items()}, **{f'val_{k}':v for k,v in va.items()}, 'lr':sched.get_last_lr()[0]}
        rows.append(row); pd.DataFrame(rows).to_csv(out_dir/'log.csv', index=False)
        print(json.dumps(row, ensure_ascii=False))
        torch.save(model.state_dict(), out_dir/'latest.pth')
        if va['dice'] > best:
            best=va['dice']; torch.save(model.state_dict(), out_dir/'best.pth'); print(f'=> best val dice {best:.5f}')
    model.load_state_dict(torch.load(out_dir/'best.pth', map_location=device)); model.to(device)
    te=run_epoch(model, test_loader, criterion, opt, scaler, device, False)
    (out_dir/'test_metrics.json').write_text(json.dumps(te,indent=2), encoding='utf-8')
    export_visuals(model, test_loader, out_dir/'visualizations_individual', device, 16)
    print('TEST', json.dumps(te, ensure_ascii=False))

if __name__=='__main__':
    main()
