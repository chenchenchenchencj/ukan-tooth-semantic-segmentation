#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extended binary semantic segmentation evaluation for dental panoramic masks.

Outputs image-level and dataset-level metrics useful for thesis reporting:
Dice, IoU, precision, recall, specificity, accuracy, balanced accuracy,
MCC, HD95, ASSD/ASD, and boundary F1. Supports both custom UKAN models and
segmentation_models_pytorch models used in the screening experiments.
"""
import argparse, json, math, random
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import ndimage as ndi
from torch.utils.data import Dataset, DataLoader


def load_split(data_dir, split_file, seed=20260515):
    split_file = Path(split_file)
    if split_file.exists():
        return json.loads(split_file.read_text(encoding='utf-8'))
    ids = sorted([p.stem for p in (Path(data_dir)/'busi'/'images').glob('*.png')])
    rng = random.Random(seed); rng.shuffle(ids)
    n=len(ids); nv=int(round(n*0.1)); nt=int(round(n*0.1))
    split={'train':ids[:n-nv-nt], 'val':ids[n-nv-nt:n-nt], 'test':ids[n-nt:]}
    split_file.parent.mkdir(parents=True, exist_ok=True)
    split_file.write_text(json.dumps(split, indent=2, ensure_ascii=False), encoding='utf-8')
    return split


class ToothSemanticDataset(Dataset):
    def __init__(self, data_dir, ids=None, h=320, w=640, image_dir=None, mask_dir=None, image_ext='.png', mask_suffix='_mask.png'):
        self.data_dir=Path(data_dir); self.h=h; self.w=w
        self.img_dir=Path(image_dir) if image_dir else self.data_dir/'busi'/'images'
        self.mask_dir=Path(mask_dir) if mask_dir else self.data_dir/'busi'/'masks'/'0'
        if not self.mask_dir.exists(): self.mask_dir=self.data_dir/'busi'/'masks'
        self.image_ext=image_ext; self.mask_suffix=mask_suffix
        if ids is None:
            self.ids=sorted([p.stem for p in self.img_dir.glob('*'+self.image_ext)])
        else:
            self.ids=list(ids)
    def __len__(self): return len(self.ids)
    def __getitem__(self, idx):
        img_id=self.ids[idx]
        img_path=self.img_dir/f'{img_id}{self.image_ext}'
        mask_path=self.mask_dir/f'{img_id}{self.mask_suffix}'
        if not mask_path.exists():
            mask_path=self.mask_dir/f'{img_id}.png'
        img=cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None: raise FileNotFoundError(img_path)
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask=cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None: raise FileNotFoundError(mask_path)
        img=cv2.resize(img,(self.w,self.h),interpolation=cv2.INTER_LINEAR)
        mask=cv2.resize(mask,(self.w,self.h),interpolation=cv2.INTER_NEAREST)
        img=img.astype(np.float32)/255.0
        img=(img-np.array([0.485,0.456,0.406],np.float32))/np.array([0.229,0.224,0.225],np.float32)
        mask=(mask>0).astype(np.uint8)
        return torch.from_numpy(img.transpose(2,0,1)), torch.from_numpy(mask[None].astype(np.float32)), img_id


def build_model_from_config(config):
    if 'arch' in config:
        import archs
        embed=[int(x) for x in str(config.get('input_list','128,160,256')).split(',')]
        return archs.__dict__[config['arch']](1,3,False,embed_dims=embed,no_kan=bool(config.get('no_kan',False)))
    import segmentation_models_pytorch as smp
    kwargs=dict(encoder_name=config.get('encoder','resnet34'), encoder_weights=None, in_channels=3, classes=1)
    name=config.get('model','UnetPlusPlus')
    if name=='Unet': return smp.Unet(**kwargs)
    if name=='UnetPlusPlus': return smp.UnetPlusPlus(**kwargs)
    if name=='DeepLabV3Plus': return smp.DeepLabV3Plus(**kwargs)
    if name=='FPN': return smp.FPN(**kwargs)
    if name=='PAN': return smp.PAN(**kwargs)
    raise ValueError(name)


def safe_div(a,b):
    return float(a / b) if b != 0 else 0.0


def surface_distances(pred, gt):
    pred=pred.astype(bool); gt=gt.astype(bool)
    if pred.sum()==0 and gt.sum()==0:
        return np.array([0.], dtype=np.float32), np.array([0.], dtype=np.float32)
    if pred.sum()==0 or gt.sum()==0:
        diag=float(np.hypot(*pred.shape))
        return np.array([diag], dtype=np.float32), np.array([diag], dtype=np.float32)
    struct=ndi.generate_binary_structure(2,1)
    pred_border=pred ^ ndi.binary_erosion(pred, structure=struct, border_value=0)
    gt_border=gt ^ ndi.binary_erosion(gt, structure=struct, border_value=0)
    dt_gt=ndi.distance_transform_edt(~gt_border)
    dt_pred=ndi.distance_transform_edt(~pred_border)
    d1=dt_gt[pred_border]; d2=dt_pred[gt_border]
    if d1.size==0: d1=np.array([0.], dtype=np.float32)
    if d2.size==0: d2=np.array([0.], dtype=np.float32)
    return d1.astype(np.float32), d2.astype(np.float32)


def boundary_f1(pred, gt, tolerance=2):
    pred=pred.astype(bool); gt=gt.astype(bool)
    if pred.sum()==0 and gt.sum()==0: return 1.0
    if pred.sum()==0 or gt.sum()==0: return 0.0
    struct=ndi.generate_binary_structure(2,1)
    pb=pred ^ ndi.binary_erosion(pred, structure=struct, border_value=0)
    gb=gt ^ ndi.binary_erosion(gt, structure=struct, border_value=0)
    if pb.sum()==0 or gb.sum()==0: return 0.0
    pb_dil=ndi.binary_dilation(pb, structure=struct, iterations=tolerance)
    gb_dil=ndi.binary_dilation(gb, structure=struct, iterations=tolerance)
    p=safe_div((pb & gb_dil).sum(), pb.sum())
    r=safe_div((gb & pb_dil).sum(), gb.sum())
    return safe_div(2*p*r, p+r)


def image_metrics(prob, gt, thr=0.5):
    pred=(prob>=thr); gt=gt.astype(bool)
    tp=int((pred & gt).sum()); fp=int((pred & ~gt).sum()); fn=int((~pred & gt).sum()); tn=int((~pred & ~gt).sum())
    dice=safe_div(2*tp, 2*tp+fp+fn); iou=safe_div(tp, tp+fp+fn)
    precision=safe_div(tp, tp+fp); recall=safe_div(tp, tp+fn); specificity=safe_div(tn, tn+fp)
    accuracy=safe_div(tp+tn, tp+tn+fp+fn); balanced_accuracy=(recall+specificity)/2.0
    npv=safe_div(tn, tn+fn); fpr=safe_div(fp, fp+tn); fnr=safe_div(fn, fn+tp)
    denom=math.sqrt(max((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn), 1))
    mcc=float(((tp*tn)-(fp*fn))/denom) if denom else 0.0
    d1,d2=surface_distances(pred, gt); all_d=np.concatenate([d1,d2])
    return dict(dice=dice,iou=iou,precision=precision,recall=recall,specificity=specificity,accuracy=accuracy,balanced_accuracy=balanced_accuracy,npv=npv,fpr=fpr,fnr=fnr,mcc=mcc,hd95=float(np.percentile(all_d,95)),hd=float(np.max(all_d)),assd=float((d1.mean()+d2.mean())/2.0),asd=float(all_d.mean()),boundary_f1=boundary_f1(pred,gt,2),tp=tp,fp=fp,fn=fn,tn=tn)


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--exp', required=True); ap.add_argument('--ckpt', default='best.pth')
    ap.add_argument('--data_dir', default=None); ap.add_argument('--split_file', default=None); ap.add_argument('--split', default='test')
    ap.add_argument('--input_h', type=int, default=None); ap.add_argument('--input_w', type=int, default=None)
    ap.add_argument('--batch_size', type=int, default=4); ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--thresholds', default='0.3,0.4,0.5,0.6,0.7'); ap.add_argument('--out_dir', default=None)
    ap.add_argument('--image_dir', default=None); ap.add_argument('--mask_dir', default=None); ap.add_argument('--image_ext', default='.png'); ap.add_argument('--mask_suffix', default='_mask.png')
    args=ap.parse_args(); exp=Path(args.exp); config=json.loads((exp/'config.json').read_text(encoding='utf-8'))
    data_dir=args.data_dir or config.get('data_dir','inputs'); split_file=args.split_file or config.get('split_file','splits/tooth_semantic_80_10_10_seed20260515.json')
    h=args.input_h or int(config.get('input_h',320)); w=args.input_w or int(config.get('input_w',640))
    if args.image_dir and args.mask_dir:
        ds=ToothSemanticDataset(data_dir, None, h, w, args.image_dir, args.mask_dir, args.image_ext, args.mask_suffix)
    else:
        split=load_split(data_dir, split_file, int(config.get('seed',20260515))); ds=ToothSemanticDataset(data_dir, split[args.split], h, w)
    loader=DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=True)
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'); model=build_model_from_config(config).to(device)
    model.load_state_dict(torch.load(exp/args.ckpt, map_location=device), strict=True); model.eval()
    thresholds=[float(x) for x in args.thresholds.split(',') if x.strip()]; per_thr={thr:[] for thr in thresholds}
    for img,mask,ids in loader:
        prob=torch.sigmoid(model(img.to(device, non_blocking=True))).detach().cpu().numpy()[:,0]; gt=mask.numpy()[:,0]
        for b in range(prob.shape[0]):
            for thr in thresholds:
                m=image_metrics(prob[b], gt[b], thr); m['img_id']=ids[b]; per_thr[thr].append(m)
    out={}
    for thr, rows in per_thr.items():
        keys=[k for k in rows[0] if k!='img_id']; summary={k:float(np.mean([r[k] for r in rows])) for k in keys}; summary['n_images']=len(rows)
        out[str(thr)]={'summary':summary,'per_image':rows}
    best_thr=max(thresholds, key=lambda t: out[str(t)]['summary']['dice']); out['best_by_dice_threshold']=best_thr
    out_dir=Path(args.out_dir) if args.out_dir else exp/'extended_eval'; out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir/'extended_metrics.json').write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'exp':str(exp),'ckpt':args.ckpt,'n_images':len(ds),'best_thr':best_thr,'best_summary':out[str(best_thr)]['summary']}, indent=2, ensure_ascii=False))

if __name__=='__main__': main()
