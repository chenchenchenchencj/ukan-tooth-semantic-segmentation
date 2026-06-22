import argparse, json, os, random
from pathlib import Path
import cv2, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
import archs

class ToothDataset(Dataset):
    def __init__(self, data_dir, names, input_h=320, input_w=640):
        self.data_dir=Path(data_dir); self.names=list(names); self.input_h=input_h; self.input_w=input_w
    def __len__(self): return len(self.names)
    def __getitem__(self, idx):
        name=str(self.names[idx])
        base = name if name.endswith('.png') else name + '.png'
        stem = base[:-4] if base.endswith('.png') else base
        ip=self.data_dir/'busi'/'images'/base
        mp=self.data_dir/'busi'/'masks'/'0'/(stem + '_mask.png')
        img=cv2.imread(str(ip), cv2.IMREAD_COLOR)
        mask=cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None: raise FileNotFoundError(f'{ip} | {mp}')
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img=cv2.resize(img,(self.input_w,self.input_h),interpolation=cv2.INTER_LINEAR)
        mask=cv2.resize(mask,(self.input_w,self.input_h),interpolation=cv2.INTER_NEAREST)
        img=img.astype(np.float32)/255.0
        img=(img-np.array([0.485,0.456,0.406],np.float32))/np.array([0.229,0.224,0.225],np.float32)
        mask=(mask>127).astype(np.float32)
        img=torch.from_numpy(img.transpose(2,0,1))
        mask=torch.from_numpy(mask[None])
        return img,mask,name

def global_metrics(probs, masks, th):
    p=(probs>=th).astype(np.uint8); g=(masks>0.5).astype(np.uint8)
    inter=(p & g).sum(dtype=np.float64)
    ps=p.sum(dtype=np.float64); gs=g.sum(dtype=np.float64)
    union=(p | g).sum(dtype=np.float64)
    dice=(2*inter+1e-7)/(ps+gs+1e-7)
    iou=(inter+1e-7)/(union+1e-7)
    prec=(inter+1e-7)/(ps+1e-7)
    rec=(inter+1e-7)/(gs+1e-7)
    return dict(dice=float(dice), iou=float(iou), precision=float(prec), recall=float(rec), pred_pixels=float(ps), gt_pixels=float(gs))

def infer(args, split_name):
    split=json.load(open(args.split_file, encoding='utf-8'))
    names=split[split_name]
    ds=ToothDataset(args.data_dir,names,args.input_h,args.input_w)
    dl=DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=True)
    model=getattr(archs,args.arch)(1,3,False,embed_dims=[int(x) for x in args.input_list.split(',')])
    ck=torch.load(args.ckpt,map_location='cpu')
    model.load_state_dict(ck['model'] if isinstance(ck,dict) and 'model' in ck else ck, strict=False)
    model.cuda().eval()
    probs=[]; masks=[]
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
        for img,mask,_ in dl:
            out=model(img.cuda(non_blocking=True))
            prob=torch.sigmoid(out).float().cpu().numpy()
            probs.append(prob); masks.append(mask.numpy())
    return np.concatenate(probs,0), np.concatenate(masks,0)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True); ap.add_argument('--arch', required=True)
    ap.add_argument('--data_dir', default='inputs'); ap.add_argument('--split_file', default='splits/tooth_semantic_80_10_10_seed20260515.json')
    ap.add_argument('--input_h', type=int, default=320); ap.add_argument('--input_w', type=int, default=640); ap.add_argument('--input_list', default='128,160,256')
    ap.add_argument('--batch_size', type=int, default=8); ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--out', required=True)
    args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    vp,vm=infer(args,'val'); tp,tm=infer(args,'test')
    thresholds=[round(x,2) for x in np.arange(0.05,0.96,0.05)] + [0.975,0.99]
    val=[]; test=[]
    for th in thresholds:
        val.append({'threshold':th,**global_metrics(vp,vm,th)})
        test.append({'threshold':th,**global_metrics(tp,tm,th)})
    best=max(val,key=lambda x:x['dice']); best_th=best['threshold']
    best_test=global_metrics(tp,tm,best_th); best_test['threshold']=best_th
    fixed=global_metrics(tp,tm,0.5); fixed['threshold']=0.5
    res={'best_val':best,'test_at_best_val_threshold':best_test,'test_at_0.5':fixed,'val_curve':val,'test_curve':test}
    (out/'threshold_sweep.json').write_text(json.dumps(res,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(res['best_val'],ensure_ascii=False))
    print('TEST_BEST_TH', json.dumps(best_test,ensure_ascii=False))
    print('TEST_0.5', json.dumps(fixed,ensure_ascii=False))
if __name__=='__main__': main()
