#!/usr/bin/env python3
import json, os, shutil, subprocess, zipfile, time
from pathlib import Path
from PIL import Image
import numpy as np
ROOT=Path('/media/zdp1/Datas1/cly/U-KAN-main/U-KAN-main/Seg_UKAN')
PYTHON='/home/zdp1/anaconda3/envs/umamba/bin/python'
DATA_ROOT=Path('/home/zdp1/external_datasets/mendeley_pixel_tooth_semantic')
ZIP=DATA_ROOT/'mendeley_jrz4nj82zv_v1.zip'
RAW=DATA_ROOT/'raw'
ORG=DATA_ROOT/'organized'
IMG=ORG/'images'
MSK=ORG/'masks'
OUTROOT=ROOT/'outputs_semantic_ddp/mendeley_pixel_tooth_external_eval_20260522'
EXPS=[
 'A_ukan_bcedice_base_e220_b12_fair',
 'B_msag_bcedice_e220_b12_fair',
 'C_egms_bcedice_e220_b12_fair',
 'D_egms_boundary_full_e220_b12_fair',
 'E_egms_augv2_boundaryfocal_e220_b12_fair',
 'F_egms_augv2_bcedice_e220_b12_fair',
 'G_ukan_weg_e220_b5_edim160_boundary',
 'H_ukan_lka_e220_b5_edim160_boundary',
 'I_ukan_globallite_e220_b5_edim160_boundary',
 'J_ukan_proposedxl_full_e260_b4_edim160_boundary',
 'SOTA_PlainUNet_b12_e220',
 'SOTA_ResUNet_b12_e220',
 'SOTA_UNetPP_b8_e220',
 'SOTA_DeepLabV3PlusLite_b12_e220',
 'SOTA_SegFormerMini_b12_e220',
 'SMP_UnetPP_resnet34_b8_e220',
 'SMP_DeepLabV3Plus_resnet34_b8_e220',
 'SMP_FPN_resnet34_b12_e220',
 'SMP_PAN_resnet34_b12_e220',
 'K2_strong_unetpp_effb4_512x1024_b2_e260_boundary',
 'L2_strong_deeplab_effb4_512x1024_b3_e260_boundary',
 'M2_strong_fpn_mitb2_512x1024_b3_e260_boundary',
 'N2_strong_unetpp_resnest50d_512x1024_b2_e260_boundary',
 'O2_roi_unetpp_effb4_512x1024_b2_e260_boundary',
 'P2_roi_fpn_mitb2_512x1024_b3_e260_boundary',
 'Q2_roi_deeplab_effb4_512x1024_b3_e260_boundary',
]

def wait_zip_stable():
    if not ZIP.exists():
        raise FileNotFoundError(ZIP)
    last=-1; stable=0
    while stable<2:
        size=ZIP.stat().st_size
        if size==last:
            stable+=1
        else:
            stable=0; last=size
        time.sleep(15)
    if not zipfile.is_zipfile(ZIP):
        raise RuntimeError(f'Zip not valid yet: {ZIP} size={ZIP.stat().st_size}')

def extract_if_needed():
    RAW.mkdir(parents=True, exist_ok=True)
    marker=RAW/'.extracted.ok'
    if marker.exists(): return
    with zipfile.ZipFile(ZIP) as z:
        z.extractall(RAW)
    marker.write_text('ok')

def looks_mask(p: Path):
    s=p.as_posix().lower()
    return any(x in s for x in ['mask','label','annotation','annotated','segmentation','ground_truth','groundtruth','gt'])

def find_pairs():
    exts={'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}
    files=[p for p in RAW.rglob('*') if p.suffix.lower() in exts]
    masks=[p for p in files if looks_mask(p)]
    imgs=[p for p in files if p not in set(masks)]
    # fallback by parent folder names
    if not masks:
        masks=[p for p in files if 'mask' in p.parent.name.lower() or 'label' in p.parent.name.lower()]
        imgs=[p for p in files if p not in set(masks)]
    def stemkey(p):
        st=p.stem.lower()
        for tok in ['_mask','-mask',' mask','_label','-label',' label','_gt','-gt',' gt','_seg','-seg']:
            st=st.replace(tok,'')
        return ''.join(ch for ch in st if ch.isalnum())
    imgmap={stemkey(p):p for p in imgs}
    pairs=[]
    for m in masks:
        k=stemkey(m)
        if k in imgmap: pairs.append((imgmap[k],m))
    if len(pairs)<10:
        # try sorted same count under likely dirs
        img_dirs=[]; mask_dirs=[]
        for d in {p.parent for p in files}:
            name=d.as_posix().lower()
            if any(x in name for x in ['image','xray','radiograph','opg']): img_dirs.append(d)
            if any(x in name for x in ['mask','label','annotation','gt']): mask_dirs.append(d)
        for idr in img_dirs:
            il=sorted([p for p in idr.glob('*') if p.suffix.lower() in exts])
            for mdr in mask_dirs:
                ml=sorted([p for p in mdr.glob('*') if p.suffix.lower() in exts])
                if len(il)==len(ml) and len(il)>len(pairs): pairs=list(zip(il,ml))
    return pairs, files

def organize():
    IMG.mkdir(parents=True, exist_ok=True); MSK.mkdir(parents=True, exist_ok=True)
    if len(list(IMG.glob('*.png')))>20 and len(list(MSK.glob('*.png')))>20:
        return len(list(IMG.glob('*.png')))
    pairs, files=find_pairs()
    if len(pairs)<10:
        raise RuntimeError('Could not infer image/mask pairs. Files sample: '+ '\n'.join(str(p) for p in files[:80]))
    for i,(im,ma) in enumerate(pairs,1):
        name=f'mendeley_pixel_{i:04d}.png'
        Image.open(im).convert('RGB').save(IMG/name)
        arr=np.array(Image.open(ma).convert('L'))
        arr=(arr>0).astype('uint8')*255
        Image.fromarray(arr).save(MSK/name)
    meta={'source':'Mendeley Data jrz4nj82zv v1, DOI 10.17632/jrz4nj82zv.1','n':len(pairs),'pairs':[(str(a),str(b)) for a,b in pairs[:20]]}
    (ORG/'dataset_meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    return len(pairs)

def run_eval(n):
    OUTROOT.mkdir(parents=True, exist_ok=True)
    thresholds='0.2,0.3,0.4,0.5,0.6,0.7,0.8'
    procs=[]
    for i,exp in enumerate(EXPS):
        out=OUTROOT/exp
        if (out/'extended_metrics.json').exists():
            print('[SKIP]',exp,flush=True); continue
        gpu=str(i%4); env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=gpu
        cmd=[PYTHON,'eval_semantic_extended_metrics.py','--exp',f'outputs_semantic_ddp/{exp}','--image_dir',str(IMG),'--mask_dir',str(MSK),'--mask_suffix','.png','--image_ext','.png','--batch_size','8','--workers','4','--thresholds',thresholds,'--out_dir',str(out)]
        log=open(OUTROOT/f'{exp}.log','w')
        print('[START]',exp,'gpu',gpu,flush=True)
        procs.append((exp,subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT),log))
        if len(procs)>=4:
            exp0,p0,log0=procs.pop(0); rc=p0.wait(); log0.close(); print('[DONE]',exp0,rc,flush=True)
    for exp,p,log in procs:
        rc=p.wait(); log.close(); print('[DONE]',exp,rc,flush=True)
    rows=[]
    for exp in EXPS:
        f=OUTROOT/exp/'extended_metrics.json'
        if not f.exists(): continue
        data=json.loads(f.read_text())
        th=str(data['best_by_dice_threshold']); s=data[th]['summary']
        row={'Experiment':exp,'BestThr':th}
        for k in ['dice','iou','precision','recall','specificity','accuracy','mcc','hd95','assd','boundary_f1']:
            row[k]=s.get(k)
        rows.append(row)
    rows.sort(key=lambda r:r.get('dice') or 0, reverse=True)
    (OUTROOT/'summary.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8')
    md=['# Mendeley Pixel-level Tooth Semantic Segmentation External Evaluation','', 'Dataset: Mendeley Data jrz4nj82zv v1, Dental Panoramic Radiography Dataset for Pixel-level Semantic Segmentation of Teeth, n=%d.'%n, '', '| Experiment | Thr | Dice | IoU | Precision | Recall | Specificity | Accuracy | MCC | HD95 | ASSD | Boundary F1 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        md.append('| {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |'.format(r['Experiment'],r['BestThr'],r['dice'],r['iou'],r['precision'],r['recall'],r['specificity'],r['accuracy'],r['mcc'],r['hd95'],r['assd'],r['boundary_f1']))
    (OUTROOT/'summary.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print(OUTROOT/'summary.md')

if __name__=='__main__':
    wait_zip_stable(); extract_if_needed(); n=organize(); run_eval(n)
