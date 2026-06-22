#!/usr/bin/env python3
import json, subprocess, os
from pathlib import Path
ROOT=Path('/media/zdp1/Datas1/cly/U-KAN-main/U-KAN-main/Seg_UKAN')
PYTHON='/home/zdp1/anaconda3/envs/umamba/bin/python'
IMG='/home/zdp1/external_datasets/childrens_dental_adult_tooth_semantic/images'
MSK='/home/zdp1/external_datasets/childrens_dental_adult_tooth_semantic/masks'
OUTROOT=ROOT/'outputs_semantic_ddp/children_adult_external_eval_selected_20260522'
OUTROOT.mkdir(parents=True, exist_ok=True)
exps=[
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
thresholds='0.2,0.3,0.4,0.5,0.6,0.7,0.8'
procs=[]
for i,exp in enumerate(exps):
    out=OUTROOT/exp
    if (out/'extended_metrics.json').exists():
        print('[SKIP]',exp, flush=True)
        continue
    gpu=str(i%4)
    env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=gpu
    cmd=[PYTHON,'eval_semantic_extended_metrics.py','--exp',f'outputs_semantic_ddp/{exp}', '--image_dir',IMG,'--mask_dir',MSK,'--mask_suffix','.png','--image_ext','.png','--batch_size','8','--workers','4','--thresholds',thresholds,'--out_dir',str(out)]
    log=open(OUTROOT/f'{exp}.log','w')
    print('[START]',exp,'gpu',gpu, flush=True)
    procs.append((exp,subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT),log))
    if len(procs)>=4:
        exp0,p0,log0=procs.pop(0); rc=p0.wait(); log0.close(); print('[DONE]',exp0,rc,flush=True)
for exp,p,log in procs:
    rc=p.wait(); log.close(); print('[DONE]',exp,rc,flush=True)
rows=[]
for exp in exps:
    f=OUTROOT/exp/'extended_metrics.json'
    if not f.exists(): continue
    data=json.loads(f.read_text())
    thr=str(data['best_by_dice_threshold'])
    s=data[thr]['summary']
    row={'Experiment':exp,'BestThr':thr}
    for k in ['dice','iou','precision','recall','specificity','accuracy','mcc','hd95','assd','boundary_f1']:
        row[k]=s.get(k)
    rows.append(row)
rows.sort(key=lambda r:r.get('dice') or 0, reverse=True)
(OUTROOT/'summary.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8')
md=['# Children Adult Tooth Segmentation External Evaluation','', 'Dataset: Kaggle truthisneverlinear/childrens-dental-panoramic-radiographs-dataset, Adult tooth segmentation / Panoramic radiography database, n=598.', '', '| Experiment | Thr | Dice | IoU | Precision | Recall | Specificity | Accuracy | MCC | HD95 | ASSD | Boundary F1 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
    md.append('| {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |'.format(
        r['Experiment'], r['BestThr'], r['dice'], r['iou'], r['precision'], r['recall'], r['specificity'], r['accuracy'], r['mcc'], r['hd95'], r['assd'], r['boundary_f1']))
(OUTROOT/'summary.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
print(OUTROOT/'summary.md')
