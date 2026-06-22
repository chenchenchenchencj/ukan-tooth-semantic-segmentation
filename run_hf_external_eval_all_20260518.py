#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, subprocess, time
from pathlib import Path

ROOT = Path('/media/zdp1/Datas1/cly/U-KAN-main/U-KAN-main/Seg_UKAN')
PY = '/home/zdp1/anaconda3/envs/umamba/bin/python'
EVAL = ROOT / 'eval_semantic_extended_metrics.py'
IMAGE_DIR = '/home/zdp1/external_datasets/hf_panoramic_xray_tooth/X-RAY/X-ray image'
MASK_DIR = '/home/zdp1/external_datasets/hf_panoramic_xray_tooth/X-RAY/masks_machine'

EXPS = [
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

def safe_batch(config):
    h = int(config.get('input_h', 320)); w = int(config.get('input_w', 640))
    return 2 if h * w >= 512 * 1024 else 4

def main():
    summary = []
    for name in EXPS:
        exp = ROOT / 'outputs_semantic_ddp' / name
        cfg = exp / 'config.json'
        ckpt = exp / 'best.pth'
        if not cfg.exists() or not ckpt.exists():
            print('[SKIP] {}: missing config or best.pth'.format(name), flush=True)
            summary.append({'name': name, 'status': 'missing'})
            continue
        out_dir = exp / 'extended_eval_hf_external_machine'
        out_file = out_dir / 'extended_metrics.json'
        if out_file.exists():
            print('[SKIP] {}: exists'.format(name), flush=True)
        else:
            config = json.loads(cfg.read_text(encoding='utf-8'))
            bs = safe_batch(config)
            cmd = [
                PY, str(EVAL), '--exp', str(exp), '--batch_size', str(bs), '--workers', '4',
                '--image_dir', IMAGE_DIR, '--mask_dir', MASK_DIR,
                '--image_ext', '.jpg', '--mask_suffix', '.png', '--out_dir', str(out_dir),
            ]
            print('[RUN] {} batch={}'.format(name, bs), flush=True)
            t0 = time.time()
            ret = subprocess.run(cmd, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            print(ret.stdout, flush=True)
            if ret.returncode != 0:
                print('[FAIL] {} returncode={}'.format(name, ret.returncode), flush=True)
                summary.append({'name': name, 'status': 'failed', 'returncode': ret.returncode})
                continue
            print('[DONE] {} {:.1f}s'.format(name, time.time() - t0), flush=True)
        try:
            data = json.loads(out_file.read_text(encoding='utf-8'))
            thr = str(data['best_by_dice_threshold'])
            s = data[thr]['summary']
            row = {'name': name, 'status': 'ok', 'best_thr': float(thr)}
            for k in ['dice','iou','precision','recall','specificity','accuracy','mcc','hd95','assd','boundary_f1']:
                row[k] = s.get(k)
            summary.append(row)
        except Exception as e:
            summary.append({'name': name, 'status': 'parse_error', 'error': str(e)})
    out = ROOT / 'outputs_semantic_ddp' / 'hf_external_all_experiments_summary_20260518.json'
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    md = ROOT / 'outputs_semantic_ddp' / 'hf_external_all_experiments_summary_20260518.md'
    lines = ['# HF external generalization evaluation summary (masks_machine)', '', '| Experiment | Dice | IoU | Precision | Recall | Specificity | Accuracy | MCC | HD95 | ASSD | Boundary F1 |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        if r.get('status') != 'ok':
            lines.append('| {} | {} |  |  |  |  |  |  |  |  |  |'.format(r['name'], r.get('status')))
        else:
            lines.append('| {name} | {dice:.4f} | {iou:.4f} | {precision:.4f} | {recall:.4f} | {specificity:.4f} | {accuracy:.4f} | {mcc:.4f} | {hd95:.4f} | {assd:.4f} | {boundary_f1:.4f} |'.format(**r))
    md.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('[SUMMARY] {}'.format(out), flush=True)
    print('[SUMMARY] {}'.format(md), flush=True)

if __name__ == '__main__':
    main()
