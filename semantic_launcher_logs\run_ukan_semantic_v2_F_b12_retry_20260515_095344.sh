#!/usr/bin/env bash
set -euo pipefail
cd '/media/zdp1/Datas1/cly/U-KAN-main/U-KAN-main/Seg_UKAN'
PY=/home/zdp1/anaconda3/envs/umamba/bin/python
NAME=F_egms_augv2_bcedice_e220_b12_retry_after_sigkill
if [ -f "outputs_semantic_ddp/${NAME}/test_metrics.json" ]; then
  echo "[SKIP] $NAME already done"
  exit 0
fi
echo "[START] $NAME $(date)"
CUDA_VISIBLE_DEVICES=0,1,2,3 $PY -m torch.distributed.run --standalone --nproc_per_node=4 \
  train_ukan_tooth_semantic_ddp_v2.py \
  --arch UKAN_EGMS \
  --loss BCEDiceLoss \
  --name "$NAME" \
  --epochs 220 \
  --patience 35 \
  --batch_size 12 \
  --workers 5 \
  --input_h 320 \
  --input_w 640 \
  --input_list 128,160,256 \
  --lr 1e-4 \
  --kan_lr 1e-3 \
  --seed 20260516
echo "[DONE] $NAME $(date)"
