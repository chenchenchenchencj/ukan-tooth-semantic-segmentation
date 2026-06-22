#!/usr/bin/env bash
set -euo pipefail
cd '/media/zdp1/Datas1/cly/U-KAN-main/U-KAN-main/Seg_UKAN'
PY=/home/zdp1/anaconda3/envs/umamba/bin/python
run_exp() {
  local NAME=$1; local ARCH=$2; local LOSS=$3
  if [ -f outputs_semantic_ddp/${NAME}/test_metrics.json ]; then
    echo [SKIP] $NAME already has test_metrics.json
    return 0
  fi
  echo [START] $NAME arch=$ARCH loss=$LOSS script=train_ukan_tooth_semantic_ddp_v2.py $(date)
  CUDA_VISIBLE_DEVICES=0,1,2,3 $PY -m torch.distributed.run --standalone --nproc_per_node=4 \
    train_ukan_tooth_semantic_ddp_v2.py \
    --arch $ARCH \
    --loss $LOSS \
    --name $NAME \
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
  echo [DONE] $NAME $(date)
}
run_exp D_egms_boundary_full_e220_b12_fair UKAN_EGMS BoundaryBCEDiceLoss
run_exp C_egms_bcedice_e220_b12_fair UKAN_EGMS BCEDiceLoss
run_exp B_msag_bcedice_e220_b12_fair UKAN_MSAG BCEDiceLoss
run_exp A_ukan_bcedice_base_e220_b12_fair UKAN BCEDiceLoss
echo [ALL DONE] remaining b12 D/C/B/A queue $(date)
