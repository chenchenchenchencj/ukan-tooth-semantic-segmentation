#!/usr/bin/env bash
set -euo pipefail
cd '/media/zdp1/Datas1/cly/U-KAN-main/U-KAN-main/Seg_UKAN'
PY=/home/zdp1/anaconda3/envs/umamba/bin/python
wait_for_sota_queue() {
  echo [WAIT] waiting for self-implemented SOTA queue to finish $(date)
  while pgrep -af 'run_semantic_sota_base_screen_b12|train_ukan_tooth_semantic_ddp_v2.py' >/dev/null; do
    sleep 180
  done
  echo [WAIT DONE] $(date)
}
run_smp() {
  local NAME=$1; local MODEL=$2; local ENCODER=$3; local BATCH=${4:-12}
  if [ -f outputs_semantic_ddp/${NAME}/test_metrics.json ]; then
    echo [SKIP] $NAME already completed
    return 0
  fi
  echo [START] $NAME model=$MODEL encoder=$ENCODER batch=$BATCH $(date)
  set +e
  CUDA_VISIBLE_DEVICES=0,1,2,3 $PY -m torch.distributed.run --standalone --nproc_per_node=4 \
    train_smp_tooth_semantic_ddp.py \
    --model $MODEL \
    --encoder $ENCODER \
    --encoder_weights imagenet \
    --loss BCEDiceLoss \
    --name $NAME \
    --epochs 220 \
    --patience 35 \
    --batch_size $BATCH \
    --workers 5 \
    --input_h 320 \
    --input_w 640 \
    --lr 1e-4 \
    --seed 20260518
  rc=$?
  set -e
  if [ $rc -ne 0 ]; then
    echo [FAIL] $NAME rc=$rc $(date)
    if [ $BATCH -gt 8 ]; then
      echo [RETRY] $NAME with batch=8 workers=3
      CUDA_VISIBLE_DEVICES=0,1,2,3 $PY -m torch.distributed.run --standalone --nproc_per_node=4 \
        train_smp_tooth_semantic_ddp.py \
        --model $MODEL --encoder $ENCODER --encoder_weights imagenet --loss BCEDiceLoss --name $NAME \
        --epochs 220 --patience 35 --batch_size 8 --workers 3 --input_h 320 --input_w 640 --lr 1e-4 --seed 20260518
    else
      return $rc
    fi
  fi
  echo [DONE] $NAME $(date)
}
wait_for_sota_queue
run_smp SMP_UnetPP_resnet34_b12_e220 UnetPlusPlus resnet34 12
run_smp SMP_DeepLabV3Plus_resnet34_b12_e220 DeepLabV3Plus resnet34 12
run_smp SMP_FPN_resnet34_b12_e220 FPN resnet34 12
run_smp SMP_PAN_resnet34_b12_e220 PAN resnet34 12
echo [ALL DONE] SMP official base queue $(date)
