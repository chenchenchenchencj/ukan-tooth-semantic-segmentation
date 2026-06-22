#!/usr/bin/env bash
set -euo pipefail
cd '/media/zdp1/Datas1/cly/U-KAN-main/U-KAN-main/Seg_UKAN'
PY=/home/zdp1/anaconda3/envs/umamba/bin/python
run_arch() {
  local NAME=$1; local ARCH=$2; local BATCH=$3; local WORKERS=$4
  if [ -f outputs_semantic_ddp/${NAME}/test_metrics.json ]; then echo [SKIP] $NAME; return 0; fi
  echo [START] $NAME arch=$ARCH batch=$BATCH workers=$WORKERS $(date)
  CUDA_VISIBLE_DEVICES=0,1,2,3 $PY -m torch.distributed.run --standalone --nproc_per_node=4 \
    train_ukan_tooth_semantic_ddp_v2.py --arch $ARCH --loss BCEDiceLoss --name $NAME \
    --epochs 220 --patience 35 --batch_size $BATCH --workers $WORKERS --input_h 320 --input_w 640 \
    --input_list 128,160,256 --lr 1e-4 --kan_lr 1e-3 --seed 20260517
  echo [DONE] $NAME $(date)
}
run_smp() {
  local NAME=$1; local MODEL=$2; local ENCODER=$3; local BATCH=$4; local WORKERS=$5
  if [ -f outputs_semantic_ddp/${NAME}/test_metrics.json ]; then echo [SKIP] $NAME; return 0; fi
  echo [START] $NAME model=$MODEL encoder=$ENCODER batch=$BATCH workers=$WORKERS $(date)
  CUDA_VISIBLE_DEVICES=0,1,2,3 $PY -m torch.distributed.run --standalone --nproc_per_node=4 \
    train_smp_tooth_semantic_ddp.py --model $MODEL --encoder $ENCODER --encoder_weights imagenet \
    --loss BCEDiceLoss --name $NAME --epochs 220 --patience 35 --batch_size $BATCH --workers $WORKERS \
    --input_h 320 --input_w 640 --lr 1e-4 --seed 20260518
  echo [DONE] $NAME $(date)
}
run_arch SOTA_UNetPP_b8_e220 NestedUNetPP 8 3
run_arch SOTA_DeepLabV3PlusLite_b12_e220 DeepLabV3PlusLite 12 5
run_arch SOTA_SegFormerMini_b12_e220 SegFormerMini 12 5
run_smp SMP_UnetPP_resnet34_b8_e220 UnetPlusPlus resnet34 8 3
run_smp SMP_DeepLabV3Plus_resnet34_b8_e220 DeepLabV3Plus resnet34 8 3
run_smp SMP_FPN_resnet34_b12_e220 FPN resnet34 12 5
run_smp SMP_PAN_resnet34_b12_e220 PAN resnet34 12 5
echo [ALL DONE] semantic resume queue $(date)
