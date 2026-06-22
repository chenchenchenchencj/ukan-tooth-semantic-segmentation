#!/usr/bin/env bash
set -euo pipefail
cd '/media/zdp1/Datas1/cly/U-KAN-main/U-KAN-main/Seg_UKAN'
PY=/home/zdp1/anaconda3/envs/umamba/bin/python
run_exp() {
  local NAME=$1; local ARCH=$2; local BATCH=${3:-12}; local WORKERS=${4:-5}
  if [ -f outputs_semantic_ddp/${NAME}/test_metrics.json ]; then
    echo [SKIP] $NAME already completed
    return 0
  fi
  echo [START] $NAME arch=$ARCH batch=$BATCH workers=$WORKERS $(date)
  set +e
  CUDA_VISIBLE_DEVICES=0,1,2,3 $PY -m torch.distributed.run --standalone --nproc_per_node=4 \
    train_ukan_tooth_semantic_ddp_v2.py \
    --arch $ARCH \
    --loss BCEDiceLoss \
    --name $NAME \
    --epochs 220 \
    --patience 35 \
    --batch_size $BATCH \
    --workers $WORKERS \
    --input_h 320 \
    --input_w 640 \
    --input_list 128,160,256 \
    --lr 1e-4 \
    --kan_lr 1e-3 \
    --seed 20260517
  rc=$?
  set -e
  if [ $rc -ne 0 ]; then
    echo [FAIL] $NAME rc=$rc $(date)
    if [ $BATCH -gt 10 ]; then
      echo [RETRY] $NAME with batch=10 workers=3
      run_exp $NAME $ARCH 10 3
    else
      return $rc
    fi
  else
    echo [DONE] $NAME $(date)
  fi
}
run_exp SOTA_PlainUNet_b12_e220 PlainUNet 12 5
run_exp SOTA_ResUNet_b12_e220 ResUNet 12 5
run_exp SOTA_UNetPP_b12_e220 NestedUNetPP 12 5
run_exp SOTA_DeepLabV3PlusLite_b12_e220 DeepLabV3PlusLite 12 5
run_exp SOTA_SegFormerMini_b12_e220 SegFormerMini 12 5
echo [ALL DONE] semantic sota base screen $(date)
