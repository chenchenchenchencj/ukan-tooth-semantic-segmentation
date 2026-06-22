#!/usr/bin/env bash
set -u
ROOT='/media/zdp1/Datas1/cly/U-KAN-main/U-KAN-main/Seg_UKAN'
cd "$ROOT" || exit 1
ENV=/home/zdp1/anaconda3/envs/umamba/bin/python
mkdir -p semantic_launcher_logs outputs_semantic_ddp
export CUDA_VISIBLE_DEVICES=0,1,2,3
unset PYTORCH_CUDA_ALLOC_CONF
export OMP_NUM_THREADS=1
COMMON="--data_dir inputs --out_dir outputs_semantic_ddp --epochs 160 --patience 25 --batch_size 13 --input_h 320 --input_w 640 --lr 1e-4 --kan_lr 1e-3 --input_list 128,160,256 --seed 20260515"
run_exp() {
  NAME="$1"; ARCH="$2"; LOSS="$3"; PORT="$4"
  echo "============================================================"
  echo "START $NAME arch=$ARCH loss=$LOSS batch_per_gpu=13 global_batch=52 time=$(date '+%F %T')"
  echo "============================================================"
  if [ -f "outputs_semantic_ddp/$NAME/test_metrics.json" ]; then
    echo "SKIP $NAME because test_metrics.json already exists"
    return 0
  fi
  $ENV -m torch.distributed.run --nproc_per_node=4 --master_port="$PORT" train_ukan_tooth_semantic_ddp.py     --name "$NAME" --arch "$ARCH" --loss "$LOSS" $COMMON
  RC=$?
  echo "END $NAME rc=$RC time=$(date '+%F %T')"
  if [ "$RC" -ne 0 ]; then exit "$RC"; fi
}
run_exp D_egms_boundary_full_e160_b13_retry UKAN_EGMS BoundaryBCEDiceLoss 30711
run_exp C_egms_bcedice_e160_b13_retry UKAN_EGMS BCEDiceLoss 30712
run_exp B_msag_bcedice_e160_b13_retry UKAN_MSAG BCEDiceLoss 30713
run_exp A_ukan_bcedice_base_e160_b13_retry UKAN BCEDiceLoss 30714
echo "ALL DONE time=$(date '+%F %T')"
