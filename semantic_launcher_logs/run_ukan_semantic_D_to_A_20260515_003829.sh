#!/usr/bin/env bash
set -u
cd '/media/zdp1/Datas1/cly/U-KAN-main/U-KAN-main/Seg_UKAN' || exit 1
PY=/home/zdp1/anaconda3/envs/umamba/bin/python
PORT_BASE=30450
run_one() {
  local idx="$1"; local name="$2"; local arch="$3"; local loss="$4"; local port=$((PORT_BASE + idx))
  local out="outputs_semantic_ddp/${name}"
  echo "[START] $name arch=$arch loss=$loss port=$port $(date)"
  if [ -f "$out/test_metrics.json" ]; then
    echo "[SKIP] $name already has test_metrics.json"
    return 0
  fi
  CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m torch.distributed.run --nproc_per_node=4 --master_port="$port" \
    train_ukan_tooth_semantic_ddp.py \
    --name "$name" --arch "$arch" --loss "$loss" \
    --data_dir inputs --out_dir outputs_semantic_ddp \
    --epochs 160 --patience 25 --batch_size 8 \
    --input_h 320 --input_w 640 --lr 1e-4 --kan_lr 1e-3 \
    --input_list 128,160,256 --seed 20260515
  local code=$?
  echo "[END] $name code=$code $(date)"
  return $code
}
run_one 1 "D_egms_boundary_full_e160" "UKAN_EGMS" "BoundaryBCEDiceLoss" || exit $?
run_one 2 "C_egms_bcedice_e160" "UKAN_EGMS" "BCEDiceLoss" || exit $?
run_one 3 "B_msag_bcedice_e160" "UKAN_MSAG" "BCEDiceLoss" || exit $?
run_one 4 "A_ukan_bcedice_base_e160" "UKAN" "BCEDiceLoss" || exit $?
echo "[ALL DONE] $(date)"
