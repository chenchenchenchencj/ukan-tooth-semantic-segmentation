#!/usr/bin/env bash
set -u
ROOT=/media/zdp1/Datas1/cly/U-KAN-main/U-KAN-main/Seg_UKAN
PY=/home/zdp1/anaconda3/envs/umamba/bin/python
cd "$ROOT" || exit 1
LOG=semantic_launcher_logs/run_semantic_strongencoder_roi_patched_20260516_1915.log
mkdir -p semantic_launcher_logs
run_exp() {
  local script=$1; shift
  local name=$1; shift
  local model=$1; shift
  local encoder=$1; shift
  local batch=$1; shift
  local port=$1; shift
  local extra="$@"
  local outdir="outputs_semantic_ddp/${name}"
  if [ -f "${outdir}/test_metrics.json" ]; then echo "[$(date '+%F %T')] SKIP ${name}" | tee -a "$LOG"; return 0; fi
  echo "[$(date '+%F %T')] RUN ${name}: ${model}/${encoder}, b=${batch}, ${script}, ${extra}" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 --master_port=$port "$script" \
    --model "$model" --encoder "$encoder" --encoder_weights imagenet --loss BoundaryFocalTverskyLoss \
    --name "$name" --epochs 260 --patience 45 --batch_size "$batch" --workers 4 \
    --input_h 512 --input_w 1024 --lr 8e-5 --seed "$((20260900+port))" $extra
  local code=$?
  echo "[$(date '+%F %T')] EXIT ${name}: ${code}" | tee -a "$LOG"
  return 0
}
echo "[$(date '+%F %T')] Patched queue started" | tee -a "$LOG"
run_exp train_smp_tooth_semantic_ddp.py K2_strong_unetpp_effb4_512x1024_b2_e260_boundary UnetPlusPlus timm-efficientnet-b4 2 29801
run_exp train_smp_tooth_semantic_ddp.py L2_strong_deeplab_effb4_512x1024_b3_e260_boundary DeepLabV3Plus timm-efficientnet-b4 3 29802
run_exp train_smp_tooth_semantic_ddp.py M2_strong_fpn_mitb2_512x1024_b3_e260_boundary FPN mit_b2 3 29803
run_exp train_smp_tooth_semantic_ddp.py N2_strong_unetpp_resnest50d_512x1024_b2_e260_boundary UnetPlusPlus timm-resnest50d 2 29804
run_exp train_smp_tooth_semantic_roi_ddp.py O2_roi_unetpp_effb4_512x1024_b2_e260_boundary UnetPlusPlus timm-efficientnet-b4 2 29805 --roi_crop --crop_x1 0.02 --crop_y1 0.18 --crop_x2 0.98 --crop_y2 0.92
run_exp train_smp_tooth_semantic_roi_ddp.py P2_roi_fpn_mitb2_512x1024_b3_e260_boundary FPN mit_b2 3 29806 --roi_crop --crop_x1 0.02 --crop_y1 0.18 --crop_x2 0.98 --crop_y2 0.92
run_exp train_smp_tooth_semantic_roi_ddp.py Q2_roi_deeplab_effb4_512x1024_b3_e260_boundary DeepLabV3Plus timm-efficientnet-b4 3 29807 --roi_crop --crop_x1 0.02 --crop_y1 0.18 --crop_x2 0.98 --crop_y2 0.92
echo "[$(date '+%F %T')] Patched queue finished" | tee -a "$LOG"