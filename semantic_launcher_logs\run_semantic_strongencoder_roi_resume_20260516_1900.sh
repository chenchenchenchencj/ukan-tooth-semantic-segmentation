#!/usr/bin/env bash
set -u
ROOT=/media/zdp1/Datas1/cly/U-KAN-main/U-KAN-main/Seg_UKAN
PY=/home/zdp1/anaconda3/envs/umamba/bin/python
cd "$ROOT" || exit 1
LOG=semantic_launcher_logs/run_semantic_strongencoder_roi_resume_20260516_1900.log
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
    --input_h 512 --input_w 1024 --lr 8e-5 --seed "$((20260700+port))" $extra
  local code=$?
  echo "[$(date '+%F %T')] EXIT ${name}: ${code}" | tee -a "$LOG"
  if [ $code -ne 0 ] && [ "$batch" -gt 2 ]; then
    local rb=$((batch-1)); local rname="${name}_retry_b${rb}"
    echo "[$(date '+%F %T')] RETRY ${rname}: b=${rb}" | tee -a "$LOG"
    CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 --master_port=$((port+50)) "$script" \
      --model "$model" --encoder "$encoder" --encoder_weights imagenet --loss BoundaryFocalTverskyLoss \
      --name "$rname" --epochs 260 --patience 45 --batch_size "$rb" --workers 4 \
      --input_h 512 --input_w 1024 --lr 8e-5 --seed "$((20260800+port))" $extra
    echo "[$(date '+%F %T')] RETRY_EXIT ${rname}: $?" | tee -a "$LOG"
  fi
}
echo "[$(date '+%F %T')] Resume queue started" | tee -a "$LOG"
run_exp train_smp_tooth_semantic_ddp.py K_strong_unetpp_effb4_512x1024_b3_e260_boundary UnetPlusPlus timm-efficientnet-b4 3 29701
run_exp train_smp_tooth_semantic_ddp.py L_strong_deeplab_effb4_512x1024_b4_e260_boundary DeepLabV3Plus timm-efficientnet-b4 4 29702
run_exp train_smp_tooth_semantic_ddp.py M_strong_fpn_mitb2_512x1024_b4_e260_boundary FPN mit_b2 4 29703
run_exp train_smp_tooth_semantic_ddp.py N_strong_unetpp_resnest50d_512x1024_b2_e260_boundary UnetPlusPlus timm-resnest50d 2 29704
run_exp train_smp_tooth_semantic_roi_ddp.py O_roi_unetpp_effb4_512x1024_b3_e260_boundary UnetPlusPlus timm-efficientnet-b4 3 29705 --roi_crop --crop_x1 0.02 --crop_y1 0.18 --crop_x2 0.98 --crop_y2 0.92
run_exp train_smp_tooth_semantic_roi_ddp.py P_roi_fpn_mitb2_512x1024_b4_e260_boundary FPN mit_b2 4 29706 --roi_crop --crop_x1 0.02 --crop_y1 0.18 --crop_x2 0.98 --crop_y2 0.92
run_exp train_smp_tooth_semantic_roi_ddp.py Q_roi_deeplab_effb4_512x1024_b4_e260_boundary DeepLabV3Plus timm-efficientnet-b4 4 29707 --roi_crop --crop_x1 0.02 --crop_y1 0.18 --crop_x2 0.98 --crop_y2 0.92
echo "[$(date '+%F %T')] Resume queue finished" | tee -a "$LOG"