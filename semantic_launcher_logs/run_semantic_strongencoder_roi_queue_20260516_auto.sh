#!/usr/bin/env bash
set -u
ROOT=/media/zdp1/Datas1/cly/U-KAN-main/U-KAN-main/Seg_UKAN
PY=/home/zdp1/anaconda3/envs/umamba/bin/python
cd "$ROOT" || exit 1
mkdir -p semantic_launcher_logs
MASTER_PORT_BASE=29640

echo "[$(date '+%F %T')] Queue started: strong encoder + ROI high-resolution semantic segmentation" | tee -a semantic_launcher_logs/run_semantic_strongencoder_roi_queue_20260516_auto.log

echo "[$(date '+%F %T')] Waiting for current UKAN ProposedXL final queue to finish..." | tee -a semantic_launcher_logs/run_semantic_strongencoder_roi_queue_20260516_auto.log
while pgrep -af "run_ukan_proposedxl_final_20260516_091523.sh|train_ukan_tooth_semantic_ddp_v2.py.*I_ukan_globallite" | grep -v grep >/dev/null; do
  echo "[$(date '+%F %T')] Current UKAN queue still running; sleep 10 min." | tee -a semantic_launcher_logs/run_semantic_strongencoder_roi_queue_20260516_auto.log
  sleep 600
done

echo "[$(date '+%F %T')] Current queue finished or not found. Starting new route experiments." | tee -a semantic_launcher_logs/run_semantic_strongencoder_roi_queue_20260516_auto.log

run_exp() {
  local script=$1; shift
  local name=$1; shift
  local model=$1; shift
  local encoder=$1; shift
  local batch=$1; shift
  local port=$1; shift
  local extra="$@"
  local outdir="outputs_semantic_ddp/${name}"
  if [ -f "${outdir}/test_metrics.json" ]; then
    echo "[$(date '+%F %T')] SKIP ${name}: test_metrics.json exists" | tee -a semantic_launcher_logs/run_semantic_strongencoder_roi_queue_20260516_auto.log
    return 0
  fi
  echo "[$(date '+%F %T')] RUN ${name}: ${model}/${encoder}, batch=${batch}, script=${script}, extra=${extra}" | tee -a semantic_launcher_logs/run_semantic_strongencoder_roi_queue_20260516_auto.log
  set +e
  CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 --master_port=$port "$script" \
    --model "$model" --encoder "$encoder" --encoder_weights imagenet --loss BoundaryFocalTverskyLoss \
    --name "$name" --epochs 260 --patience 45 --batch_size "$batch" --workers 4 \
    --input_h 512 --input_w 1024 --lr 8e-5 --seed "$((20260530 + port))" $extra
  local code=$?
  set -e
  if [ $code -ne 0 ]; then
    echo "[$(date '+%F %T')] FAIL ${name} exit=${code}" | tee -a semantic_launcher_logs/run_semantic_strongencoder_roi_queue_20260516_auto.log
    if [ "$batch" -gt 2 ]; then
      local rb=$((batch-1))
      local rname="${name}_retry_b${rb}"
      echo "[$(date '+%F %T')] RETRY ${name} as ${rname}, batch=${rb}" | tee -a semantic_launcher_logs/run_semantic_strongencoder_roi_queue_20260516_auto.log
      CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 --master_port=$((port+100)) "$script" \
        --model "$model" --encoder "$encoder" --encoder_weights imagenet --loss BoundaryFocalTverskyLoss \
        --name "$rname" --epochs 260 --patience 45 --batch_size "$rb" --workers 4 \
        --input_h 512 --input_w 1024 --lr 8e-5 --seed "$((20260630 + port))" $extra
      code=$?
      echo "[$(date '+%F %T')] RETRY_DONE ${rname} exit=${code}" | tee -a semantic_launcher_logs/run_semantic_strongencoder_roi_queue_20260516_auto.log
    fi
  else
    echo "[$(date '+%F %T')] DONE ${name}" | tee -a semantic_launcher_logs/run_semantic_strongencoder_roi_queue_20260516_auto.log
  fi
  return 0
}

# Route 1: strong pretrained encoder, full panoramic image, high resolution.
run_exp train_smp_tooth_semantic_ddp.py K_strong_unetpp_effb4_512x1024_b3_e260_boundary UnetPlusPlus timm-efficientnet-b4 3 $((MASTER_PORT_BASE+1))
run_exp train_smp_tooth_semantic_ddp.py L_strong_deeplab_effb4_512x1024_b4_e260_boundary DeepLabV3Plus timm-efficientnet-b4 4 $((MASTER_PORT_BASE+2))
run_exp train_smp_tooth_semantic_ddp.py M_strong_fpn_mitb2_512x1024_b4_e260_boundary FPN mit_b2 4 $((MASTER_PORT_BASE+3))
run_exp train_smp_tooth_semantic_ddp.py N_strong_unetpp_resnest50d_512x1024_b2_e260_boundary UnetPlusPlus timm-resnest50d 2 $((MASTER_PORT_BASE+4))

# Route 2: deterministic dental-arch ROI crop + high-resolution fine segmentation.
run_exp train_smp_tooth_semantic_roi_ddp.py O_roi_unetpp_effb4_512x1024_b3_e260_boundary UnetPlusPlus timm-efficientnet-b4 3 $((MASTER_PORT_BASE+5)) --roi_crop --crop_x1 0.02 --crop_y1 0.18 --crop_x2 0.98 --crop_y2 0.92
run_exp train_smp_tooth_semantic_roi_ddp.py P_roi_fpn_mitb2_512x1024_b4_e260_boundary FPN mit_b2 4 $((MASTER_PORT_BASE+6)) --roi_crop --crop_x1 0.02 --crop_y1 0.18 --crop_x2 0.98 --crop_y2 0.92
run_exp train_smp_tooth_semantic_roi_ddp.py Q_roi_deeplab_effb4_512x1024_b4_e260_boundary DeepLabV3Plus timm-efficientnet-b4 4 $((MASTER_PORT_BASE+7)) --roi_crop --crop_x1 0.02 --crop_y1 0.18 --crop_x2 0.98 --crop_y2 0.92

echo "[$(date '+%F %T')] Queue finished." | tee -a semantic_launcher_logs/run_semantic_strongencoder_roi_queue_20260516_auto.log
