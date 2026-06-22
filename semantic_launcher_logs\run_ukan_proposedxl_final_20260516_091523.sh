#!/usr/bin/env bash
set -euo pipefail
cd /media/zdp1/Datas1/cly/U-KAN-main/U-KAN-main/Seg_UKAN
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/zdp1/anaconda3/envs/umamba/bin/python
run_exp() {
  local name=$1 arch=$2 loss=$3 batch=$4 workers=$5 input_list=$6 epochs=$7 patience=$8 seed=$9
  echo [START] $name $(date)
  if [ -f outputs_semantic_ddp/${name}/test_metrics.json ]; then
    echo [SKIP] $name already has test_metrics.json
    return 0
  fi
  $PY -m torch.distributed.run --standalone --nproc_per_node=4 train_ukan_tooth_semantic_ddp_v2.py \
    --arch $arch --loss $loss --name $name \
    --epochs $epochs --patience $patience --batch_size $batch --workers $workers \
    --input_h 320 --input_w 640 --input_list $input_list \
    --lr 8e-5 --kan_lr 8e-4 --seed $seed
  echo [DONE] $name $(date)
}
# Final heavy model first. If it finishes, run module-only variants with the same heavy channels for ablation/follow-up.
run_exp J_ukan_proposedxl_full_e260_b4_edim160_boundary UKAN_ProposedXL BoundaryFocalTverskyLoss 4 4 160,192,320 260 45 20260519
run_exp G_ukan_weg_e220_b5_edim160_boundary UKAN_WEG BoundaryFocalTverskyLoss 5 4 160,192,320 220 35 20260520
run_exp H_ukan_lka_e220_b5_edim160_boundary UKAN_LKA BoundaryFocalTverskyLoss 5 4 160,192,320 220 35 20260521
run_exp I_ukan_globallite_e220_b5_edim160_boundary UKAN_GlobalLite BoundaryFocalTverskyLoss 5 4 160,192,320 220 35 20260522
echo [ALL DONE] proposedxl queue $(date)
