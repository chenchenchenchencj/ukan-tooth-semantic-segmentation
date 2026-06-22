# U-KAN Tooth Semantic Segmentation Experiments

This repository contains the cleaned research code used for 2D panoramic tooth semantic segmentation experiments in a master's thesis workflow.

## Main thesis model

The recommended thesis mainline is:

`I_ukan_globallite_e220_b5_edim160_boundary`

Internal test metrics:

- Dice: 0.9150
- IoU: 0.8502
- Precision: 0.9113
- Recall: 0.9225

This is the best result among the UKAN-based proposed variants. Strong encoder baselines such as `N2_strong_unetpp_resnest50d_512x1024_b2_e260_boundary` are included as comparison baselines.

## Repository contents

- `archs.py`, `kan.py`: network definitions and UKAN variants.
- `losses.py`: BCE-Dice, boundary/focal-style losses used in experiments.
- `dataset.py`: tooth semantic segmentation dataset loader.
- `train_ukan_tooth_semantic_ddp_v2.py`: main distributed training entry for UKAN variants.
- `train_smp_tooth_semantic_ddp.py`: SMP comparison baselines.
- `train_smp_tooth_semantic_roi_ddp.py`: ROI high-resolution comparison route.
- `eval_semantic_extended_metrics.py`: extended metric evaluation.
- `semantic_launcher_logs/*.sh`: reproducibility launch scripts.
- `results/`: selected small metric files and summaries only.

Large checkpoints, datasets, prediction images, and generated visualization packages are intentionally excluded.

## Notes

The original experiment directory on the lab server contained many exploratory runs and large artifacts. This repository is a cleaned code snapshot for paper writing and reproducibility.
