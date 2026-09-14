# CASIA2 training and All8 evaluation — 2026-09-14

The user authorized one additional training experiment on the other free GPU, changing the training dataset to `/data1/data/datasets/CASIA2.0` and using the eight named test roots in [benchmark_all8.json](../config/benchmark_all8.json). The user also requested one evaluation of the original GIT10K run's saved best checkpoint on this same suite after training. The later correction explicitly requires using the existing checkpoint file directly; no alias is created.

## Training arm: CASIA2-ALL8-V1

- Configuration: `config/experiments/casia2_all8.yaml`, inheriting `config/reproduction.yaml`.
- Initialization: fresh `paper-aligned-v1` with the same verified ImageNet PVTv2-b2 initialization. No trained GIT10K weights are reused for training.
- Training population: all 5,123 `Tp` images paired bijectively with `Gt` masks after removing the mask's `_gt` suffix. Authentic `Au` images are not added, preserving the original tampered-image-only training convention. No internal training holdout or extra seed is introduced.
- Unchanged: architecture, loss, 352px input, batch 6, 100 epochs, seed 42, FP32, AdamW LR 0.001, weight decay, cosine schedule, synchronized horizontal flip, workers, sampling, normalization and postprocessing. The launcher compares resolved configurations and rejects non-data parameter drift.
- Each epoch uses the existing diagnostic path on the concatenated All8 population, once per image. `model-best.pt` continues to minimize **pooled per-image MAE** across 4,295 images; it is not selected by F1, by equal-dataset macro averaging, or by separately choosing a checkpoint for each dataset.
- After 100 epochs, retain `model-final.pt` and the existing recovery checkpoints, then run one explicit All8 report using `model-best.pt`. These eight-set best scores are **test-selected** and must not be presented as performance on held-out selection data. Trainer fixed-final status fields describe the retained training endpoint; the requested benchmark report explicitly identifies the best-MAE checkpoint.

## Original GIT10K arm: one additional external evaluation

The running `DCDSDIFF-GIT10K-RECON-20260914-A` keeps its original source, training, diagnostic population, checkpoint policy and fixed-final GIT10K evaluation. A separate detached server follow-up waits for the existing controller to report COMPLETED and exit. It then acquires that run's GPU 0 and run locks and evaluates the saved `model-best.pt` once on All8.

This checkpoint remains selected by the original reconstructed GIT10K Mix 1,000-image MAE. All8 is not used to reselect it. The additional result is distinct from the original final99 reproduction result and from the CASIA2 arm's All8-selected best result. No original training process is restarted to install the follow-up.

## Frozen benchmark population

| Dataset | Images | Mask pairing |
|---|---:|---|
| Casiav1 | 920 | remove `_gt` |
| Columbia | 180 | remove `_gt` |
| NIST16 | 564 | exact stem |
| IMD2020 | 2010 | remove `_mask` |
| DSO-1 | 100 | remove `_gt` |
| wild | 201 | exact stem |
| coverage | 100 | exact stem |
| Korus | 220 | exact stem |
| Total | 4295 | each sample once |

The snapshot is `data/casia2-all8-v1`, with dataset-prefixed names preventing collisions. Original images and masks are read through links; binary 0/1 masks, if present, receive an explicitly recorded 0/255 conversion in the new snapshot. Other soft labels retain the baseline /255 convention. Original-resolution GT, independent 352px image/mask resizing for training, and the baseline detail/HFVG generation algorithm are preserved. Source masks/images are never edited.

The manifest binds raw paths, file hashes, decoded RGB hashes, dimensions, mask conversion and auxiliary hashes. Preparation records exact duplicates and anomalies; the formal launcher rejects exact CASIA2-train/All8-test RGB overlap. This does not prove absence of transformed duplicates or shared source-image content. Actual f/m/d/t file membership and hashes are checked before formal training/resume and test-file hashes before final benchmark evaluation.

The independent full raw-image audit found zero exact decoded-RGB overlaps for both CASIA2 training versus All8 and original GIT10K training versus All8. See [overlap audit](all8_overlap_audit.json). This check does not address near duplicates.

## Inference, reports and control

Evaluation uses the existing temporal postprocessing, 10 sampling steps, seed 0 and threshold >0.5. F1, IoU and MAE are calculated per image at original GT dimensions. Reports include eight individual dataset rows, their equal-dataset macro mean, and the image-weighted pooled mean. Empty/empty F1 and IoU remain 1. Checkpoint hashes, selection epoch/population, source commits, manifest hashes and input verification accompany every result.

New controllers use a GPU lock and a run lock, inherited by child processes. The legacy global lock is attributed to its actual holder's GPU so GPU 1 can run alongside GPU 0 safely. Training and evaluation failure releases owned processes and records the error; it does not silently change the protocol or retry with another checkpoint. Only small receipts/reports belong in Git. Temporary validation artifacts are removed after validation; weights and full runtime outputs remain on the server.
