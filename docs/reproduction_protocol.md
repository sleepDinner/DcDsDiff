# GIT10K-PAPER-RECON-V1

Registered before formal training on 2026-09-14. The user explicitly accepted a reconstructed split with all differences recorded. Scope: one fresh main-model training run and its fixed-final evaluation. Ablations, other datasets, extra seeds and checkpoint sweeps are not part of this launch.

## Sources and implementation policy

- [IJCAI 2025 paper](https://www.ijcai.org/proceedings/2025/0120.pdf), especially sections 3.2–3.5 and 4.1.
- [Official source d2ad59fc218727c2148328e73c3d7a9fcb2b8bde](https://github.com/QixianHao/DcDsDiff-and-GIT10K/tree/d2ad59fc218727c2148328e73c3d7a9fcb2b8bde).
- Detailed evidence and unresolved differences: [paper audit](paper_protocol_audit.md), [code audit](code_audit.md).

Where explicit paper descriptions conflict with public code, this run uses the paper: AdamW initial LR 0.001, synchronized random horizontal flip p=0.5, spatial attention with channel max pooling and 3×3 convolution, channel attention with global max pooling and one 1×1 convolution/ReLU/Sigmoid. The channel convolution preserves channel count; unspecified biases follow the public implementation. This architecture is named `paper-aligned-v1`; historical trained checkpoints are incompatible.

## Frozen training and inference

| Item | This run | Basis |
|---|---|---|
| Input / global batch / epochs | 352×352 / 6 / 100 (0–99) | Paper |
| Device / precision | One RTX 4090, one process, FP32 | Reproduction execution choice |
| Optimizer | AdamW LR 0.001, weight decay 0.01, default betas/eps | LR/type paper; other values public/default |
| LR schedule | One cosine step per epoch, eta_min 1e-6 | Public code; paper does not specify |
| Seed | Training 42, evaluation 0 | Fixed reproducibility choice/public seed |
| Backbone | Two independently trainable PVTv2-b2, same ImageNet initialization | Public code |
| Model | RGB/HFVG conditioning, dual mask/detail diffusion, MMFF, MSIE | Paper and public code |
| Loss | Correct per-pixel weighted BCE + weighted IoU; 0.5×(detail L1+MSE) | Paper and public formula; fix `reduce` API bug |
| Augmentation | Joint horizontal flip only | Paper |
| Trace input | Training [0,1] then diffusion maps to [-1,1]; evaluation directly [-1,1] | Documented consistency repair |
| Sampling | 10 nonlinear steps; public temporal mean/minmax/positive-logit majority postprocessing | Steps paper; detailed postprocessing public code |
| Endpoint | Final epoch 99 only | Predeclared local policy; paper omits selection details |
| Metrics | Per-image F1 and IoU, then mean; threshold >0.5; empty/empty=1; MAE secondary | Explicit local convention; paper omits details |

All inputs and soft training masks use PIL bilinear resize. Original-size GT is used for final evaluation and predictions are resized back with bilinear interpolation. Pixel labels use /255 scaling and threshold 0.5. Unequal image/mask dimensions are independently mapped onto the common 352×352 grid; no source files are changed. This cannot prove perfect geometric correspondence of the released annotations.

## Dataset reconstruction

Original source: `/data0/hl/DcDsDiff-and-GIT10K/GIT10K/{Image,Mask}`. Require exactly 10,000 matching PNG stems. Natural-sort stems, group identical decoded RGB images, shuffle those groups using Python `Random(42)`, and assign whole groups to training until 9,000 images are filled; remaining 1,000 are test. This fixes the historical 8,996/1,004 count but does **not** reconstruct the unavailable official 4×250 test lists. Report actual filename prefixes without assuming BN/RBN or EI/Flux/e/t/z mappings.

Before any formal training, the first ungrouped shuffle was rejected by the leakage gate: 26 two-image duplicate groups existed, and 3 crossed the candidate split. The group-bound policy fixes this without deleting samples or inspecting model scores. Raw data include 9,974 distinct RGB images, 1,627 image/mask size mismatches, 7,516 soft masks and one empty mask. All are retained. Auxiliary files from the rejected candidate were reused only after exact pixel equality with regeneration in the isolated pinned environment. Its rejection receipt is retained in `docs/rejected_split_receipt.json`; redundant candidate files were removed after verification.

For every sample retain input-file hashes, decoded RGB hash, dimensions, soft/empty-mask flags and generated d/t hashes. Reject exact decoded-RGB duplicates across the train/test split. This detects exact content leakage, not semantic/source-image relationships or transformed near duplicates.

Read-only symlinks reuse original f/m data. Rebuild auxiliary d/t in the new project because the author provides no exact generation program: detail uses radius-15 linear distance decay from a 3-pixel elliptical morphological boundary; HFVG uses a circular Fourier high-pass radius `0.5×min(H,W)/2`, absolute real inverse transform, ×10 enhancement, clipping and PNG quantization. These are explicit reconstruction assumptions, not proven pixel-equivalent author targets.

## Reporting and recovery

Every epoch evaluates each Mix image once for diagnostic MAE; no duplicate subset expansion. The lowest-MAE checkpoint is labelled `TEST_SELECTED_MAE_DIAGNOSTIC` and is not the primary result. The final evaluator never searches checkpoints, thresholds or seeds. Runtime success and completed training are reported separately from agreement with paper scores.

Full epoch-boundary checkpoints store model, optimizer, scheduler, scaler, RNG states, completed epoch, next epoch, metrics history and configuration contract. The current epoch may be replayed after interruption. New initialization is required when protocol or architecture changes. Keep last, diagnostic best, every 10th epoch and fixed final; model-99 is a hard link to final.

All command-line entrypoints restart Python with `-s` and set `PYTHONNOUSERSITE=1` before importing project dependencies. This excludes the server user's unrelated `~/.local` packages. Launcher checks the exact environment freeze against the verified READY receipt before executing.

Scientific completion requires 100 complete epochs, final evaluation covering exactly 1,000 images, finite metrics and `controller_status.json` = `COMPLETED`. A training startup receipt alone is not a completed reproduction.
