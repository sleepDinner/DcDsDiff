# Active-code audit (2026-09-14)

Scope: local working tree initially based on `d60e4e893ad391a699fe5229a4be0903cfc3cbf3`. The initial uncommitted annotations are preserved where useful. This is a source audit, not a claim of completed training or paper-level equivalence. The same-day historical audit, retained as `docs/history/pre-rebuild-audit.md`, documents older runs and remains historical evidence.

## Reached implementation

`train.py` loads YAML through `utils.init_utils.add_args`, instantiates `model.net.net`, wraps it in `model.SimpleDiffSef.CondGaussianDiffusion`, and trains via `modification_train_val_forward`. Both PVTv2 backbones, MMFF, Decoder1 and MSIE participate. EmptyObject is only a legacy constructor argument. The diffusion objective is x0; a mask structure loss is added to half of detail L1+MSE. Noise is shared between the mask and detail targets. Sampling uses ten sine-spaced steps, tanh x0 estimates, and a majority-consistency time ensemble.

Live upstream source checked at commit [d2ad59fc218727c2148328e73c3d7a9fcb2b8bde](https://github.com/QixianHao/DcDsDiff-and-GIT10K/tree/d2ad59fc218727c2148328e73c3d7a9fcb2b8bde), including train.py, dataset/data_val.py and model/SimpleDiffSef.py. The problems below are inherited unless specifically identified as local tooling.

## Findings before changes

| Priority | Path / behavior | Consequence and required handling |
|---|---|---|
| P1 | `utils/init_env.py` sets CUDA_VISIBLE_DEVICES=0,1 unconditionally | Overrides an assigned GPU and can touch unrelated jobs. Remove device override; enforce one process for the new baseline. |
| P1 | `model/loss.py`, structure_loss uses reduce='none' | Deprecated argument is truthy and yields scalar mean BCE, defeating per-pixel weighting. Correct to reduction='none' and record the mathematical change. |
| P1 | `dataset/data_val.py` independently sorts four folders | Missing/differently named files can silently mismatch modalities. Require exact stem sets and an explicit geometry policy for mismatched dimensions; do not silently intersect datasets. |
| P1 | Trace train/test normalization differs | Train trace is mapped to [-1,1] in diffusion.forward; test trace uses ImageNet normalization and sample performs no corresponding conversion. Freeze and document one consistent representation. |
| P1 | `utils/trainer.py` resume stores only model/epoch/scaler | Optimizer, scheduler, RNG and best score are lost; restart repeats the saved epoch. Require full atomic state and next_epoch; reject legacy checkpoints for exact resume. |
| P1 | `train.py` validates Mix plus two repeated slices | Duplicates reweight selected images. Use each Mix image once. Label MAE-selected checkpoint test-selected; use fixed final checkpoint as primary. |
| P1 | `utils/trainer.py` ensemble MAE lacks rank aggregation | Historical multi-process best selection sees a shard. New baseline must run one process; avoid repeating this behavior. |
| P1 | `train.py` passes fp16 as amp; Accelerator reads another fp16 argument | Mixed precision configuration is inconsistent. Use one supported Accelerator mixed_precision setting. |
| P2 | `utils/init_utils.py` overwrites YAML with every argparse default | Changing YAML epochs/batch can have no effect. Preserve YAML unless a CLI argument was explicitly supplied. |
| P2 | `model/train_val_forward.py` hardcodes 20 history chunks | Non-ten-step inference breaks. Derive mask/detail histories from alternating entries or enforce ten-step protocol. |
| P2 | `utils/trainer.py` val uses pred_de for mask MAE | Non-ensemble evaluation scores the detail head. Use pred_gt or fail unsupported modes; batch ensemble is also incompatible with the active dual-output model. |
| P2 | `utils/trainer.py` global best and validation reseeding | Best leaks across trainer instances, and validation changes training RNG trajectory. Store best per trainer and preserve/restore RNG exactly around evaluation. |
| P2 | `model/net.py` downloads mutable HF main during model construction | Training startup depends on network/cache resolution. Accept a pinned local pretrained asset with checksum provenance. |

Known source/description questions must be decided in the paper protocol: original YAML lr=1e-4; actual dataset augmentation is disabled; spatial attention uses mean+max with 7x7 convolution; detail preprocessing and filename-to-generator mapping are not fully released. Do not infer exact scientific equivalence from runtime success.

## Cleanup assessment

The old `FinalTrainData_352x352.yaml`, dormant `DcDsDiff_384x384.yaml`, test-selected checkpoint scanning workflow, broken legacy sample/validation modes and source-string tests are not suitable entrypoints for the new main experiment. Remove or explicitly archive obsolete entrypoints only after retaining their history/provenance. Tests under `tests/` mostly verify source strings, old absolute paths or pure helper behavior; they do not verify actual model backward, complete checkpoint resume, or dataset/paper equivalence. The user requests removal of temporary test code after server verification; retain compact test receipts instead.

Keep the vendored diffusion package: the active model imports its modified GaussianDiffusion.q_sample and UViT classes. Do not delete apparently unused package files without an import-closure check. Keep the local paper, useful source comments, pretrained asset provenance and historical audit.

## Bounded remediation ownership

The training-path work in this task is restricted to train.py, utils/trainer.py, utils/init_env.py and utils/init_utils.py. It adds a single-process contract, configuration precedence, deterministic evaluation, finite guards, fixed-final primary selection, atomic resumable checkpoints and compact status/metric records. Dataset/model/loss changes and server lifecycle management are separate owned edits and require their own verification. Final runtime results must be recorded separately from this pre-change audit.
