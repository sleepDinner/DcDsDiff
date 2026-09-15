# CASIA2: user-requested early endpoint and All8 evaluation

On 2026-09-15 the user requested ending GPU 1 training early and evaluating all eight datasets. This supersedes the remaining training endpoint of `DCDSDIFF-CASIA2-SEL3-ALL8-20260914-B` only. Evaluation protocol: `CASIA2-SEL3-ALL8-EARLYSTOP-V1`.

- Training stopped cleanly at **2026-09-15 14:13:12 Asia/Shanghai**. Controller/trainer 3903032/3903033 exited; GPU 1 and run locks were released.
- Completed epochs: **0–84 (85 total)**, including the three inherited All8-monitored parent epochs. The last complete checkpoint has next_epoch 85 and global_step 72590. Epoch 85 was interrupted after the last observed step 72794; its partial updates are excluded.
- Existing `model-best.pt`: **epoch 52**, selected by pooled Casiav1/Columbia/NIST16 MAE **0.10169355626924069**. SHA-256: `a280c3d7db017d9af81fe5aa18e754024de95115dc7aaae7519ce9902bbef2c5`. No checkpoint alias or new selection is introduced.
- Training source remains commit `01db26c96581818499120240dc01ffaa5e71e3a9`; frozen source/config/checkpoints/history are retained. The original `training_status.json` is the final live progress snapshot; `controller_status.json=INTERRUPTED` and `early_stop.json=EARLY_STOPPED_BY_USER` establish the actual endpoint. Do not label this as 100-epoch completion or a final99 result.
- All8 remains 4,295 images: Casiav1 920, Columbia 180, NIST16 564, IMD2020 2010, DSO-1 100, wild 201, coverage 100, Korus 220. Fixed seed 0, 10 sampling steps, threshold >0.5, batch 6, 352px network input and original-size metrics. Report per-image F1/IoU/MAE averaged per dataset, plus equal-dataset macro and equal-image pooled means.
- The three selection datasets remain test-selected. The other five were reporting-only after the continuation boundary; the parent's prior All8 exposure remains disclosed. Early stopping was user-requested, not an automated MAE stopping rule.

The existing detached follow-up controller freezes an evaluation-only source snapshot, binds the stopped status/config/history and original best/last checkpoint hashes, and holds GPU 1/run/follow-up locks during inference. Its source checks preserve the exact frozen scientific Python implementation. Normal evaluations still require 100 completed epochs. Ordinary training resume rejects this closed run before and after resource acquisition. Failed registration can reuse the same pinned early-stop receipt without changing checkpoint selection.

```bash
cd /data1/hl/DcDsDiff-and-GIT10K
PY=/data0/hl/conda_envs/dcdsdiff/bin/python
# Already authorized; register only once after the training process has exited:
$PY tools/queue_benchmark.py queue --run-id DCDSDIFF-CASIA2-SEL3-ALL8-20260914-B --early-stop
$PY tools/queue_benchmark.py status --run-id DCDSDIFF-CASIA2-SEL3-ALL8-20260914-B
# Only an interrupted evaluation may resume; do not resume training:
$PY tools/queue_benchmark.py resume --run-id DCDSDIFF-CASIA2-SEL3-ALL8-20260914-B
```

Result directory: `runs/DCDSDIFF-CASIA2-SEL3-ALL8-20260914-B/followups/all8-best/evaluation/` (`report.md`, `results.json`, `per_image.csv`). Only follow-up `status.json=COMPLETED` establishes successful evaluation completion. [Dedicated-environment validation](validation_early_stop.json) records bounded CPU checks; temporary test scripts/checkpoints are removed after validation.

GPU 0 was already completed before this request: fixed epoch-99 GIT10K evaluation and its separately selected best-checkpoint All8 evaluation both finished on 2026-09-15. Neither is rerun or changed by this request.

## Completed evaluation

All8 completed on 2026-09-15 at 14:36:01 Asia/Shanghai using evaluation commit `8deb7dc4f505632ca1ac82a93477e9bed30d6231`. All 4295 images were evaluated: macro F1 `2.4819740338628007e-7`, pooled F1 `7.576679125569692e-7`; six datasets have zero F1. Only two images have positive F1. This negative result is retained without reselection. The evaluator and controller exited and all owned locks were released. [Results and interpretation](results/README.md), [verification receipt](results/completion_receipt.json).
