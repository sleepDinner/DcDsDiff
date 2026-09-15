# Experiment ledger

## GIT10K-PAPER-RECON-V1 — COMPLETED

- Request date: 2026-09-14.
- Scope: one fresh paper-aligned main model, one fixed epoch-99 evaluation; no ablation/sweep/additional seed.
- Protocol/config: [frozen protocol](docs/reproduction_protocol.md), `config/reproduction.yaml`.
- Data: original GIT10K 10,000 pairs, content-group-bound seeded 9,000/1,000 split; filename prefix categories are not author-confirmed generator labels. Zero cross-split identical RGB. [Manifest](manifests/git10k-recon-v1.csv), [receipt](docs/dataset_receipt.json). Manifest SHA-256 `d865abb3144ae843ba44a9d9d8e3bef547acdccaaad5936a64bd82e99b72d189`.
- Initialization: ImageNet PVTv2-b2, SHA-256 `80711cd1b37ffba12bec6c7a2a7c54efe2315ed635e6d2055fd51c3e909ede4d`; no old trained checkpoint.
- Engineering: isolated new environment/Pip check, 0/4-worker CPU recovery, controller lifecycle/orphan cleanup, full batch6 GPU backward and actual 10-step evaluator entrypoint all PASS. See `docs/validation_*.json`.
- Run ID: `DCDSDIFF-GIT10K-RECON-20260914-A`; started 2026-09-14 19:20:01 Asia/Shanghai (11:20:01 UTC).
- Execution commit: `20d44286b9354e0af189e179ba1bd162eac9b11e`, obtained on the server from GitHub before launch. Later documentation commits do not change the frozen `runs/<RUN_ID>/source` snapshot.
- Location/device: `/data1/hl/DcDsDiff-and-GIT10K/runs/DCDSDIFF-GIT10K-RECON-20260914-A`, dedicated `/data0/hl/conda_envs/dcdsdiff`, GPU 0 only.
- Status at 2026-09-14 19:24:30 Asia/Shanghai: RUNNING, epoch 0, global step 829/1500; finite loss 0.33141148. Observed advancement from step 1. Controller PID 3787713 and training PID 3787714 run from the frozen source with user-site imports disabled. Project lock held; all 58 frozen source-file hashes match. [Startup receipt](docs/startup_receipt.json).
- Automatic endpoint: 100 completed epochs followed by fixed epoch-99 evaluation of 1,000 test images. The detached server controller advances this sequence without an active SSH session.
- Historical scientific status at startup: no formal result yet. Running training and successful engineering checks do not establish agreement with paper scores. Final closure requires controller `COMPLETED`, final evaluation receipts, and publication of the small result files; weights remain on the server.

Historical runs are summarized separately in [pre-rebuild audit](docs/history/pre-rebuild-audit.md). They do not fulfill this registered protocol.

## CASIA2-ALL8-V1 — STOPPED_FOR_AUTHORIZED_CONTINUATION

- Authorized 2026-09-14: one fresh CASIA2 training arm on GPU 1 and one All8 best evaluation after training. Run ID: `DCDSDIFF-CASIA2-ALL8-20260914-A`; started 2026-09-14 20:39:52 Asia/Shanghai (12:39:52 UTC).
- Protocol: [CASIA2/All8](docs/casia2_all8_protocol.md); config `config/experiments/casia2_all8.yaml` inherits all non-data model/training parameters from the original baseline.
- Training: 5,123 paired CASIA2 Tp/Gt samples; All8 test count 4,295. Data preparation READY; [manifest](manifests/casia2-all8-v1.csv) SHA-256 `b5d7c329a957bb62e39d22bbef79f4b3d3bb071487fc97212ca2e99455c1bd8b`. [Dataset receipt](docs/casia2_all8_dataset_receipt.json). All image/mask dimensions match, no empty masks or unit-mask scaling, and zero exact train/test RGB overlaps. All 9,418 manifest RGB hashes match the independent raw-image audit.
- Reporting: existing `model-best.pt` minimizes pooled All8 per-image MAE. Eight individual datasets plus macro/pooled summaries; explicitly test-selected. No aliases, checkpoint sweep or extra seed.
- Execution commit: `cd0df49f07a9bf250a62e9ba80e65f079c5b8e0f`, published to GitHub and pulled on the server before launch. Location: `/data1/hl/DcDsDiff-and-GIT10K/runs/DCDSDIFF-CASIA2-ALL8-20260914-A`; Python: `/data0/hl/conda_envs/dcdsdiff/bin/python`. All 75 frozen source hashes match.
- Startup at 2026-09-14 20:41 Asia/Shanghai: controller PID 3843754, trainer PID 3843758, GPU 1, epoch 0, global step 201/854, finite loss 0.47797334. GPU and run locks are held independently of the original GPU 0 controller. [Startup and follow-up receipt](docs/casia2_all8_startup_receipt.json).
- Continued progress verified at 20:43:55 Asia/Shanghai: CASIA2 advanced from step 201 to 603 with finite loss; original GIT10K advanced from 13,243 to 13,500 and entered its epoch-8 diagnostic evaluation. Both detached training sessions retain their original identities. The follow-up controller's state timestamp advanced while remaining `WAITING_FOR_TRAINING`.
- Original GIT10K follow-up: registered at 20:40:08 Asia/Shanghai under `runs/DCDSDIFF-GIT10K-RECON-20260914-A/followups/all8-best`; detached controller PID 3844284 is `WAITING_FOR_TRAINING`. Its frozen evaluation commit is `cd0df49f07a9bf250a62e9ba80e65f079c5b8e0f`. After the original controller completes and exits, it will acquire GPU 0 and evaluate the original GIT10K-MAE-selected `model-best.pt` once on All8. The original controller/trainer PIDs remain 3787713/3787714; all 58 original source hashes match, and its final99 primary result remains unchanged.
- Engineering: new environment CPU lifecycle/lock guards and actual GPU 1 six-image backward / eight-set sample evaluation passed. The 5,123-image batch sampler covers every sample without padding duplicates (854 batches, last batch 5). Formal launch passed configuration/environment/weight checks and verified all 37,672 data files against the 9,418-sample manifest. Temporary validation code/data/checkpoints and preparation staging were removed; [cleanup receipt](docs/transfer_cleanup_receipt.json).
- Superseded on user request at 2026-09-14 21:46:25 Asia/Shanghai: controller is INTERRUPTED with no cleanup error; controller/children exited and locks released. Last complete checkpoint is epoch 2 (3 completed epochs), step 2562. Partial epoch 3 is not retained. No formal All8 result exists for A; the authorized continuation below replaces its remaining training and final evaluation.

## CASIA2-SEL3-ALL8-V1 — EARLY_STOPPED_BY_USER; ALL8_EVALUATION_COMPLETED

- User revision: per-epoch testing uses only Casiav1/Columbia/NIST16 (920/180/564 = 1664 images); final best-checkpoint F1 covers all eight datasets (4295 images).
- Run: `DCDSDIFF-CASIA2-SEL3-ALL8-20260914-B`; [protocol](docs/casia2_sel3_continuation.md), config `config/experiments/casia2_sel3_all8.yaml`.
- Parent: A/model-last.pt SHA-256 `845e53184fa8b73ee3f0bc907f243cdc50350b2e7c38bba8aebb1311494b06ee`. Preserve model, optimizer, scheduler, scaler, RNG and step 2562; continue epochs 3–99, 100 total epochs. New best starts with no candidate, is selected by three-set pooled MAE, and cannot reuse the parent's All8-selected best.
- Parent history retains its original All8 scores with explicit dataset labels. The final report records the selection boundary and prior All8 exposure. No fresh-run or completely held-out-history claim.
- Dedicated environment validation passed, including actual state restoration/one GPU step, exact subset membership, eight sample inferences and failure/race guards; [receipt](docs/validation_sel3.json). Temporary scripts/test checkpoints were deleted; [cleanup](docs/sel3_cleanup_receipt.json).
- Execution commit: `01db26c96581818499120240dc01ffaa5e71e3a9`, published to GitHub then pulled on the server before registration. All 37,672 data files passed formal integrity checks; all 82 child source files match frozen provenance.
- Live verification at 2026-09-14 22:06:27 Asia/Shanghai: detached controller/trainer PIDs 3903032/3903033 on GPU 1, epoch 3, global step 2770 (208 new training batches), finite loss. Parent checkpoint hash is unchanged; imported optimization/RNG state begins at step 2562. New best remains unset until the first completed three-set diagnostic. [Startup receipt](docs/sel3_startup_receipt.json).
- Continued advancement verified at 22:07:48 Asia/Shanghai: step 2770 to 2980, still RUNNING with finite loss and the same three-dataset status labels.
- The parent's superseded marker is REGISTERED; A remains stopped. Child GPU/run locks and original GIT10K/follow-up locks are held, and original controller 3787713 / follow-up 3844284 remain alive. Both final All8 F1 reports remain pending under their server controllers.
- GPU 0 original training and its frozen All8 saved-best follow-up remain active and unchanged.

## 2026-09-15 authorized early endpoint and result closure

- GPU 1 B stopped cleanly at 14:13:12 Asia/Shanghai; 85 complete epochs (0–84), last complete step 72590. Partial epoch 85 excluded. Existing best epoch 52, three-set pooled MAE 0.10169355626924069; checkpoint SHA-256 a280c3d7db017d9af81fe5aa18e754024de95115dc7aaae7519ce9902bbef2c5. No further training. [Early-stop protocol](docs/casia2_early_stop.md); All8 evaluation pending at registration.
- GPU 0 original fixed epoch-99 evaluation completed at 11:10:39 Asia/Shanghai: GIT10K Mix1000 F1 0.1805838822467181, IoU 0.10530443043682479, MAE 0.4921142281591892. Its independent saved-best All8 follow-up completed at 11:26:51; pooled F1 0.0529762559682249, macro F1 0.05822430823011286. These existing results will be published without rerunning either evaluator. Earlier RUNNING statements above are historical startup evidence.

- GPU 1 All8 evaluation completed at 2026-09-15 14:36:01 Asia/Shanghai, evaluation commit `8deb7dc4f505632ca1ac82a93477e9bed30d6231`. All 4295 test images / 17180 input files verified. Only 2 images have positive F1: macro F1 2.4819740338628007e-7, pooled F1 7.576679125569692e-7; six datasets have exactly zero F1. This is a recorded negative result, not successful paper-score reproduction. No checkpoint reselection or extra inference followed.
- Final closure verified at 14:37:17: both follow-ups COMPLETED, GPU 0 original controller COMPLETED, GPU 1 training remains explicitly INTERRUPTED/EARLY_STOPPED_BY_USER. All controller/owned evaluation children exited; GPU/run/follow-up locks released. All frozen source hashes and checkpoint hashes match; all three CSV result aggregations agree. Temporary validation files removed. [Combined report](docs/results/README.md), [completion receipt](docs/results/completion_receipt.json).

- [TECT-Diff TECT-DIFF-FULL-R512-S42-20260915-A](analysis_reports/runs/TECT-DIFF-FULL-R512-S42-20260915-A/report.md); test_selected All8 F1; fixed final reported separately.
