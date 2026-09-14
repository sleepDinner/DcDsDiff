# Experiment ledger

## GIT10K-PAPER-RECON-V1 — RUNNING

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
- Scientific status: no formal result yet. Running training and successful engineering checks do not establish agreement with paper scores. Final closure requires controller `COMPLETED`, final evaluation receipts, and publication of the small result files; weights remain on the server.

Historical runs are summarized separately in [pre-rebuild audit](docs/history/pre-rebuild-audit.md). They do not fulfill this registered protocol.
