# Project rules

- This checkout reproduces DcDsDiff and is the base for future controlled innovations. Read README.md, docs/paper_protocol_audit.md, docs/reproduction_protocol.md and experiment_ledger.md before changing a scientific path.
- Baseline `GIT10K-PAPER-RECON-V1` is a documented reconstruction, not an exact official split. Never silently relabel filename prefixes as paper generator categories, reuse historical trained weights as fresh initialization, or claim a running job has reproduced paper scores.
- Keep the baseline fixed. A new idea gets its own named config/protocol ID, Git commit and run ID. Do not change a running source snapshot or frozen data manifest. Additional formal runs need user authorization.
- Work only in this project, `/data1/hl/DcDsDiff-and-GIT10K`, and its environment `/data0/hl/conda_envs/dcdsdiff`. Original `/data0/hl/DcDsDiff-and-GIT10K/GIT10K` is a read-only data source. Read/copy useful old project resources; do not modify other environments, processes or datasets.
- Never overwrite CUDA_VISIBLE_DEVICES in imported code. The current controller is single GPU/single process, global batch 6. Check GPU occupancy and the project lock before launching. Never stop an unrelated process.
- Save full optimizer/scheduler/scaler/RNG state and next_epoch; resume only the same frozen protocol at an epoch boundary. Old checkpoints without those fields are not exact resumptions.
- Primary result is predeclared final epoch 99, inference seed 0, 10 steps, threshold 0.5. Best MAE is test-selected diagnostic only. No post-hoc checkpoint or threshold sweep.
- Deep-model validation runs in the new server environment. Use bounded temporary validation scripts and remove them afterward; retain concise validation receipts, not test checkpoints or large outputs, in Git.
- Keep datasets, checkpoints, caches and local cleanup backups out of Git. Record source commit, environment, data/weight hashes and real run state in each run. Keep the ledger current.
- Supervision lives on the server: detached controller, inherited project lock, 60-second lightweight state updates, then fixed final evaluation. Do not use continuous local SSH/sleep loops.
- Before publication inspect the actual diff, run relevant bounded checks, and confirm local/GitHub/server commit agreement. Never force-push shared history.
