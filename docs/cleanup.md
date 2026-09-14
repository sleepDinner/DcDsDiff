# Cleanup record — 2026-09-14

Initial commit: `d60e4e893ad391a699fe5229a4be0903cfc3cbf3`.
Before cleanup, all Git history was saved in `.local/rebuild-backup/before-rebuild.bundle`, uncommitted changes in `initial-working-tree.patch`, and the untracked prior audit folder in `prior-audits.zip`. The useful prior audit narrative is retained as `docs/history/pre-rebuild-audit.md`; its path/line references describe the old checkout.

Removed from the active project and placed in the ignored, recoverable `.local/cleanup-archive/`:

- Obsolete 384px/COD dataset configs, FinalTrainData config and hardcoded 352px data paths; replaced by `config/reproduction.yaml`.
- `sample.py` (hardcoded IA/GPU paths), the large legacy evaluation wrapper, and post-hoc F1 checkpoint selector; replaced by the fixed-endpoint evaluator.
- Old tests bound to obsolete paths/source strings. Bounded validation is performed in the new server environment and only its results are retained in Git.
- Unreferenced CDS2K loader, boundary modification helpers, duplicate PVT module, ViT config and saliency metric/evaluation modules. Reference search confirmed the active reproduction does not import them.
- Incomplete tracked Hugging Face cache reference, old temporary image samples, bytecode cache and redundant prior audit exports.

The initial bulk deletion command was rejected by automatic approval review. Cleanup instead used verified, project-scoped, reversible moves.

After validation, the newly created local/server test scripts, eight-image test dataset, temporary model checkpoint/evaluator outputs, staging checkout/archive and redundant rejected data candidate were deleted with explicit path and process checks. Small validation logs/receipts remain; see `docs/cleanup_receipt.json`. No test script or test checkpoint is part of the published runtime.

The official vendored diffusion library remains because its package initializer imports its modules. The local raw `GIT10K.rar` and supplied paper PDF remain useful original resources. Server source datasets, historical checkpoints/runs and unrelated projects were not deleted.
