# DcDsDiff reproduction rebuild

## Active task: TECT-Diff, authorized 2026-09-15

The complete user-supplied TECT_Diff_Codex_Prompt.md defines this new experiment. Its explicit dual-GPU batch8, FinalTrainData, All8 macro-F1 test selection, Image task and 100-epoch endpoint supersede old baseline settings only for this new protocol.

1. [complete] Verify local/server/user GitHub, pinned scientific sources, environment and legal reference sources. Data decoding audits all 52924 pairs; 17 invalid training coordinate pairs are isolated, existing All8 PIL-L semantics retained.
2. [complete] Implement isolated TECT model, training-only fitted calibration, DDP pipeline, recovery and automatic report publication.
3. [complete] Reviewed implementation, published execution commit 4b4b785103f10ec9d9281cc8af095e3f91609433 and verified local/GitHub/server agreement. CPU math/data and dual-rank full512 backward/sampling/in-memory checkpoint recovery passed.
4. [complete] Launched TECT-DIFF-FULL-R512-S42-20260915-A with immutable125-file snapshot, held GPU0/GPU1/run locks and detached controller447738. Real reference training advanced step1 ->550, losses finite and no AMP skip; input remains512.
5. [complete] Startup evidence and server-generated report published; server controller owns A20 -> training-only calibration -> C100/All8 each epoch -> terminal reports/GitHub. Trained-artifact full-path probe runs automatically after calibration. Scientific experiment is RUNNING, not complete; this status does not mark the experiment endpoint achieved.

No additional seed, baseline or ablation training is authorized. Historical results and frozen sources remain protected.

User-authorized scope: audit and clean local project, publish to the existing GitHub origin, create /data1/hl/DcDsDiff-and-GIT10K, create /data0/hl/conda_envs/dcdsdiff, reuse project resources, verify on the server and start the reproduction automatically. Keep other server projects untouched.

## Phases
1. [complete] Capture initial state; independently audit paper, active code and server resources.
2. [complete] Freeze a documented reproduction protocol and clean proven obsolete artifacts.
3. [complete] Repair execution/recovery paths and add scoped environment and experiment management.
4. [complete] Build new server environment, stage resources and run bounded validation; remove temporary test code.
5. [complete] Review, commit and push; pull identical code on server and launch supervised formal training.
6. [complete] Verify live training, provenance, durable controller and management instructions. Startup receipt shows step 1 to 829, a held lock and 58 matching frozen source hashes. Scientific training/evaluation remains RUNNING under server control.

## Decisions
- Preserve existing uncommitted work before cleanup.
- Separate published-paper claims from official-code behavior and reconstruction assumptions.
- Record train/test manifests and fixed reporting policy before training; do not select favorable checkpoints after results.
- Heavy computation uses the new server environment only.
- Completion of this setup task requires verified experiment startup and durable server control; scientific reproduction completion requires actual final evaluation.

## Authorized CASIA2 / All8 extension — 2026-09-14

1. [complete] Verify GPU 1 is free and audit the requested dataset roots and pairing contracts.
2. [complete] Freeze CASIA2 + All8 inputs and record data-only training protocol / existing-best checkpoint policy.
3. [complete] Validate per-GPU/run controller locks, eight-set evaluator and durable original-run follow-up in the dedicated environment.
4. [complete] Review, publish local to GitHub, pull on server, launch CASIA2 on GPU 1 and queue original-run best evaluation. Both use frozen execution commit cd0df49; original training remains on 20d4428.
5. [complete] Verified continued live training (CASIA2 step 201 to 603; original step 13243 to 13500), detached controller identities, live original-run follow-up and removal of temporary tests. Startup receipts and ledger are published with final local/GitHub/server checkout agreement. Scientific training and both requested All8 evaluations remain under server control; no final scores are claimed.

## Authorized three-set selection change — 2026-09-14

1. [complete] Stop GPU 1 only; verify controller/children exit and resource locks release. Parent checkpoint completed epoch 2, next_epoch 3; GPU 0 unchanged.
2. [complete] Implement a new CASIA2 continuation with the same optimization/RNG state, 100 total epochs, three-set pooled MAE selection and final All8 F1 reporting. Reset best across the selection boundary and label historical metrics.
3. [complete] Validated migration, exact subset membership, real GPU resume/backward, final inference/guards and registration race recovery in the dedicated environment; independent review approved; all temporary tests removed.
4. [complete] Published 01db26c and pulled on the server; continued GPU 1 from epoch 3/step 2562 with exact 1664-image selection and final All8 evaluation. Verified detached live training, held locks, immutable sources/parent checkpoint, and unchanged original GPU 0/All8 queue. Startup receipts/ledger are synchronized across local/GitHub/server; scientific training and final reports remain running/pending.
