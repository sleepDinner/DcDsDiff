# DcDsDiff reproduction rebuild

User-authorized scope: audit and clean local project, publish to the existing GitHub origin, create /data1/hl/DcDsDiff-and-GIT10K, create /data0/hl/conda_envs/dcdsdiff, reuse project resources, verify on the server and start the reproduction automatically. Keep other server projects untouched.

## Phases
1. [complete] Capture initial state; independently audit paper, active code and server resources.
2. [complete] Freeze a documented reproduction protocol and clean proven obsolete artifacts.
3. [complete] Repair execution/recovery paths and add scoped environment and experiment management.
4. [complete] Build new server environment, stage resources and run bounded validation; remove temporary test code.
5. [in_progress] Review, commit and push; pull identical code on server and launch supervised formal training.
6. [pending] Verify live training, provenance, durable controller and management instructions.

## Decisions
- Preserve existing uncommitted work before cleanup.
- Separate published-paper claims from official-code behavior and reconstruction assumptions.
- Record train/test manifests and fixed reporting policy before training; do not select favorable checkpoints after results.
- Heavy computation uses the new server environment only.
- Completion of this setup task requires verified experiment startup and durable server control; scientific reproduction completion requires actual final evaluation.
