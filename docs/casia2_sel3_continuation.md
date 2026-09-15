# CASIA2: three-set selection, final All8 report

**Superseded endpoint on 2026-09-15:** the user ended B early after 85 complete epochs (0–84). Use the existing epoch-52 best for All8 under the [early-stop protocol](casia2_early_stop.md). The 100-epoch continuation/resume instructions below describe the original registration and must not be used to restart B. Final B results are now under `followups/all8-best/evaluation/`.

The user requested stopping GPU 1 and using only Casiav1, Columbia and NIST16 for every subsequent epoch test. Both the original GPU 0 run and the modified CASIA2 experiment must finish with eight-dataset F1 reports from their existing `model-best.pt` files.

## Boundary and preserved training

- Parent: `DCDSDIFF-CASIA2-ALL8-20260914-A`, cleanly stopped at 2026-09-14 21:46:25 Asia/Shanghai. Controller 3843754 and trainer 3843758 exited; GPU/run locks were released.
- Last complete checkpoint: `model-last.pt`, epoch 2, next epoch 3, global step 2562; SHA-256 `845e53184fa8b73ee3f0bc907f243cdc50350b2e7c38bba8aebb1311494b06ee`.
- Continuation: `DCDSDIFF-CASIA2-SEL3-ALL8-20260914-B`, protocol `CASIA2-SEL3-ALL8-V1`, config `config/experiments/casia2_sel3_all8.yaml`.
- Preserve exact model, optimizer, scheduler, scaler and RNG states. Continue epochs 3–99 for **100 total epochs**, including the three completed parent epochs. The interrupted parent epoch 3 is replayed from its complete boundary.
- CASIA2 training remains the same 5,123 Tp/Gt pairs; architecture, loss, 352px, batch 6, FP32, seed 42, optimizer and scheduler are unchanged. Only the diagnostic dataset changes; the trainer additionally records diagnostic dataset names.

This is an explicit protocol continuation, not a fresh run or an unchanged-protocol resume. The parent source, checkpoints and original records are retained. The child records parent source/checkpoint hashes and the imported checkpoint hash. Ordinary child resume subsequently requires its exact frozen contract.

## Selection and final reporting

| Per-epoch selection dataset | Images |
|---|---:|
| Casiav1 | 920 |
| Columbia | 180 |
| NIST16 | 564 |
| Total | 1664 |

The subset reuses the existing immutable All8 image/mask/detail/trace files, with the original transforms and ordering restricted to the selected samples. The formal validator checks exact manifest membership, not just filename prefixes. The final evaluator still loads all eight datasets and all 4,295 images.

After changing the selection population, old All8 MAE values must not compete with the new three-set MAE. Child best is reset to infinity/no epoch and is first set by the completed epoch-3 diagnostic. Best candidates are epochs 3–99; no retrospective sweep of parent checkpoints is added. Each image has equal weight in the pooled MAE. The inherited epoch-0/1/2 records keep their original values and explicitly list all eight diagnostic datasets. New records/status list only the three named datasets.

After epoch 99, the CASIA2 child evaluates its saved three-set-MAE best once on All8, reporting per-dataset F1/IoU/MAE and macro/pooled summaries. The three selection datasets are test-selected. The other five become reporting-only after this boundary; the parent previously monitored them, so this does not establish a fresh, entirely held-out selection history.

GPU 0 keeps its original GIT10K 9000/1000 training/selection and fixed-final reproduction report. Its already-running detached follow-up still evaluates its GIT10K-MAE-selected best once on All8. No GPU 0 restart, checkpoint alias, threshold sweep or change to best selection is introduced.

## Control and validation

Registration holds the new GPU/run locks and the parent run lock, requires the parent controller/children to be stopped, validates the frozen data/environment/source and imports the complete checkpoint transactionally. A recoverable parent PREPARING marker blocks resume before the child is published, then becomes REGISTERED. Root resume checks the marker again after acquiring resource locks. The child uses the existing detached train/final-evaluate controller.

[Validation receipt](validation_sel3.json): exact model/optimizer/scheduler/RNG serialization and real Trainer restore, next_epoch/global_step parity, old/new population labels, exact 1664-member loader, config/missing-state guards, injected registration failures and recovery, post-lock parent-resume race rejection, one real GPU 1 backward/optimizer step, and one original-size 10-step prediction from each of the eight datasets. These are engineering fixtures, not additional formal epochs or reported scientific scores. Temporary scripts/checkpoints are removed before startup.

Independent review approved the final diff with no remaining Critical/Required finding after verifying the registration and resume-race fixes. Changed-source hashes match the primary receipt or its final guard recheck. [Cleanup receipt](sel3_cleanup_receipt.json) confirms temporary validation scripts, imported test checkpoint and staging were removed locally/on the server; the real parent checkpoint remains intact.

```bash
cd /data1/hl/DcDsDiff-and-GIT10K
PY=/data0/hl/conda_envs/dcdsdiff/bin/python
# Register this authorized transition only once:
$PY tools/continue_casia_selection.py \
  --parent-run DCDSDIFF-CASIA2-ALL8-20260914-A \
  --run-id DCDSDIFF-CASIA2-SEL3-ALL8-20260914-B --gpu 1
$PY tools/manage_experiment.py status --run-id DCDSDIFF-CASIA2-SEL3-ALL8-20260914-B
# If interrupted later, resume the child, never the superseded parent:
$PY tools/manage_experiment.py resume --run-id DCDSDIFF-CASIA2-SEL3-ALL8-20260914-B --gpu 1
```

CASIA2 final F1: `runs/DCDSDIFF-CASIA2-SEL3-ALL8-20260914-B/evaluation/report.md`.
Original GPU 0 All8 F1: `runs/DCDSDIFF-GIT10K-RECON-20260914-A/followups/all8-best/evaluation/report.md`.
