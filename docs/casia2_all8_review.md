# CASIA2 / All8 implementation review and validation

Scope: data-only CASIA2 training on GPU 1, All8 best-checkpoint reporting, and a durable additional All8 evaluation after the original frozen GPU 0 run. The user's correction is honored: both evaluators read the existing `model-best.pt` directly.

Independent review using the code-review-and-quality workflow found no remaining Critical or Required issue after fixes. Required fixes covered GPU/run lock inheritance and validation, legacy-controller compatibility, pinned GPU resume, preservation of the original final-checksum status field, atomic follow-up registration, actual data-file integrity checks, and correct descriptions of the two distinct checkpoint-selection populations.

Validation used `/data0/hl/conda_envs/dcdsdiff`:

- [CPU receipt](validation_transfer_cpu.json): strict pairing/config guards; real success and orphan-failure controller processes; inherited lock exclusion/release; missing-FD and wrong-GPU rejection; dependent controller completion/exit; real follow-up dispatch; simulated registration failure cleanup; manifest and overlap rejection.
- [GPU receipt](validation_transfer_gpu.json): six real CASIA2 training samples, one optimizer step with finite loss/gradients, plus one real original-size image from each of eight datasets. Actual CLI checkpoint reload and 10-step inference passed. The independent pooled MAE exactly matched the training diagnostic on this fixture. A later evaluator guard recheck is separately recorded with its source hashes.
- [Source receipt](validation_transfer_provenance.json): all original frozen source files remain unchanged, and model/dataset/training-utility Python content agrees across arms in Git's canonical LF form. No architecture, loss, augmentation or training-loop changes are part of this extension.
- [Raw-image audit](all8_overlap_audit.json): 9,418 new raw images checked; zero exact CASIA2-train/All8-test and GIT10K-train/All8-test RGB overlaps.

The GPU fixture explicitly simulated the completed-training gate after **one optimizer step and zero formal epochs** to exercise the real final evaluator. Its checkpoint and numeric scores are engineering evidence only, never a scientific experiment result. Temporary scripts/data/checkpoints are removed before handoff; concise receipts are retained. Formal preparation and launch evidence are recorded separately in the experiment ledger.
