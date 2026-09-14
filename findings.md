# Findings

2026-09-14: Local main is at d60e4e893ad391a699fe5229a4be0903cfc3cbf3 and origin is git@github.com:sleepDinner/DcDsDiff.git. Remote main matches. Twelve files have existing modifications, mainly comments plus imports/config changes. Untracked prior audit evidence exists. No relevant memory registry hit. SSH imdl-server works (hostname amax).

Official source: https://github.com/QixianHao/DcDsDiff-and-GIT10K . Its README only supplies the GIT10K download link. Independent paper/code/server audits are underway.

Initial runtime issues: utils/init_env.py overwrites CUDA_VISIBLE_DEVICES and points at a nonexistent vendored-library directory. Trainer checkpoints do not save optimizer/scheduler/RNG state; current resume is not an exact continuation. These require correction for reliable long-term management.

Paper/official differences confirmed: paper LR .001 vs public .0001; paper horizontal flip absent from public active path; WBCE reduce typo; attention pooling/convolution differs. Rebuilt paper-aligned-v1 explicitly and documented remaining underspecified auxiliary/split/metric conventions. Server has 10000 raw matching PNG pairs, nine filename prefixes with no confirmed mapping to four paper generators, soft masks and some dimension mismatches. New source-only baseline may not use any historical trained checkpoint. Full data preparation rejects cross-split identical decoded RGB images; all f/m/d/t hashes are retained.
