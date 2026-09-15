# TECT-DIFF-FULL-R512-S42

2026-09-15 后续状态：原A已停止，冻结协议和快照保留；版本化修复配置与验证边界见 [REFNORM_V2.md](REFNORM_V2.md)。用户随后授权修复版本重新完整训练，新运行身份与前期监督见 [RESTART_V2.md](RESTART_V2.md)。

Registered from the complete user-supplied `TECT_Diff_Codex_Prompt.md` on 2026-09-15. This is one new engineering/scientific protocol, not an exact reproduction of DcDsDiff or MedSegFactory. The inherited source at registration is `f257a17e969152800ff69c05a2061122aac0ec7e`; origin is the authenticated user's `git@github.com:sleepDinner/DcDsDiff.git`, development branch `feature/tect-diff`.

The supplied 523-line task brief SHA256 is `61d36d9cdc578368be7bf1ab1a0996dff452f96693723b5dc7216985a514fe5e`.

## Actual source audit

- DcDsDiff upstream anchor: [d2ad59fc](https://github.com/QixianHao/DcDsDiff-and-GIT10K/tree/d2ad59fc218727c2148328e73c3d7a9fcb2b8bde). Current project already repairs MSIE constructor, backbone_t naming, feature-key pretrained loading, structure_loss reduction and duplicate test expansion. These are engineering fixes, not TECT innovations.
- Active DVCN is two time-conditioned `pvt_v2_b2` backbones, four `MMFF` and `MSFF`, not a class named DVCN. The new path reuses these and the actual Mask `down2/up2/pred2` architecture. It never instantiates DIB/detail targets/detail diffusion or original MSIE. Original baseline files remain unchanged.
- `denoising_diffusion_pytorch.simple_diffusion.logsnr_schedule_cosine` includes both `-2 log(tan(theta))` and `-2 log(sin(theta))`; shift is `2 log(64/resolution)`. Reuse the actual function, not a textbook cosine substitute. Mask x0 changes from the legacy tanh path to `2 sigmoid(logits_ctrl)-1` and epsilon is recomputed consistently.
- MedSegFactory [b227c6b5](https://github.com/jwmao1/MedSegFactory/tree/b227c6b5f0ff6b02d6046a1cdf57fc47cb74ae96), [paper v2](https://arxiv.org/abs/2504.06897v2): inspected `tutorial_train.py`, `StableDiffusion/Our_Attention.py` and `StableDiffusion/Our_Pipe.py`. The batch-halving processor mixes query/output names, reuses overwritten features and drops its mask; no direct copy is used. TECT uses explicit streams and query-owned local windows. It does not reproduce two freely generated latent chains or recover an attacker's actual generation history.
- Read SHA256: tutorial `235f8522665b9e62e57b48add55716496cf7560b38d21aa3f8295e04fd0e2c17`; attention `8e29883e95d031e00edc1b8ceaa1255189a2eb6de1449bd5d18c3f511bac296e`; pipe `f69c2bf52af8967d5074446a93c96bbb452a4ac9419d010d788ae9b18f28166c`.
- Existing PVT ImageNet initialization is checked against SHA256 `80711cd1b37ffba12bec6c7a2a7c54efe2315ed635e6d2055fd51c3e909ede4d`. Both 332-key feature backbones must load completely; only documented time/mask-projection parameters are fresh. No historical trained localization checkpoint is reused.

## Fixed execution and data

The adjacent JSON resolves all settings. GPU0+GPU1, two-process DDP, BF16 when supported, batch2 per rank, two accumulation microbatches, effective batch8. Main AdamW 1e-4/wd0.01/cosine to1e-6, epochs0–99, seed42. Reference AdamW2e-4/wd0.01, constant LR, effective batch16, 20 epochs, fixed final only. No gradient clipping is imposed; gradient norm/finiteness are measured. OOM order is graph/cache audit, serial reference, micro1/accum4, checkpointing/eval1, then and only then a separately named R352 run with new calibration.

Train on all valid `/data0/hl/FinalTrainData` pairs, no validation split. Exact relative stems and registered mask suffixes determine pairs. Preserve original data. Explicitly invalid coordinate pairs are quarantined; decoded RGB and available source-ID test overlaps are excluded from training and listed. Hash matching does not exclude every transformed or unidentified same-source relationship.

Authentic reference eligibility requires a confirmed source identity and an actual empty paired mask. Train/reference/calibration roles are source-group bound; all retained role images still train the main model. If only protected annotation-clean crops are available, use `reference_source_mode=mask_clean_proxy`, add MASKCLEAN to run/report and describe global editing-pipeline limitations. Missing legal reference sources block the dependent stages.

RGB directly resizes from the original to final size. HFVG uses the existing original-image FFT circular highpass (radius min(H,W)/4), abs(real IFFT)*10, clipping/uint8 quantization, then the same resize/flip; never resize to352 then512. RGB PVT uses ImageNet normalization; HFVG and Image observations use [-1,1]. Masks use nearest binary resize. Main augmentation is only synchronized horizontal flip with probability0.5. DataLoader workers restart with epoch/rank seed at each epoch boundary; sampler set_epoch and deterministic worker seeds make boundary replay possible. DistributedSampler's at most one padding duplicate is explicitly recorded for training; evaluation never pads.

## Complete mechanism

Reference U-Net: PixelUnshuffle4 -> widths64/128/256/256, two time-conditioned residual blocks per level, symmetric skips, GroupNorm/SiLU ->256-channel H/4 features ->48-channel epsilon ->PixelShuffle4. Its forward accepts only noisy image/time. It is trained with true epsilon MSE then requires_grad=False/eval, including under outer train(). Frozen tensors use no_grad/detach, not inference tensors saved by later autograd.

Two independent image noises are reused across lambda[2,3,4,5] within each replica. Every scale reanchors to I_obs. Reference response is the fixed linear Phi of epsilon_ref-epsilon_draw, mathematically equal to a/sigma Phi(I_obs-I0_ref); reconstruction is not clamped. Phi is RGB horizontal derivative/vertical derivative/Laplacian (9 channels), with fixed boundary exclusion. The artifact records content descriptors, variance floors, shrinkage and hashes.

Content-only spatial-grid KNN uses up to1024 candidates, eight remote references, exclusion radius max(H,W)/8 and bounded chunks. It cannot read GT, predicted masks or replica anomaly responses. Subtract training-fit normal mu(content), then the shared graph's weighted normal responses. Concatenate current prefix response and lambda-normalized finite differences. Each prefix has fixed positive diagonal W with shrinkage, variance floor and dimension normalization. For R2 compute signed `v1^T W v2`; do not rectify negative estimates.

Calibration fits at most2048 source-role training images, up to128 query pixels per available class per image. Normal mu/W use only training GT=0. Two regularized low-dimensional logit models share identical sampled pixels: joint(A,content) minus content-only, clipped[-5,5]. Reliability uses matching distance, reference availability and cross-replica variation with training-fitted d0/v0. Each prefix saves complete fitted tensors; test inference never fits. This is estimated evidence, not an exact density ratio or proof that all authentic pixels have zero anomaly.

Image adapter genuinely predicts the added image noise for both replicas and receives MSE averaged over batch/replica/channel/pixel. Mask feedback is only previous predicted probability resized to32, binarized, detached, with explicit history availability. Current Image features are K/V and Mask features Q in256-channel/4-head/8x8 attention with padding masks. Mask cannot alter reference epsilon or measurement A.

Main loss is FP32 structure_loss(logits_ctrl,GT)+MSE(epsilon_joint,epsilon_draw). No Detail loss. One uniform Mask time per microbatch maps through progress=1-(2/pi)asin(t), j=floor(10*progress), m=min(4,1+floor(4j/10)). Exactly the visible prefix is measured; no future scales enter training. About50% microbatches perform one no_grad/eval preliminary prediction for self-conditioning, preserving BatchNorm buffers; inference uses actual prior predictions.

At each step: logits_ctrl=logits_base+2sigmoid(g)*(j/9)*q*ell, with global gain initially0.1 and prefix1 control disabled. Recompute Y0_ctrl and epsilon from the current noisy mask. Deterministic DDIM reuses that same corrected parameterization; the final zero-noise step returns Y0 directly. Evidence is reapplied relative to each step's own base logits, not accumulated in an external tenfold logit sum.

## Selection and checkpoint semantics

Every completed main epoch evaluates all eight specified sets, exactly once per stable sample ID, no TTA, no normalization/threshold/checkpoint sweep. Image/Mask noise is derived from fixed seed0, ID and purpose, independent of rank/epoch/batch. Final output is the last Mask step's P_ctrl. Sigmoid precedes original-GT-coordinate bilinear interpolation. Threshold>=0.5, positive=tampered, empty/empty F1/IoU=1; all-ignore fails explicitly.

Primary is equal mean of eight per-dataset means of per-image Pixel-F1. Extra IoU, Boundary-F1 (ceil0.005*diagonal, minimum1), positive-only F1, authentic false-positive rates and pooled Pixel-F1 are reported. Auxiliary metrics do not veto selection. All eight finite result files must exist before atomic best update. Ties keep earlier epoch. best is one full epoch, never a per-dataset combination.

**selection_protocol=test_selected: 按八测试集逐 epoch F1 选择 checkpoint；这些测试集参与模型选择，不是独立泛化评估。** Also report fixed final.pth(epoch99). last stores full optimizer/scheduler/scaler/all-rank RNG/sampler/next_epoch and pending evaluation. Interrupted evaluation replays this same pending epoch's full All8 before training advances. Best/final contain full task/reference state plus calibration artifact, source/config/data hashes and preprocessing/sampling config.

## Reporting, limits and ownership

Detached controller holds both GPU locks and run lock, checks the legacy lock, uses fixed published source, emits60-second heartbeat, executes prerequisites automatically and publishes terminal reports through a separate project-local worktree. Only small code/config/reports/ledgers are Git content; images/visualizations/checkpoints/cache/large logs remain server-only. Push failures retain pending_sync and bounded retry. Training logic failure stops with a report; no unbounded retraining or hidden protocol change.

Engineering diagnostics establish finite gradients, isolation, correct math, coverage and resumability. They do not establish an innovation gain. Report q collapse, negligible gamma, degenerate Image adaptation or harmful correction honestly. No matched full baseline is authorized; do not claim superiority over the historical models with different data/selection/postprocessing. Startup is not completed training.
