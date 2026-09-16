# Experiment ledger

## TECT-PILOT-CASIA2-GN8-R512-S42-DATA2-N8192-V1 — COMPLETED / NO_GO

- C于2026-09-16 00:20:55 UTC正常结束十轮（epoch9/step6830），资源释放；自动结果发布提交`9ab9e02241a3dac630806ded3cdf162feaed10dd`。最佳epoch4平均Test2为0.507495但真实图误报未过门槛，末轮0.235143，未获得连续三轮通过；完整两库确认及Full/All8未执行。[终态报告](analysis_reports/runs/TECT-PILOT-CASIA2-GN8-N8192-R512-S42-20260916-C/report.md)。保留所有历史结果，以下启动描述是历史时点。
- 下一项仅为[固定C最终检查点的有界训练图诊断](analysis_reports/tect_diff/c_nogo_diagnostic_plan_20260916.md)，旧B诊断不重放，不自动增加数据/epochs或改变门槛。

- Next run `TECT-PILOT-CASIA2-GN8-N8192-R512-S42-20260916-C`; [registered scope](configs/tect_diff/PILOT_N8192.md), [config](configs/tect_diff/pilot_casia2_gn8_data2_n8192_r512_s42.json). Registered before C results; actual startup/source/manifest receipts must be recorded after execution.
- Fresh MAIN, same GN8/mechanisms/optimizer/batch/Test2/gates; fixed nested8192 train (4096Au+4096Tp), maximum10epochs/4h. Data coverage and total update budget change jointly. The [fixed B-final training diagnosis](analysis_reports/tect_diff/gn8_pilot_nogo_diagnostic_20260916.md) supports working conditioning and sampling, not proof of insufficient data or guaranteed improvement. C remains development; no premature Full/All8.
- C实际于2026-09-15 21:50:36 UTC进入准备，21:51:30首个正式更新，冻结源码`02691917da17016f64f26ef820d5ccfa175e18ab`、配置hash`c42769f4f4fedc18f5ae21ae4c6525bb551a416e2e5ffd98ed067bc6877bd702`。21:52:56核验epoch0/step60，207冻结文件、8192包含原2048行、固定角色及拟合hash、双rank8次工程更新后fresh状态恢复均通过；856梯度同步，无AMP跳步，四个GPU/run/controller锁持有。显存当时21260/21396MiB；瞬时利用率不作为吞吐结论。见[启动收据](analysis_reports/tect_diff/gn8_n8192_startup_20260916.json)。尚无完整C epoch、Test2或收敛结论；tect-v2已ACTIVE绑定C，每30分钟检查。

## TECT-PILOT-CASIA2-GN8-R512-S42-DATA2-V1 — COMPLETED / NO_GO

- Current run `TECT-PILOT-CASIA2-GN8-R512-S42-20260916-B`; [DATA2 repair](configs/tect_diff/PILOT_DATA2_REPAIR.md), [configuration](configs/tect_diff/pilot_casia2_gn8_data2_r512_s42.json).
- A failed before GPU startup on two LA masks whose luminance/alpha channels disagree. B quarantines only those two exact hash-bound image/mask pairs; all unknown semantic errors still block. Same GN8 model, fitting dependencies, fresh initialization, sample/epoch budgets, precision and effectiveness gates. No pilot training result exists at this registration.

## TECT-PILOT-CASIA2-GN8-R512-S42-V1 — FAILED_DATA_PREPARATION

- User revision 2026-09-16 authorizes supervised framework/network repairs without repeat approval and smaller CASIA2/Casiav1/Columbia development experiments before full-data training.
- Run `TECT-PILOT-CASIA2-GN8-R512-S42-20260916-A`; [registered pilot protocol](configs/tect_diff/PILOT_CASIA2_GN8.md), [configuration](configs/tect_diff/pilot_casia2_gn8_r512_s42.json).
- Fresh seed42/ImageNet MAIN with the 15 task BatchNorm sites replaced by GroupNorm8; E and earlier held runs remain unchanged. Reuse C's fixed epoch19 reference and training-fitted calibration through E's hash-bound receipts; no inherited MAIN weights. Fitting exposure remains FinalTrainData, explicitly disclosed.
- Fixed 2048 CASIA2 train (1024 authentic/1024 tampered), 128 Casiav1 + 128 Columbia each epoch, 64 main-training-disjoint authentic probes. Two GPUs, micro6/accumulation1/global12, evaluation micro8 and four metric threads per rank. At most ten pilot epochs or four cumulative hours. No new model result at registration.
- Full gradient/parameter/reference guards and eight reversible engineering updates precede training. Three consecutive fixed effectiveness/false-positive gates trigger one same-checkpoint complete two-set confirmation, reusing the current subset predictions. Only confirmed effectiveness permits a separately registered full-data restart.
- `metrics_per_epoch.jsonl` uses the user-requested epoch/dataset/selection structure with `average_test2`. `selection_protocol=test_selected`; this is development feedback, not independent generalization evidence.

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

## 2026-09-15 TECT reference collapse: stopped and repaired

- A remains `INTERRUPTED` after epoch15 (16 complete reference epochs, optimizer_step11968); both ranks and controller exited, GPU/run locks released. The frozen epoch15 checkpoint and source remain unchanged. Calibration and main training did not start.
- Diagnostic evidence identifies unnormalized reference paths amplifying intermediate features while injected-noise dependence vanishes. The first triggering optimizer update is not recoverable from the retained every-50-step logs/checkpoint; BF16 alone is not established as the cause.
- `TECT-DIFF-FULL-R512-S42-REFNORM-V2` is prepared with normalized reference transition/projection paths and explicit training/fixed-final health gates. No new formal run exists and no historical checkpoint was reselected.
- A bounded 512/BF16/two-rank regression completed1024 updates using64 training-role authentic images: fixed-probe four-scale MSE0.1192773/0.1251472/0.1421247/0.1841749. This is engineering learning evidence, not full20-epoch stability or localization performance. See [repair report](analysis_reports/reference_repair_20260915.md).
- selection_protocol=test_selected remains the registered main-run selection rule; this task did not evaluate All8 or compare against a full matched baseline.

## 2026-09-15 REFNORM-V2 complete restart authorization

- User authorized a fresh complete repaired TECT run: `TECT-DIFF-FULL-R512-S42-REFNORM-V2-20260915-B`, config `configs/tect_diff/full_r512_s42_refnorm_v2.json`, [restart and early supervision](configs/tect_diff/RESTART_V2.md).
- Execute fresh reference epochs0–19, training-only calibration, then main epochs0–99 with all registered mechanisms and selection_protocol=test_selected. Original A stays stopped; no old reference or diagnostic training checkpoint is reused.
- A 30-minute task covers the reference stage and first three complete main epochs, stops the new run on confirmed abnormalities, performs bounded versioned repairs, and pauses itself after recording healthy early training. Server training/checkpoint/report control remains autonomous.
- Registration before launch: run not yet dispatched; actual launch/source/process/data evidence will be appended after server verification.
- B实际于2026-09-15 17:38 Asia/Shanghai启动，执行源码`100d46edb1f73dd59fa0d57277e07c2925c53278`；143个冻结文件校验、512双卡preflight通过。参考epoch0–1完成、step1496，健康检查通过。随后按用户追加资源优化要求暂停做有界对照，已退出并释放锁；此处是工程暂停，不是参考退化。

## 2026-09-15 measured performance continuation

- Registered C: `TECT-DIFF-FULL-R512-S42-REFNORM-V2-PERF-20260915-C`; [protocol and state transfer](configs/tect_diff/PERFORMANCE_CONTINUATION.md). Same configuration hash `0b0358aa4908b852bb53113467184483fbb11d4be517eb58548ba80757052f88`; inherit B's complete reference epoch1/step1496, continue epoch2–19, then calibration and main0–99.
- Versioned synchronization/statistics scheduling changes passed two-rank bounded parity for model/optimizer/RNG/loss/gradient/health moments. CPU affinity did not demonstrate a useful improvement and was reverted. No scientific parameters or reference/main budgets are changed; this is one continued scientific arm.
- The 30-minute `tect-v2` early-supervision task follows C after launch. Actual transfer, startup and end-to-end throughput evidence will be appended after verification; selection_protocol=test_selected.
- C于2026-09-15 18:00:33 Asia/Shanghai完成状态导入并派发；执行提交`f9c1b90d2fc19c914dd3998f49c2c6665d580ca2`。仅source_commit字段变化，其余恢复状态逐项相等；B checkpoint SHA256 `df51ace1ebccc1bd9177b83e164b4d9bac9a8eeeaef0c606ede5bcf60852d220`未变。A/B均保持hold、无存活训练进程。
- 18:04:49实际C为REFERENCE、epoch3、step2450；controller521647、torchrun522370、两rank522403/522404存活，两GPU/run/controller锁均持有。149个冻结文件校验、新快照双卡preflight和完整恢复通过；[启动收据](analysis_reports/tect_diff/refnorm_v2_startup.json)。参考前三轮完成且健康，尚未覆盖旧版约4200-step退化位置，校准和主训练仍待自动推进。
- 正式参考窗口约84.68→85.78 images/s，仅约1.3%的观测变化，不宣称显著端到端加速；驻留输入对照约9%改善及其数值等价证据见[性能收据](analysis_reports/tect_diff/refnorm_v2_performance.json)。20项CPU异常/状态转移检查通过，有界GPU诊断未保留权重。`tect-v2`已更新为每30分钟检查C，完成健康的参考20轮、校准及main0–2后暂停监督。
- 18:11:51交接检查：C已完成参考epoch0–5，正在epoch6、step4700，超过旧A约4200-step退化点；epoch5四尺度MSE/zero_MSE为0.07265/0.09776/0.13720/0.19496，噪声相关性0.9630/0.9499/0.9289/0.8973，健康状态LEARNED、无失败、无AMP跳步。两rank及控制器身份保持，149个冻结文件仍匹配；这仍不是20轮最终参考检查或定位结果。

- [TECT-Diff TECT-DIFF-FULL-R512-S42-REFNORM-V2-20260915-B](analysis_reports/runs/TECT-DIFF-FULL-R512-S42-REFNORM-V2-20260915-B/report.md); test_selected All8 F1; fixed final reported separately.

- [TECT-Diff TECT-DIFF-FULL-R512-S42-REFNORM-V2-PERF-20260915-C](analysis_reports/runs/TECT-DIFF-FULL-R512-S42-REFNORM-V2-PERF-20260915-C/report.md); test_selected All8 F1; fixed final reported separately.

## 2026-09-15 early monitor AMP/DDP main repair

- C的参考20轮、epoch19实际最终权重probe及2048图训练集校准已通过；主epoch0的两rank梯度范数多次不一致，11:24:25 UTC确认控制器/worker退出、两GPU资源释放并保持hold。C无完整主epoch/checkpoint/All8，部分主训练无效，日志和冻结来源保留。
- CPU及实际512/BF16双卡复现定位为no_grad自条件预测污染外层AMP权重缓存，造成梯度张量从856降至263，并触发旧DDP累积同步问题。缓存隔离恢复完整反传；原实现8-update对照出现参数分歧，修复版8次均保持两rank梯度/参数hash一致。见 [修复边界](configs/tect_diff/AMP_DDP_REPAIR.md)。
- 登记修复D：`TECT-DIFF-FULL-R512-S42-REFNORM-V2-AMPFIX-20260915-D`；配置hash不变，复用C原hash参考/校准，main按seed42重新初始化并执行epoch0–99。发布及正式派发以实际启动收据为准；A/B/C均不恢复，不增加科学对照臂。selection_protocol=test_selected。
- D于2026-09-15 19:51:01 Asia/Shanghai完成依赖导入并派发，执行提交`84fcf47e5721317d99fe186feec08d365fc3f72c`，159个冻结文件校验通过。最终保护版本另8次双卡更新的梯度和参数hash均一致，末次优化器hash一致；30项CPU测试通过，独立复核无阻断项。全部24次诊断更新不保存训练权重，参考epoch19和2048图校准原artifact hashes保持不变。
- D双卡preflight、实际main probe完整梯度覆盖通过；19:54:39主epoch0/step100两rank梯度张量均856、范数一致、无AMP跳步。控制器599808、torchrun600544、rank600611/600612存活，两GPU/run/controller锁持有；A/B/C均停止。step50–100实测6.6744 images/s；14.56秒GPU利用率均值57.875%/69.75%，CPU I/O等待0.058%，不据短窗口或失效C主训练宣称加速。见[启动与资源收据](analysis_reports/tect_diff/ampfix_startup.json)及[修复证据](analysis_reports/tect_diff/amp_ddp_repair_20260915.json)。
- `tect-v2`已更新并核验为ACTIVE，每30分钟跟随D；完成健康main0–2及各轮All8后暂停监督，服务器继续main至99。目前没有完整主epoch或收敛结论。临时验证脚本和候选源码归属/hash核验后已清理，小型证据保留。

- [TECT-Diff TECT-DIFF-FULL-R512-S42-REFNORM-V2-AMPFIX-20260915-D](analysis_reports/runs/TECT-DIFF-FULL-R512-S42-REFNORM-V2-AMPFIX-20260915-D/report.md); test_selected All8 F1; fixed final reported separately.

## 2026-09-15 user-authorized micro6 fresh MAIN restart

- D于13:02:25 UTC因用户调参要求健康停止，最后主epoch0/step3150，无完整主checkpoint或All8；A/B/C/D均保持hold，所有原始拟合与冻结来源保留。
- 用户进一步明确每卡6张，工程测量通过：micro6/accum1/global12、LR1e-4，10.0515 images/s，比micro4快10.68%、原micro2快58.98%，张量峰值16.76GiB/卡；完整856梯度及双rank参数/优化器hash检查通过。测试选每卡8张、CPU指标线程4/队列8，64张训练图的端到端6.9186 images/s，为batch1/线程0的2.0366倍。更大测试batch更慢。
- 登记唯一替代运行 `TECT-DIFF-FULL-R512-S42-REFNORM-V2-MEMB6-20260915-E`，复用C固定参考epoch19与2048训练图校准，主模型seed42/ImageNet重新初始化epochs0–99。global batch8→12会改变优化轨迹；BF16测试批次按版本化数值预算验收，存在小幅概率/二值指标差异，不称exact continuation或数值完全等价。不增科学臂，不用All8调参。见[修订协议](configs/tect_diff/MEMORY_BATCH_RESTART.md)、[工程实测](analysis_reports/tect_diff/memory_batch_performance_20260915.md)。
- 本条为发布前登记，正式执行提交、配置hash、启动及初始健康状态待实际收据补充。30分钟tect-v2监督跟随E完整main0–2及其All8正常后暂停，训练继续到99；selection_protocol=test_selected。

- [TECT-Diff TECT-DIFF-FULL-R512-S42-REFNORM-V2-MEMB6-20260915-E](analysis_reports/runs/TECT-DIFF-FULL-R512-S42-REFNORM-V2-MEMB6-20260915-E/report.md); test_selected All8 F1; fixed final reported separately.

- E于22:10:25 Asia/Shanghai正式派发，执行提交`ad8fb6e5f3f6f295b1e94b87e28175a5fabc779c`、配置hash`63e53b3e50df4d4798d9310cb4ca2065efee1fbaea4fd2ba485fb88a37116275`。22:14两rank epoch0/step100启动核验PASS，171冻结文件及拟合hash正确、856梯度同步、无AMP跳步；正式step50–100为10.070 images/s，nvidia-smi总显存19.33/19.43GiB。四资源/控制器锁持有，A/B/C/D均hold/stopped。36项原生CPU测试、实际启动门控及双卡训练检查通过，见[启动收据](analysis_reports/tect_diff/memory_batch_startup_20260915.json)。当前无完整epoch/All8或收敛结论；tect-v2已绑定E且ACTIVE每30分钟，到完整健康main0–2和All8后暂停监督。

## 2026-09-16 E early foreground saturation and held diagnosis

- 上述E启动状态是历史记录。E完整main epochs0–1的自然All8预测正像素99.8368%→99.9993%，真实7.7854%；持续严重退化触发受控停止，2026-09-15 18:22:48 UTC controller/worker均退出。保留完整epoch1/step8102 last、best0、冻结171文件；部分epoch2最后日志step10200不计入完整epoch。没有final99或健康收敛结论。
- 16张登记训练图的只读前向对照：保存BN统计时原10步末输出99.6924%正像素，临时无更新批统计为4.9307%，GT7.2411%；j0尚无history/gamma校正已严重偏正。两rank15个BN计数均8102，完整模型/缓冲区/RNG/参考/校准/hash保持，未发现额外BN更新或源码偏离冻结协议。证据证明有界归一化敏感性，不证明GN修复有效或最终定位质量。
- E保持`HELD_PENDING_NORMALIZATION_PROTOCOL_DECISION`，未启动新正式模型。已准备TECT-only15层GroupNorm8替代版本、训练图有界验证和fresh MAIN提案；架构修订仍需明确范围。详见[诊断及具体提案](analysis_reports/tect_diff/main_foreground_diagnostic_20260916.md)和[数值收据](analysis_reports/tect_diff/main_foreground_diagnostic_20260916.json)。selection_protocol=test_selected；没有新增All8推理或checkpoint重选。

<!-- pilot:TECT-PILOT-CASIA2-GN8-R512-S42-20260916-A:begin -->
- [TECT pilot TECT-PILOT-CASIA2-GN8-R512-S42-20260916-A](analysis_reports/runs/TECT-PILOT-CASIA2-GN8-R512-S42-20260916-A/report.md): FAILED / HOLD; selection_protocol=test_selected; complete epochs=0; full training not started.
<!-- pilot:TECT-PILOT-CASIA2-GN8-R512-S42-20260916-A:end -->

<!-- pilot:TECT-PILOT-CASIA2-GN8-R512-S42-20260916-B:begin -->
- [TECT pilot TECT-PILOT-CASIA2-GN8-R512-S42-20260916-B](analysis_reports/runs/TECT-PILOT-CASIA2-GN8-R512-S42-20260916-B/report.md): COMPLETED / NO_GO; selection_protocol=test_selected; complete epochs=10; full training not started.
<!-- pilot:TECT-PILOT-CASIA2-GN8-R512-S42-20260916-B:end -->

<!-- pilot:TECT-PILOT-CASIA2-GN8-N8192-R512-S42-20260916-C:begin -->
- [TECT pilot TECT-PILOT-CASIA2-GN8-N8192-R512-S42-20260916-C](analysis_reports/runs/TECT-PILOT-CASIA2-GN8-N8192-R512-S42-20260916-C/report.md): COMPLETED / NO_GO; selection_protocol=test_selected; complete epochs=10; full training not started.
<!-- pilot:TECT-PILOT-CASIA2-GN8-N8192-R512-S42-20260916-C:end -->
