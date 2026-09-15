# TECT CASIA2 / GN8 小规模稳定性验证

2026-09-16 用户授权：监督发现问题时可修改框架/网络修复，今后不再为此重复询问；模型不稳定阶段先使用CASIA2及Casiav1/Columbia，允许用更少图片提高验证效率。以下配置、抽样规则和晋级条件在新模型结果产生前固定。

## 运行与修复

- 协议 `TECT-PILOT-CASIA2-GN8-R512-S42-V1`；run `TECT-PILOT-CASIA2-GN8-R512-S42-20260916-A`；配置 [pilot_casia2_gn8_r512_s42.json](pilot_casia2_gn8_r512_s42.json)。A/B/C/D/E旧运行继续保持hold和原始结果。
- 主网络15个BN改为8组GroupNorm，eps1e-5/affine，架构 `tect-diff-full-v2-main-gn8`。只修改TECT实例；原基线和旧配置仍用BN。Mask、MMFF、MSFF的归一化改变是明确的架构修订，不声称与E等价。
- 主模型重新seed42/ImageNet初始化。复用C固定参考epoch19（SHA256 `800f50c392a275fde3ec249c6f52275c6db79813e6f8b7e8b534f9ddb52a523f`）和2048训练图校准（SHA256 `bb349a143385b25798fe0a9c7c43904ba08c26f38ed7a6aee991add7ed16f59c`），经E原收据追溯。参考/校准不重新拟合。它们原来用FinalTrainData拟合，**不能宣称所有组件仅在CASIA2训练**。
- 512输入、双4090 DDP、BF16、每卡6/accum1/global12、AdamW LR1e-4。保留100轮cosine日程的前10轮，最小LR1e-6；本次最多10轮，不能称100轮完成。两GPU和run/controller锁、冻结已发布源码、完整梯度/参数保护、参考隔离、真实Image MSE、原证据与10步DDIM保持。

## 只读数据与固定子集

CASIA2只读根 `/data1/data/datasets/CASIA2.0/`：Au=7491，Tp/Gt=5123对，`_gt`精确配对。Au的真实性由目录/文件名/列表明确确认，在项目cache建立原尺寸零mask；不根据缺mask推断真实。尺寸不一致、解码失败等显式隔离，Au/Tp标签冲突和Tp全零mask须明确解决。实际可用数以解码审计收据为准。

先排除与完整Casiav1/Columbia的精确RGB或已知命名空间source ID重叠，再固定哈希抽样：

| 角色 | 图片数 | 用途 |
|---|---:|---|
| CASIA2主训练 | 2048（1024 Au+1024 Tp） | 每epoch完整遍历 |
| 主训练内工程probe | 64，真假平衡 | 新机制检查和8次实际更新；之后恢复初始化，不保留诊断权重 |
| Casiav1快速测试 | 128 / 完整920 | 每epoch发展反馈和选模 |
| Columbia快速测试 | 128 / 完整180 | 每epoch发展反馈和选模 |
| CASIA2真实图探针 | 64 | 每epoch观察错误的篡改预测 |

真实图探针与主训练保持CASIA2 source/RGB隔离；它可能已被继承的参考/校准见过，收据逐项统计这种暴露，不能称完全独立留出数据。CASIA1/CASIA2 source命名空间不擅自合并；精确哈希不能排除变换或未知同源泄漏。

逐轮仅320张（两库256+真实图64），测试每卡8、CPU指标线程4/rank、队列8。两库以原尺寸概率恢复、阈值0.5、稳定ID噪声、无DDP padding计算原指标，真实图探针单列，不混入两库F1。

## 启动验证、停止与晋级

启动检查使用真实冻结拟合依赖，验证完整梯度/参考隔离、FP32真实10步采样的逐图/batch2差异≤1e-5，并进行8次双卡实际优化更新（含参数/Adam跨rankhash）。随后恢复全部初始权重、缓冲区、RNG，正式优化器从零开始。FP32仅诊断，正式仍BF16；CPU单层测试和短更新都不证明定位有效。

每个epoch完整保存模型/优化器/scheduler/scaler/两rank RNG/sampler/待评测状态；完成评测后按Test2等权逐图Pixel-F1严格提升选best，平分保留更早epoch。仅完整结果写 `metrics_per_epoch.jsonl`；相同epoch原子覆盖而不重复，损坏日志失败关闭。恢复不重新推理已完整且绑定同一模型的dataset结果。达到第10轮时保存 `pilot_final.pth`，其终点是pilot epoch9，不是原完整模型final99。

以下是保守的发展阶段晋级门槛，不是论文结论。连续**3个完整健康epoch**均需：

- Test2宏平均Pixel-F1 ≥0.50；每库Pixel-F1 ≥0.35，且超过该固定样本的“全部预测为前景”逐图F1均值至少0.10。
- 每库specificity ≥0.90、recall ≥0.40，避免靠全前景或全背景通过。
- 64张真实图的平均及池化误报像素率均≤0.01。
- 无NaN/Inf、AMP跳步、缺梯度或跨rank不一致；完整ID/计数覆盖。

连续3轮近全前景（池化预测正比例≥0.99）或近全背景（≤0.00001）结束本pilot并报告NO_GO；单轮低F1不判工程故障。其余情况最多10轮/累计4小时即结束，未达标为NO_GO。监督可依据保存证据启动新的版本化修复，不在本运行内改配置或根据结果降低门槛。

快速子集连续达标后，仅用**当前epoch**的同一检查点做完整两库确认：Casiav1 920+Columbia180，复用该epoch已算的256张及真实图探针，只推理剩余844张。再次检查相同门槛（单轮），不重选更有利checkpoint，也不重复确认。这个确认含已见子集，不是独立测试。

完整确认通过，输出 `READY_FOR_FULL`；否则NO_GO。服务器不会在此pilot中盲目开始FinalTrainData/All8。30分钟监督据该收据自动安排下一次已授权的版本化完整训练或诊断修复，无需再询问结构修改权限；新的完整运行仍须独立ID、fresh MAIN、完整数据审核和冻结来源。监督不是每轮额外推理或持续本地SSH轮询。

## 输出与结论边界

`metrics_per_epoch.jsonl`参考用户指定DINOv3文件的结构：`epoch_index`、`epoch_number`、`total_epochs`、`metric`、`pixel_f1`（Casiav1/Columbia）、`image_counts`、selection信息。两库均值明确写`average_test2`；附真实图误报、训练loss/吞吐、耗时与gate证据，不伪造`average_all8`或`average_mvss5`。

`selection_protocol=test_selected`。两个测试库参与选模、发展门槛和后续修复，均不能当作独立泛化证据。与它们的历史成绩或正式All8结果不构成完整同协议baseline比较。每次完成/失败/停止自动生成报告、逐轮JSONL和独立pilot台账，并经独立发布worktree同步；新模型尚未启动时不得填入训练数值。
