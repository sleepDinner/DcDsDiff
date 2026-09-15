# TECT 训练与逐轮测试资源优化重启

## 用户修订与运行身份

2026-09-15 用户明确要求停止当前训练、调大每卡批次并重新开始，同时优化训练和每个 epoch 的完整测试速度。D (`TECT-DIFF-FULL-R512-S42-REFNORM-V2-AMPFIX-20260915-D`) 于13:02:25 UTC健康停止，最后记录主epoch0/step3150，没有完整主epoch、主checkpoint或All8结果；其冻结来源、日志、原始依赖及hold全部保留。此次停止由用户调参请求触发，不是C曾发生的AMP/DDP故障。

新运行登记为 `TECT-DIFF-FULL-R512-S42-REFNORM-V2-MEMB6-20260915-E`，配置 `full_r512_s42_refnorm_v2_memb6.json`，协议 `TECT-DIFF-FULL-R512-S42-REFNORM-V2-MEMB6`，group仍为 `TECT-DIFF-FULL-REFNORM-V2`。实际来源提交、测量参数和启动状态以发布后的工程验证及启动收据为准。A/B/C/D保持停止，E是本次授权主训练的替代重启，不增加科学对照臂。

## 训练批次与拟合依赖

- 用户随后明确提出每卡6张。micro6实测比micro4快10.68%，全梯度和双卡参数/优化器一致性检查通过，因此主训练由每rank micro2/accum2改为micro6/accum1，两GPU的有效global batch由8变为12；学习率仍为1e-4，无线性放大学习率。micro4仅是未登记的工程候选，没有另起正式E4运行。每epoch优化器更新次数随global batch变化，仍遍历完整训练集并执行100轮。
- microbatch变化会改变BatchNorm的批内统计，以及批级随机时间、自条件与前缀分组。因此E按seed42及登记ImageNet权重新初始化主模型，从epoch0/step0开始执行全部100轮，不继承D部分main状态，也不称为数值等价续训。
- 复用已完成的C参考固定epoch19（SHA256 `800f50c392a275fde3ec249c6f52275c6db79813e6f8b7e8b534f9ddb52a523f`）与2048训练图拟合校准（文件SHA256 `bb349a143385b25798fe0a9c7c43904ba08c26f38ed7a6aee991add7ed16f59c`）。不重训参考、不重做校准，不变更artifact字节或拟合metadata。
- `batch_restart.json`保存D→C原始依赖来源、原始收据及hash；仅两份小型依赖收据的protocol_id/config_hash重绑至E。训练manifest、reference/evidence配置及artifact绑定必须仍通过验证。
- 保留AMP自条件缓存隔离、每次更新完整856梯度覆盖和跨rank同步检查，以及checkpoint前参数hash一致性保护。

## 工程优化与测量边界

候选优化缓存固定分辨率的候选位置、有效边界和空间排除掩膜，减少每张图重复计算；另比较按整幅图查询的相对响应计算与原分块实现。缓存为非持久私有状态，不进入state_dict，不消耗随机数，并随device迁移失效。不得删去证据机制或改变KNN候选、邻居、2048查询分块与参考轨迹；相对响应的独立像素查询调度可在有界数值验证通过后优化。

测试侧增加有界CPU指标流水线：GPU仍执行原采样、原尺寸双线性恢复与CPU拷贝，CPU线程只处理独立NumPy数组并调用原per_image_metrics。结果按提交顺序收集，队列排空后才执行分布式汇总。工程候选测试batch为1/2/4/8/16/32、metric_workers为0/4；最终新配置固定micro_batch8、metric_workers4及metric_queue_limit8，旧配置默认同步行为；相同输入数组的CPU指标必须逐值一致。

初轮工程检查采用额外严格的概率误差1e-5及零二值分歧门槛，较大BF16 batch没有通过，原始收据保留为失败的严格等价检查。16张训练图上micro8最大概率差0.00582081，957/13,065,738有效像素的二值结果不同（0.0073245%，按首pass去重）；每图F1最大差4.55196e-6，Boundary-F1最大差0.00326582。不能称指标完全等价，也不能把两次重复pass累计的1914处差异称为独立像素数。

该门槛是工程初始假设，不是用户冻结的科学条件。用户明确授权重调训练及逐轮测试batch，E尚无正式main/All8结果。独立源码审阅确认eval中的BatchNorm使用固定running statistics，归一化、注意力窗口、KNN和稳定ID噪声均按图独立；[PyTorch2.1说明](https://docs.pytorch.org/docs/2.1/notes/numerical_accuracy.html)也不保证数学相同的batch与逐图浮点计算逐值相同。因此将E的batch选择登记为`BF16_BATCH_EXECUTION_V1`数值执行修订，数学采样/评测规则与BF16精度保持不变，不伪称与D逐值相同。

新检查先用相同16张训练图、关闭matmul和cuDNN TF32的FP32 batch1/8对照排查批次耦合，最大概率差预算为1e-5。FP32只用于诊断，正式精度不变。随后在64张独立训练图上测batch1/8/16/32，预声明BF16最大绝对概率差预算0.0078125（由一个BF16 epsilon量级取值的经验接受预算，绝非10步扩散的理论误差界）。完整报告二值差异、有效像素分母及逐图F1/IoU/Boundary-F1/MAE差异；候选需有限、ID/初始噪声一致、模型/RNG保持。选择依据为完整吞吐、显存和数值预算，不按F1高低选择batch。在正式E启动前固定选定配置，后续All8不能用于调batch或误差预算。

所有候选测速使用登记训练图，不使用All8得分选择实现。训练测量使用实际数据加载器、H2D、前后向及原优化器/同步保护；推理比较micro1/2/4/8、显存峰值、稳定ID输出与指标，并比较实际加载器加完整CPU指标的端到端耗时。验证有独占双GPU及诊断run锁、执行超时和失败收据，不保存诊断权重或预测图。第一次启动与稳定吞吐、预载推理与完整测试耗时分别报告；短样本不外推为完整epoch的确定加速倍数。

初次跨运行训练状态逐字节检查未通过，随后同一原实现重复8次更新也未通过：相同初始化、输入和RNG仍产生不同反向梯度，因而不能将此检查作为优化等价门槛。活跃模型含可求导CUDA bilinear resize与AdaptiveMaxPool2d，其反向潜在非确定性见[PyTorch2.1官方说明](https://docs.pytorch.org/docs/2.1/generated/torch.use_deterministic_algorithms.html)；具体主导kernel未单独定位，正式确定性设置没有改变。保留这两次诊断失败收据，不宣称跨运行全部训练状态等价。

独立隔离证据路径的双rank验证已通过：16张登记训练图、真实已训练参考、相同ID噪声下，原实现重复、geometry-only及geometry+full-query的上下文、四尺度相对响应和全部prefix输出均逐字节一致，reference/calibration/RNG不变。这证明本次计算重排的有界前向一致性；同次训练中的完整梯度覆盖、跨rank参数一致性仍为硬性保护。

## 最终实测选择

训练micro6为10.0515 images/s，micro4为9.0818，原micro2为6.3224；32次计时更新前另有4次预热。micro6张量峰值16.76GiB、缓存保留峰值18.68GiB/卡，无OOM或AMP跳步，36次更新均完整856梯度覆盖和跨rank范数/count一致，末端参数及Adam状态hash一致。mean prefix为2.8594（micro4为2.9844），短窗口工作量不同；59%的吞吐提升不是严格等工作量因果估计，也不是全epoch保证。

FP32 batch1/8控制最大概率差4.1723e-7，通过。64张训练图的BF16完整加载/采样/原尺寸恢复/CPU指标测量中，batch8/workers4为6.9186 images/s，batch1/workers0为3.3971（2.0366倍），batch16/workers4为5.9224，batch32/workers4为2.7846；更大batch均未OOM，但更慢。选定batch8最大概率差0.0071094；首pass独立像素二值差2520/33,508,878（0.0075204%），每图F1/IoU/Boundary-F1/MAE最大绝对差分别3.3430e-5/3.4523e-5/0.00370095/4.6362e-5。正式测试保留BF16，此版本不宣称跨batch逐值等价。

## 完整实验与早期监督

科学数据、512输入、损失、参考噪声/轨迹、10步Mask DDIM、阈值0.5和All8选模规则不变。每轮对八库4295张图各评测一次，无DDP padding，使用稳定ID噪声、last-step P_ctrl、sigmoid后原尺寸恢复。只有完整有限八库结果可原子更新best；严格提升胜出、平分保留更早轮，final固定epoch99。

现有每30分钟的 `tect-v2`任务在E正式登记后跟随E；明确识别D的计划停止，不能恢复旧运行。监督覆盖继承依赖健康及E完整main epochs0–2和其逐轮All8；确认正常后暂停定时任务，服务器控制器继续全部训练。异常按用户已授权范围停止归属明确的进程、保留证据并做有界版本化工程修复。正常不通知，不使用本地常驻SSH轮询。

`selection_protocol=test_selected`：八个测试集参与checkpoint选择，不是独立泛化评测。吞吐提升与前期健康不代表最终收敛、创新收益或优于不存在的完整同协议baseline。
