# TECT E：前期全前景退化、受控停止与归一化诊断

状态：**HELD_PENDING_NORMALIZATION_PROTOCOL_DECISION**。E 已停止，未启动新的正式模型。完整主训练仅完成 epochs 0–1；不是 100 轮完成或正常收敛。机器可读证据见 [诊断收据](main_foreground_diagnostic_20260916.json)。

现有 `tect-v2` 的30分钟任务已设PAUSED并保存诊断结论，等待归一化修订范围确认；暂停不是“前期已健康通过”。新运行获准并登记后再绑定恢复，避免反复唤醒和误恢复E。

## 已完成的资源优化与异常

用户要求的每卡 6 张已用于 E：micro6、accum1、global12、LR1e-4。此前有界训练吞吐 10.0515 images/s，比 micro4 的 9.0818 快 10.68%；正式首个训练窗口 10.070 images/s，约 19.3 GiB 显存/卡。不同时间前缀工作量会影响短窗口，不能据此保证每轮固定加速。每轮测试使用 micro8、CPU 指标线程4/rank、队列8；[完整性能报告](memory_batch_performance_20260915.md)保留候选和精度边界。

监督读取已自然完成的 All8，没有追加测试推理：

| 已完成 epoch | 预测正像素比例 | 真实正像素比例 | specificity | All8 macro Pixel-F1 | macro MAE | 完整 All8 耗时 |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 99.8368% | 7.7854% | 0.00164993 | 0.20318136 | 0.87346535 | 1413.34 s |
| 1 | 99.9993% | 7.7854% | 0.00000716 | 0.20274015 | 0.87721033 | 1260.29 s |

两轮均完整覆盖 4295 图/8 库；正像素比例由总 TP/FP/TN/FN 汇总，F1 是八库逐图均值的等权平均，MAE 也是八库等权均值。像素比例与 F1/MAE 的汇总权重不同。连续严重且加重的全前景退化触发停止，不能仅以低 F1 或单批 loss 波动判故障。

E 于 **2026-09-15 18:22:48 UTC（北京时间 09-16 02:22:48）**受控停止。最后轻量日志为 epoch2/step10200，部分 epoch2 未保存为完整训练状态。保留：

- `last.pth`：epoch1、next_epoch2、step8102、epoch_complete/evaluation_complete均真；含模型、856项 Adam 状态、scheduler、scaler、两rank RNG及sampler。SHA256 `99f1dc4fa6c7fbb3cfa2cb371aa82badc25fd017f78995dfe2fbb5006c2a7354`。
- `best.pth`：按原规则仍为 epoch0，SHA256 `1962c2d7e3cc9f1b49af945e6cea036b14f6637170ec773387fc4d319a98777c`。没有重选 checkpoint。
- 冻结来源 `ad8fb6e5f3f6f295b1e94b87e28175a5fabc779c`，171文件核验保持；配置 SHA256 `63e53b3e50df4d4798d9310cb4ca2065efee1fbaea4fd2ba485fb88a37116275`。
- A/B/C/D继续保持hold。E的controller/torchrun/rank均退出；诊断结束后双GPU与run锁可重新获取，GPU无计算进程。

## 有界训练图因果对照

从原登记训练manifest，以 seed42 的固定ID哈希顺序选择16图（8真实/8篡改，原图≤4MP），每rank 4真实+4篡改，BF16、512输入、batch8。所有数值使用512训练mask有效像素，不能与原尺寸All8逐图分数直接比较。这些标签只用于构造诊断加噪输入和统计，原10步采样仍没有GT参数。

对照只改变 task network 的15个BatchNorm层：A使用checkpoint保存的运行统计；B使用当前batch统计的函数式前向，**不更新运行统计、不更新权重、不保存诊断权重**，Dropout/DropPath及其他模块保持eval。四个单步时间用相同图、mask噪声、独立Image噪声和参考/校准；自条件对照复用A生成的同一历史张量。完整轨迹直接调用冻结 `sample()`，每步历史由各自真实前一步预测生成，没有替换DDIM。

| 诊断场景 | 已保存BN：预测正像素 | 临时批统计：预测正像素 | 真实正像素 |
|---|---:|---:|---:|
| GT加噪，t=0.2，无历史 | 7.2893% | 7.2627% | 7.2411% |
| GT加噪，t=0.5，无历史 | 35.5784% | 7.2680% | 7.2411% |
| GT加噪，t=0.8，无历史 | 99.1742% | 7.7689% | 7.2411% |
| GT加噪，t=0.95，无历史 | 99.3460% | 8.8705% | 7.2411% |
| 原10步采样第一步 j0 | 98.5499% | 3.4912% | 7.2411% |
| 原10步采样末步 j9 | 99.6924% | 4.9307% | 7.2411% |

第一步没有历史预测、gamma为0；base和ctrl相同，退化已经存在。因此证据校正和迭代历史不是这批图上偏差的起点。单步t0.2可从含较强GT信号的输入得到很低误差，也解释了为什么低训练loss不足以判断自由采样正常。

两rank所有15个BN的 `num_batches_tracked` 均为8102，恰好对应实际完整更新数；未发现额外BN更新、推理误开train、标签/指标反转、重复sigmoid或DDIM时间/符号的源码偏差。保存统计模式下，部分Mask后部层的输入均值偏移达运行标准差的6倍以上；对照是所有15层一起改变，不能据此判定某一个层为唯一原因。

诊断两rank约16.58/16.10秒，张量显存峰值约5.92GiB。完整模型参数/缓冲区、reference、calibration、RNG和checkpoint文件均核验不变，无梯度或优化器构造。该耗时含统计hooks和不同诊断前向，不是新的吞吐基准。

**支持的结论：这份检查点在这16张固定训练图上的严重全前景输出，对BN统计选择具有直接因果敏感性；只改变这些归一化统计即可消除该有界样本上的近全前景饱和。** 尚未证明更小训练batch能避免问题、具体哪层主导、重新训练能否最终收敛、或任何正式修复的All8表现。

临时批统计末输出仍完全漏掉8张篡改图中的4张，并在8张真实图中的2张产生误报。整体正像素占比下降不能代替逐图定位质量验收，更不能据此宣布修复成功。

## 已准备的修复提案，尚未实施

建议对TECT主网络登记一个独立的归一化修订版本，以**GroupNorm、8组、eps1e-5、可学习逐通道仿射参数**替换上述15个task BN：MMFF的8个reduce归一化、MSFF fusion的1个、Mask down/up/pred的6个。组数使用本项目参考/Mask残差块已有的8组约定，不做分组数搜索。只作用于TECT实例，原始DcDsDiff基线及E冻结快照保持。

[PyTorch GroupNorm文档](https://docs.pytorch.org/docs/2.1/generated/torch.nn.GroupNorm.html)说明其训练和评估都从当前输入的组内计算统计；[BatchNorm2d文档](https://docs.pytorch.org/docs/2.1/generated/torch.nn.BatchNorm2d.html)说明当前BN训练用批统计、评估用运行统计。这使GN成为消除这类模式切换的具体候选，**不是已验证成功的修复**。不直接部署本次临时batch-stat模式，因为这会使一张图的预测依赖同batch其他图。

若明确允许该架构修订，执行范围为：

1. 增加版本化TECT配置/architecture标识与严格15层替换门控；默认旧配置仍走原BN，不改历史权重或manifest。
2. 先做最多64张登记训练图、最多256次更新的有界双卡验证，检查完整梯度、跨rank参数/Adam一致、参考/校准隔离、原采样和批次独立性、各前缀有限性及新版本实际吞吐/显存。失败保留证据并停止，不循环调参，不用All8挑选实现。
3. 通过后，只登记一个替代MAIN：seed42/ImageNet重新初始化epochs0–99，每卡6/accum1/global12、LR1e-4，逐轮测试8/线程4/队列8；不继承E训练权重。
4. 复用同一C参考epoch19和2048图校准，保持全部reference/loss/evidence/DDIM/All8规则。此次不增加对照、种子、消融或额外校准拟合；E保持失败前期诊断的历史结果。
5. 将现有30分钟监督绑定新运行，验证完整健康main0–2及对应All8后暂停监督；不把启动或短诊断通过等同于收敛。

需要明确这一修订范围的原因：项目 [AGENTS.md](../../AGENTS.md) 授权的是保留科学设置的有界工程修复，且 [PROTOCOL.md](../../configs/tect_diff/PROTOCOL.md)明确继承实际Mask down2/up2/pred2架构、[AMP_DDP_REPAIR.md](../../configs/tect_diff/AMP_DDP_REPAIR.md)登记rank-local BN统计。当前证据没有发现实现偏离这些规则；GN替换是架构变化，不能伪称对原协议的等价工程修复或exact resume。上述提案目前不构成新运行登记或派发授权。

`selection_protocol=test_selected`：All8参与checkpoint选择，不是独立泛化评测。本诊断没有新增All8推理、未选择更有利的历史checkpoint，也没有完整同协议baseline比较。
