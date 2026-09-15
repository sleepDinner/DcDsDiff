# TECT reference repair: REFNORM-V2

2026-09-15 用户要求停止已经退化的第一阶段真实图训练并定位修复。本轮交付是停止、修复和有界工程验证；**尚未启动新的正式训练**。

新配置：`full_r512_s42_refnorm_v2.json`；协议 ID：`TECT-DIFF-FULL-R512-S42-REFNORM-V2`。除以下参考网络稳定性与健康检查修订外，继承 [原协议](PROTOCOL.md)。完整诊断和实测见 [修复报告](../../analysis_reports/reference_repair_20260915.md)。

## 原运行保持停止

`TECT-DIFF-FULL-R512-S42-20260915-A` 在 epoch15 完成后停止，共16轮、11968次更新。其 epoch5 后段发生退化；原代码快照、配置、manifest、epoch15 checkpoint 和历史指标保留。该 checkpoint 不作为新版本的初始化，不恢复原版本，不重选早期表现较好的 checkpoint。校准和正式定位训练尚未开始。

## 明确的版本变更

参考网络版本为 `tect-reference-pixel-unet-v2-stable-paths`：

- 三个下采样和三个上采样卷积的输入增加 GroupNorm。
- 改变通道数的 residual skip 投影使用归一化输入，避免绕过规范化的连续放大。
- 参考 residual 合并乘 `1/sqrt(2)`，降低深路径的重复增幅。

宽度、PixelUnshuffle/Shuffle、真实 ε MSE、λ=[2,3,4,5]、输入512、AdamW 2e-4/0.01、全局batch16、20轮固定终点和参考冻结边界保持。主模型 Image adapter 共享的 residual 类默认保持旧行为。旧配置缺少 architecture_version 时仍实例化 v1，旧权重严格读取；v2 权重不能冒称 v1 exact resume。

GroupNorm 的统计定义参见 [PyTorch 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html)。上述放置位置和残差缩放是针对本项目实测问题的工程修订，不宣称来自原论文。

## 训练内退化保护

策略 `REFERENCE-NOISE-HEALTH-v1` 仅由新配置启用：逐尺度记录真实 ε MSE、零预测 MSE、预测能量和预测与噪声的非中心化相关性，先按每图 channel/pixel 平均，再在两 rank 汇总。

所有尺度曾达到 `MSE/zero_MSE < 0.98` 后，如连续两轮所有尺度均 `>=0.98`，保存失败指标和完整末轮状态后阻断。最终epoch19的每尺度训练误差也必须严格小于0.98倍零预测误差。该阈值是明确的工程退化检查，不用于提前选择checkpoint，不用于扩大训练预算。

为防最后若干次更新被整轮均值掩盖，固定最终权重另做一次训练内检查：取冻结 reference manifest 的前16图，仍属于完整训练集；关闭增强；seed42，按稳定ID产生两份独立噪声，各自跨四尺度共享。两rank无padding、每rank micro2，记录每尺度16个图像均值（32个图像/噪声观测），要求全部 `MSE/zero_MSE < 0.98`。

最终检查保存并恢复 RNG 和模块模式，不修改参数、不挑选权重。结果绑定实际最终参数hash、架构和reference manifest；artifact、receipt及下游加载均核验绑定。任何失败均阻断校准/主训练。没有验证集、测试统计或All8推理参与这些检查。

## 本轮验证边界

新网络通过512、BF16、双卡DDP、全局batch16的1024次固定诊断更新，使用64张已登记的真实训练角色图。另验证旧版本逐值兼容、严格加载、冻结边界和已退化旧权重的末轮检查拒绝。临时模型仅存在于内存；未保留诊断checkpoint。

1024次诊断更新未覆盖原运行约4200次更新的退化时点，也没有完成20轮全量reference训练。证据支持修复后的路径可正常学习和退化检查有效，不能保证完整训练不再出现任何退化，不能视为定位精度或创新收益结果。

后续正式执行需要新的运行身份、已发布源码快照及新参考初始化。不得修改或续跑已停止A的冻结快照来套用本修复。

selection_protocol=test_selected：继承的正式主训练仍按八测试集逐epoch Pixel-F1选择checkpoint；这些测试集不是独立泛化评估。本轮没有All8结果，也没有同协议完整baseline比较。
