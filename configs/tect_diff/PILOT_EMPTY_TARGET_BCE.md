# TECT GN8：空训练目标的 BCE 损失修订

2026-09-16，在 C 的固定最终检查点诊断完成后、D 训练结果产生前登记。用户已授权监督后的有界框架修复与CASIA2/Test2稳定性验证。

## 唯一新训练

- Run：`TECT-PILOT-CASIA2-GN8-EMPTYBCE-R512-S42-20260916-D`。
- Protocol：`TECT-PILOT-CASIA2-GN8-R512-S42-DATA2-N8192-EMPTYBCE-V1`。
- Config：[pilot_casia2_gn8_data2_n8192_emptybce_r512_s42.json](pilot_casia2_gn8_data2_n8192_emptybce_r512_s42.json)。相对C仅protocol_id及新增training.mask_loss=`empty_target_bce_v1`。其他字段逐项严格绑定C规范配置hash。
- 新seed42/ImageNet MAIN，GN8物理网络不变，不读取C/B/E MAIN权重，不续训C。所有旧运行、来源、权重及结果保持原状。

## 依据与不确定性

C最终epoch9的固定16张训练图诊断中，原采样8张Tp首步/末步F1为0.817681/0.807603；只移除末步历史为0.807636。没有证据表明严重退化由该批图的自条件历史造成。8张Au均无误报，平均前景概率约4.65e-7，但空目标的原IoU项仍为0.108731。

原空目标IoU为 `S/(S+1)`，S是512²个前景概率之和。固定该前向输入和历史时，末层公共偏置的平均mask损失导数为+0.04302535；解析去除空目标IoU后为-0.00542876。说明原目标仍在正确为空的真实图上施加较强的继续压低输出的压力。它是可检验的局部机制，不证明是跨库漏检的唯一原因，也不是实测BF16优化器更新方向。[完整诊断](../../analysis_reports/tect_diff/c_nogo_diagnostic_20260916.md)。

## 精确损失定义

每张图先按原公式计算WBCE及WIoU。新mask loss为：

`mean_over_images(WBCE + I[target has any nonzero pixel] * WIoU)`。

- “空”只指**实际512训练目标张量严格全零**，不是由缺失mask或预测结果推断真实身份。真实Au继续按已验证的显式身份配零mask；若某个原生Tp经缩放恰好变成全零，该训练目标也使用BCE-only。非零软目标即使低于0.5仍视为非空，不新增阈值。
- 每张空图继续贡献原WBCE。每张非空图保持原WBCE+WIoU及梯度；最终仍以整个batch的图数平均，不按Tp数重新归一化、不改类别权重。
- 原Image MSE、lambda_image、参考网络、校准、轨迹证据、时间分布、自条件、十步DDIM、0.5阈值与末步终点均不变。
- 原配置缺省走`structure_v1`原损失；未知、空串或None策略拒绝。D配置必须明确选新策略。
- fitting复用保护只允许diffusion.py的一条损失导入和一条训练调用改变；逆向恢复两处后必须逐字符匹配原E的受保护源码。其他forward及reference_path/task/control/sample一律不可改变。

## 固定数据、预算与晋级

全部九种清单与C逐项同hash，包含8192训练图4096Au+4096Tp、工程训练64图、64真实图probe、128+128 quick Test2、920+180完整Test2及既有reference/calibration。C必须保持已完成NO_GO且进程退出，C来源、数据bundle与终态收据hash绑定后才允许D启动。

双4090、512、BF16、micro6/rank、accum1、global12、AdamW LR1e-4及原100轮cosine的前10轮；每轮最多683次更新，最多10轮/4小时。测试micro8/rank、指标线程4/rank、队列8不变。八次可逆工程更新、完整856梯度覆盖、rank参数/Adam一致、参考隔离及fresh状态/RNG恢复通过后才进入正式pilot。

原全部门槛保持：连续三轮健康且平均Test2F1≥.50；每库F1≥.35并超过预先固定全前景基线.10；每库specificity≥.90、recall≥.40；真实图平均和池化误报像素率都≤.01。主要风险是去掉空目标IoU后真实图误报增多，因此不得放宽这些要求或调阈值。

通过quick门槛后只对当时同一checkpoint做一次完整两库确认，复用当前256图，仅新增844图。确认通过才允许另行登记fresh Full；未通过/十轮结束则NO_GO，不追加epochs或数据，不另选更有利checkpoint。D是单个损失候选开发运行，并非保证成功的修复。

沿用原C参考epoch19及训练拟合校准，二者在FinalTrainData拟合，不能称所有学习组件仅见CASIA2。每轮JSONL用average_test2，`selection_protocol=test_selected`：开发测试用于选模及修订，不是独立泛化评估。
