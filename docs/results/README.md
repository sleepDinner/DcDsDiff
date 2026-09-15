# 2026-09-15 实验结果

GPU 1 的 CASIA2 续训已按用户要求提前结束，完整完成 85 轮（epoch 0–84，包含从父实验继承的 3 轮）。本次直接测试其已有 epoch-52 `model-best.pt`，该文件由 Casiav1、Columbia、NIST16 三集的逐图平均 MAE 选出。GPU 0 完整训练 100 轮；下面另列它的原始末轮复现结果。best 沿用已登记的逐 epoch MAE 选择规则，未进行额外的事后 checkpoint 重选、阈值扫描或种子搜索。

## 八库 best 评估

两个八库评估均已完成。GPU 1 的评估于 **2026-09-15 14:36:01 Asia/Shanghai** 完成，使用提前固定的 epoch-52 best；GPU 0 使用其原 GIT10K MAE 选出的 epoch-2 best。

| 数据集 | 图像数 | GPU 0：GIT10K best F1 | GPU 1：CASIA2 best F1 |
|---|---:|---:|---:|
| Casiav1 | 920 | 0.055019 | 6.759913413e-07 |
| Columbia | 180 | 0.089444 | 0 |
| NIST16 | 564 | 0.026798 | 0 |
| IMD2020 | 2010 | 0.054859 | 1.309587886e-06 |
| DSO-1 | 100 | 0.060703 | 0 |
| wild | 201 | 0.034582 | 0 |
| coverage | 100 | 0.077931 | 0 |
| Korus | 220 | 0.066458 | 0 |
| 八集等权平均 | 4295 | 0.058224 | 2.481974034e-07 |
| 4295 张逐图平均 | 4295 | 0.052976 | 7.576679126e-07 |

**GPU 1 本次定位结果几乎为零：4,295 张图中只有 2 张的 F1 大于 0。** 八集等权平均 F1 为 2.481974034e-7，逐图平均 F1 为 7.576679126e-7。三集 MAE 选出的 best 不代表 F1 最优。

已核对 4,295 个唯一测试样本、逐图 CSV 与全部汇总值、实际 best 权重 SHA-256、训练/评估冻结源码，并确认 checkpoint 浮点参数均有限。未发现样本遗漏、权重选错或汇总不一致；定位效果失效的具体原因尚未诊断。完整负结果保留，未据此重选权重。

训练轮数和选模数据集不同，该表展示两次实际运行的结果，不能把差异全部归因于训练数据集。

[GPU 1 完整报告](DCDSDIFF-CASIA2-SEL3-ALL8-20260914-B/all8-best/report.md) · [GPU 1 逐图指标](DCDSDIFF-CASIA2-SEL3-ALL8-20260914-B/all8-best/per_image.csv) · [GPU 0 完整报告](DCDSDIFF-GIT10K-RECON-20260914-A/all8-best/report.md) · [完成与核验收据](completion_receipt.json)。

两个实验的选择数据不同：GPU 0 使用 GIT10K 重建测试集；GPU 1 在续训后使用 Casiav1/Columbia/NIST16。GPU 1 的这三集属于参与 checkpoint 选择的测试数据；其父实验 epoch 0–2 曾监测八集，因此其他五集也不能描述为整个训练历史从未接触的测试集。提前结束由用户指定，未将它包装为预先定义的自动 early-stopping 规则。

## GPU 0 固定末轮复现结果

`GIT10K-PAPER-RECON-V1` 的预先指定主结果使用 epoch 99 的 `model-final.pt`，测试 GIT10K 重建划分的 1,000 张图：**F1 0.180584、IoU 0.105304、MAE 0.492114**。这与额外八库 best 评估是两个独立结果，不能互相替代。

[完整末轮报告](DCDSDIFF-GIT10K-RECON-20260914-A/final99/report.md) 保留实际文件前缀分组；这些前缀不是已确认的论文生成器类别。工程流程完成，但本次结果未达到论文报告水平；作者原划分、生成器映射和辅助图细节未完全公开，本项目采用了已登记的重建假设。当前结果不能单独确定分数差距的原因，也不代表完成所有论文实验。详见 [论文核对](../paper_protocol_audit.md) 和 [复现协议](../reproduction_protocol.md)。

## 复查入口

- [CASIA2 提前结束协议](../casia2_early_stop.md)、[停止收据](DCDSDIFF-CASIA2-SEL3-ALL8-20260914-B/early_stop.json)、[85 轮训练记录](DCDSDIFF-CASIA2-SEL3-ALL8-20260914-B/metrics.jsonl)。
- [GIT10K 100 轮训练记录](DCDSDIFF-GIT10K-RECON-20260914-A/metrics.jsonl)、[末轮逐图指标](DCDSDIFF-GIT10K-RECON-20260914-A/final99/per_image.csv)、[八库逐图指标](DCDSDIFF-GIT10K-RECON-20260914-A/all8-best/per_image.csv)。
- 每个结果目录保留机器可读 `results.json` 与逐图 CSV；run/follow-up 的 provenance 记录训练和评估提交、数据/权重哈希及冻结源码文件哈希。原始权重只保留在服务器对应 `runs/<RUN_ID>`。
- 原始 `training_status.json` 是最后一次训练进度快照。CASIA2 的终止状态以 `controller_status.json=INTERRUPTED`、`early_stop.json=EARLY_STOPPED_BY_USER` 为准；八库测试是否完成以独立 follow-up 的 `status.json` 为准。

F1/IoU/MAE 均先在原图尺寸逐图计算再取平均；推理 seed 0、10 步、352px 输入、batch 6，二值阈值 >0.5。宏平均给八个数据集相同权重，逐图平均按各集样本数加权；它不是先合并所有像素混淆矩阵再计算的 F1。
