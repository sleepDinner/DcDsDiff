**DcDsDiff 复现完整性审计 — 2026-09-14**

结论：已完成基于作者公开代码的主模型工程复现，存在有效的训练权重、推理图和评估结果；目前不能认定为严格的 GIT10K 主实验复现，也不能认定为完整论文实验复现。完成 100 个 epoch 的 FinalTrainData 实验使用了另一套数据，不能替代 GIT10K 主实验。

本次通过 SSH 只读检查服务器代码、数据目录、日志、checkpoint 元数据与 W&B 历史记录，并在 CPU 上重新计算现有 GIT10K 预测图的 F1/IoU。没有启动训练或模型推理，没有修改服务器文件。新增文件仅为本地本次审计材料。

**证据来源与版本**

- 服务器代码：`imdl-server:/data0/hl/DcDsDiff-and-GIT10K/`，HEAD 为 `d60e4e893ad391a699fe5229a4be0903cfc3cbf3`。当前受 Git 跟踪的服务器文件没有未提交修改。
- 服务器实验：`imdl-server:/data0/hl/DcDsDiff_runs/`。
- 作者公开代码：[QixianHao/DcDsDiff-and-GIT10K](https://github.com/QixianHao/DcDsDiff-and-GIT10K/tree/d2ad59fc218727c2148328e73c3d7a9fcb2b8bde)，本次固定检查提交 `d2ad59fc218727c2148328e73c3d7a9fcb2b8bde`。
- 论文：[IJCAI 2025 正式论文](https://www.ijcai.org/proceedings/2025/0120.pdf)，主要核对第 4.1 节及表 1—4、图 7。
- 本地 Python 文件的现有未提交修改经 AST 比较与 HEAD 相同，本次以服务器文件和历史运行记录为判断依据。

**三轮训练实际完成到哪里**

以下 epoch 均为代码中的零起始编号；0—99 才是 100 个完整 epoch。

| 运行目录（均在 DcDsDiff_runs 下） | 训练/测试图像数 | 已保存 epoch | model-best.pt 内部 epoch | 实际状态 |
|---|---:|---|---:|---|
| `main` | 8,996 / 1,004 | 0—88，共 89 个 | 21 | epoch 89 训练期间收到 SIGTERM；未完成 100 个 epoch |
| `finaltraindata` | 43,766 / 4,863 | 0—20，共 21 个 | 19 | epoch 21 训练已结束、验证期间收到 SIGHUP；未完整结束 |
| `finaltraindata_20260611_150448` | 43,766 / 4,863 | 0—99，共 100 个 | 44 | 训练和随后 Mix 评估均完成 |

`main` 的历史记录包含 89 次验证、67,023 条训练学习率记录；前 89 个 epoch 各 750 步，随后又执行了 epoch 89 的 273 步。完成轮包含 100 次验证、273,600 条训练学习率记录，恰好是 100 × 2,736 步。

完成轮日志记录：2026-06-13 17:26:58 `training complete`；17:46:04 `Finished`。其评估加载的是 epoch 44 的 `model-best.pt`，不是 epoch 99。`main/epoch21_best.pth` 的内部 epoch 也是 21；`main_f1_scan` 是对已有 0—88 checkpoint 的事后 F1 扫描，不能补足缺失训练。

训练状态证据：

- `/data0/hl/DcDsDiff-and-GIT10K/logs/train_20260609_055224.log`
- `/data0/hl/DcDsDiff_runs/finaltraindata/train.log`
- `/data0/hl/DcDsDiff_runs/finaltraindata_20260611_150448/train_then_eval.log`
- `/data0/hl/DcDsDiff_runs/finaltraindata_20260611_150448/run_train_then_eval.sh`

**数据没有漏掉，但划分和子集语义尚未对齐论文**

论文主实验规定 9,000 张训练图，BN、PE、IA、PP 各 250 张测试图。服务器原始 `GIT10K/Image` 和 `GIT10K/Mask` 都有 10,000 张；处理后的训练与 Mix 测试图共 10,000 张，文件名覆盖原始全集且交集为零。两套数据的 f/m/d/t 文件名也均完整对齐。这些检查支持数据处理完整，但只是文件名级检查，不等于验证了图像内容重复、原始来源独立性或官方划分一致性。

当前 GIT10K 划分与 `tools/generate_git10k_aux.py` 的默认算法完全匹配：按九个文件名前缀分组、自然排序，每组前 90% 训练、末尾 10% 测试；每组取整后成为 8,996 / 1,004。不是论文所述的四组各 2,250 / 250。

| 报告中的子集名 | 实际合并的文件名前缀 | 测试图像数 |
|---|---|---:|
| BN | BN、RBN | 259 |
| PE | EI、Flux、e、t、z | 269 |
| IA | IA | 248 |
| PP | PP | 228 |

该合并规则由本地新增评估脚本 `tools/evaluate_checkpoint_f1_iou.py:40` 定义。作者仓库和服务器原始数据目录中未找到确认这些前缀对应关系及官方 train/test 划分的说明。因此不能仅靠输出行名称将这些子集视为论文同一测试集；尤其 PE 行包含多个未经作者映射证实的来源。

FinalTrainData 轮加载 `/data0/hl/FinalTrainData_Diff/`，清单来源为 `/data0/hl/FinalTrainData/images` 和 `masks`；训练前缀为 Fan 32,399、Au 6,731、Tp 4,636。其 Mix F1=0.9161457769、IoU=0.9034895534 是这套 4,863 张测试图的结果，不能用作 GIT10K 复现成绩。

**训练设置和调度存在实际偏离**

论文第 4.1 节设置为 352×352、AdamW、初始学习率 0.001、batch 6、100 epochs、随机水平翻转、推理 10 步。服务器符合分辨率、优化器种类和推理步数，但存在以下差异：

| 项目 | 服务器实际证据 | 判断 |
|---|---|---|
| 初始学习率 | 两轮历史学习率均从 0.0001 开始 | 比论文低一个数量级；作者公开 YAML 本身也是 0.0001 |
| 有效 batch | main 每卡 6、两进程，实际 750 步/epoch；完成轮每卡 8、两进程，2,736 步/epoch | 对应有效 batch 12 和 16，不是论文单卡 batch 6 |
| 水平翻转 | `dataset/data_val.py` 定义了增强，但 `__getitem__` 中调用被注释 | 当前训练路径没有启用论文描述的随机翻转；作者代码同样如此 |
| 余弦调度 | 两轮在 epoch 50 达到 1e-6，随后回升 | 双进程下每 epoch 推进两次，与代码 `T_max=100` 的单进程执行轨迹不同 |

最后一项已由原始 W&B 记录直接确认，而非仅由代码推断：

| epoch 开始时 | main 实际 lr | 完成的 FinalTrainData 轮实际 lr |
|---:|---:|---:|
| 0 | 0.0001000000 | 0.0001000000 |
| 25 | 0.0000505000 | 0.0000505000 |
| 50 | 0.0000010000 | 0.0000010000 |
| 75 | 0.0000505000 | 0.0000505000 |
| 88 | 0.0000865839 | 0.0000865839 |
| 99 | 未完成到此处 | 0.0000999023 |

服务器当前 Accelerate 为 1.13.0。项目传递 `Accelerator(split_batches=True)`，但该版本实际使用 `DataLoaderConfiguration` 中默认的 `split_batches=False`；包装后的 scheduler 按进程数推进。历史每 epoch 步数及 lr 曲线与该行为一致。这里不声称论文要求某条特定余弦曲线；确认的是这次双卡执行偏离了公开训练代码按单卡运行的批量和调度行为。

W&B 证据：

- `/data0/hl/DcDsDiff-and-GIT10K/wandb/offline-run-20260609_055231-5a4ul1q8/run-5a4ul1q8.wandb`
- `/data0/hl/DcDsDiff-and-GIT10K/wandb/offline-run-20260611_150517-n2g75jxo/run-n2g75jxo.wandb`

**模型主体已实现，但仍有影响严格复现的实现问题**

服务器模型与作者模型对比，主干差异集中在两处必要修复：`MSIE()` 改为传入正确通道数，及不存在的 `backbone_n` 改为 `backbone_t`。扩散、损失、前向包装和 Dataset 的 AST 与作者版本相同。七个抽查 checkpoint 均可在 CPU 上以 `weights_only=True` 读取，包含 962 个 state_dict 条目；双 PVT 主干、MSIE 和融合层权重均存在。实际前向路径确实经过双主干、多模态融合、双分支解码及 MSIE；训练调用名为 `structure_loss` 的 mask 损失与 detail 的 0.5×(L1+MSE)。这支持主模型已经落地运行，但不意味着损失实现与论文完全一致，见下述补充验证。

2026-09-14 针对“模型是否与论文一样”的补充核对发现两处明确的细节差异，均继承自作者公开代码：

- **WBCE 实际退化为普通 mean BCE。** `model/loss.py:109` 使用 `reduce='none'`，而非 `reduction='none'`。在服务器 PyTorch 2.1.2 的 CPU 验证中，前者返回零维全局均值，后续乘权重再除以权重和基本抵消，因此没有实现预期的逐像素加权 BCE。一个两样本验证中，实际两个样本 BCE 均约 0.374016，正确加权值分别约 0.438214 和 0.728477。WIoU 项与 detail 辅助项仍存在。这个测试确认 API 行为和损失差异，没有测量修正后的模型精度。
- **空间注意力的算子不同。** 论文第 3.4 节写的是沿通道最大池化、3×3 卷积、Sigmoid；当前 `model/net.py:884` 使用平均池化与最大池化拼接、7×7 卷积、Sigmoid，且实际由 `MMFF_att` 调用。故核心框架对应论文，但无法称为逐层、逐算子完全一致。

因此，对模型本身的准确判断是：作者公开实现的完整 DcDsDiff 主体加必要运行修复，而不是已证明与论文描述逐项一致的训练模型。`FinalTrainData_352x352.yaml` 继承同一模型配置，只替换项目名与数据路径，没有定义另一种主网络。

仍应先处理或明确记录以下问题：

1. **trace 的训练/推理归一化不同。** 训练 Dataset 仅把 trace 转成 [0,1]，扩散 `forward` 再映射到 [-1,1]；测试 Dataset 对 trace 使用 ImageNet mean/std，`sample` 不做对应转换。证据：`dataset/data_val.py`、`model/SimpleDiffSef.py:219`、`:272`。此问题继承自作者代码；本次未通过重训量化其性能影响。
2. **双卡验证没有聚合各 rank 的 MAE。** `utils/trainer.py:255` 只有同步等待和本 rank 求均值，没有 gather/reduce；保存 best 的主进程据自身分片 MAE 选权重。因此训练记录中的 best MAE 不是完整 Mix 的全局 MAE。后续单卡保存预测并评估的 F1/IoU 仍是完整结果，二者需区分。
3. **在测试数据上选择 checkpoint。** 训练中每 epoch 使用 `cfg.test_dataset.Mix` 选择最低 MAE，另有事后 F1 扫描。没有独立验证集的证据，故这些应标注为 test-selected 结果，不能当作从未参与选择的独立测试成绩。论文没有明确公布足以还原 checkpoint 选择的完整细则。
4. **辅助图生成包含工程选择。** 新增脚本采用半径 15 的线性距离衰减 detail 图、圆形高通滤波、IFFT 绝对值及裁剪。已实现相应辅助信号，但没有作者原始辅助图生成脚本或输出对照来证明逐像素一致。

**现有 GIT10K 数值是真实可核验的，但不支持已经对齐论文结果**

| 子集 | 论文完整模型 F1 / IoU（表 1 Scheme4） | 服务器 main 的 F1 / IoU |
|---|---|---|
| BN | 0.866 / 0.782 | 0.770250 / 0.672524 |
| PE | 0.854 / 0.803 | 0.964976 / 0.940080 |
| IA | 0.818 / 0.738 | 0.600586 / 0.501798 |
| PP | 0.919 / 0.882 | 0.976550 / 0.956903 |

服务器报告的四行宏平均是 F1=0.8280903586、IoU=0.7678263356。由于测试划分和来源映射不同，上表仅用于显示现状，不能作严格同协议的性能比较，也不能把 PE/PP 更高认定为超过论文。

证据：`/data0/hl/DcDsDiff-and-GIT10K/eval_best/evaluation_results.json`，checkpoint 为 `/data0/hl/DcDsDiff_runs/main/model-best.pt`（内部 epoch 21）。本次遍历全部九个来源的 1,004 张已保存预测 PNG，用阈值 0.5 重算每图 F1/IoU，九个来源的均值均与该 JSON 完全一致；GT 均非空。未重新运行模型，未重算 AUC。

**论文其他实验尚未形成完整复现证据**

在用户指定的两个目录中，没有发现与论文消融、四种生成器分别训练后的交叉测试、各扩展数据集分别训练/测试、Gaussian noise/gamma 扰动评估对应的完整运行和结果矩阵。

已有 `main_external_eval`、`finaltraindata_external_eval`、`finaltraindata0615_test` 是现成 checkpoint 在 CASIA v1、Columbia、NIST16、IMD2020、DSO-1、Korus 上的迁移评估。它们属于额外实验，不能代替论文第 4.5 节对每个数据集分别训练再测试的协议；也不能直接与论文表 4 比较。

此“未发现”结论仅覆盖本次检查范围，不断言其他服务器目录中不存在相关实验。

**建议使用的结论表述与后续顺序**

可以写：“基于 DcDsDiff 作者公开代码完成主模型部署、训练和评估；在自行重建的 GIT10K 划分及 FinalTrainData 数据上获得结果。当前属于工程复现，尚未完成与论文数据协议和训练设置一致的严格复现。”

如需达到严格主实验复现，应先确认作者数据前缀映射、官方划分和 checkpoint 选择口径，再明确采用论文配置还是公开代码配置，解决批量/调度、增强、trace 归一化和分布式验证问题，冻结可追溯版本后完整训练与评估。原始 checkpoint 没有保存 optimizer/scheduler 状态，因此简单使用现有 `--resume` 不能保证是数学等价的无缝续训。若目标还包括整篇论文，再补足相应消融、交叉生成器、扩展和鲁棒性实验。

本次没有执行上述修改或新增实验；现有结果应保留并按其真实协议标注。

完整机器可读采集结果见同目录 `server_evidence.json`；作者代码与服务器 HEAD 的差异见 `upstream_comparison.diff`；附加核验摘要见 `verification_summary.json`。
