# DcDsDiff 论文复现协议审查

审查日期：2026-09-14。当前重建基线标识：`paper-aligned-v1`。

本文件将论文明确要求、官方实现细节和本项目补全假设分开记录。当前工作可以形成可重复执行的基线；由于官方未公开完整数据划分、辅助图构建与 F1/IoU 评估细节，不能把一次正常训练或接近表格的分数表述成严格数值复现成功。

## 1. 证据与版本

- 论文：Hao et al., *DcDsDiff: Dual-Conditional and Dual-Stream Diffusion Model for Generative Image Tampering Localization*, IJCAI 2025, pp. 1071-1079；[正式出版页面](https://www.ijcai.org/proceedings/2025/120)，DOI `10.24963/ijcai.2025/120`。
- 论文主要证据为用户提供的本地 `DcDsDiff 论文.pdf`，共 9 页，SHA256 `bec71779e774eeb80f879e6ba03092849b1e991b323e81edac7b5818ad5e1c65`。使用文本抽取核对正文，并目视检查 PDF 第 5、6 页的设置、损失和结果表格。PDF 页码 1 对应出版页码 1071。
- [官方仓库](https://github.com/QixianHao/DcDsDiff-and-GIT10K)，审查时 HEAD：`d2ad59fc218727c2148328e73c3d7a9fcb2b8bde`。通过独立只读克隆审查，未修改当前工作区 Git refs。
- 下文“官方代码”均指上述固定提交，不能用当前项目已修改文件代替官方证据。

## 2. 首轮主实验的论文依据

首轮基线采用 Table 1 Scheme 4 / Table 2 的完整 DcDsDiff 模型方案。论文使用四种生成器混合训练并分别报告四个测试子集；本次发布数据无法核验全部文件前缀到这四种生成器的映射，因此采用 seed 42、相同解码 RGB 内容绑定同一分区的 9,000/1,000 自建划分，只报告真实文件前缀及全局结果，不能伪造 BN/PE/IA/PP 四组对应关系。其他消融、单生成器泛化、扩展数据集与鲁棒性实验是不同实验协议，不能由这一次主模型训练代替。实际冻结规则见 [reproduction_protocol.md](reproduction_protocol.md)。

| 项目 | 论文明确说明 | 官方实现或缺口 | 本轮约定与依据类型 |
| --- | --- | --- | --- |
| 训练样本 | GIT10K 总计 10,000；四种生成器各 2,500；9:1 拆分；混合训练 9,000 | 未提供本文可核验的作者 split manifest，也不能验证发布文件的完整生成器映射 | natural-sort stems 后按解码 RGB SHA256 分组，Python Random(42) 打乱内容组，整组填满训练 9,000、余下测试 1,000，保留全部样本且相同内容不跨分区；属于已声明补全假设 |
| 测试样本 | BN/PE/IA/PP 各 250，合计 1,000 | 官方 `train.py` 会重复抽取部分测试样本追加到 Mix | 本轮全局 1,000，每样本只计一次；不宣称重建四组各 250 |
| 输入尺寸 | 352×352，§4.1，PDF p5 / 出版 p1075 | 另有 384 配置，但不是正文主设置 | 固定 352×352，论文依据 |
| 优化器与初始 LR | AdamW，0.001，§4.1 | 官方 352 YAML 为 `1e-4` | 采用 `0.001`，覆盖明确冲突 |
| Batch / epochs | 6 / 100，§4.1 | 官方 CLI 默认一致 | 固定 batch 6、100 epochs |
| 数据增强 | 随机水平翻转，§4.1 | 定义了多种增强，但实际调用均被注释，训练路径没有翻转 | 四路 image/mask/detail/trace 同步水平翻转，概率 0.5；概率为补全细节 |
| 训练种子 | 未说明 | 官方 `train.py` 设置 42 | 42 来自代码；不称论文种子 |
| LR 日程 | 未说明 | CosineAnnealingLR，100 epochs，最小 LR 默认 `1e-6` | 若沿用，注明官方实现补全 |
| 权重衰减 / Adam betas | 未说明 | AdamW 未显式配置，依赖 PyTorch 默认值 | 在有效配置中固化实际值；属于官方默认参数补全 |
| 主干 | 两个预训练 PVTv2，§3.2 | 实例化两个 PVTv2-b2，时间嵌入为额外新参数 | b2 规格源自代码；两个特征主干加载同一来源权重，但参数独立训练 |
| HFVG | 分通道 FFT→高通→IFFT→×10；阈值 0.5，§3.3 Eq.9 与 §4.1 | 未提供可重建全部操作的高通/保存脚本 | 阈值数值与增强倍数来自论文；滤波形状和量化等是补全假设 |
| Detail GT | 边缘及邻域，响应随距最近边缘距离增加而降低；§3.5 | 官方读取预计算 `de_root`，没有生成代码 | 重建算法及半径必须独立登记；不能称作者原始 detail GT |
| Mask 损失 | WBCE+WIoU，§3.6 Eq.18 | `structure_loss` 的 WBCE reduction 参数拼写错误 | 修复逐像素 WBCE，再按 31×31 边界权重汇总 |
| Detail 损失 | 0.5×(L1+L2)，§3.6 Eq.18 | 官方以 MSE 实现 L2 | 沿用 0.5×(L1+MSE) |
| 扩散日程 | SNR-based，§4.1 | cosine log-SNR，`noise_d=64` | 具体数值来自官方 YAML/扩散库 |
| 采样 | 10 步，非线性选 t，§4.1 | `sin(linspace(1,0,11)×π/2)` | 固定 10 步，具体函数沿用官方 |
| 时间集成 | 对迭代中的多个预测求平均，§3.2 | 另有原尺寸双线性插值、单图 min-max 与符号多数投票门控 | 保留的后处理应明确记为官方代码细节，不能省略后仍声称同协议 |
| F1 / IoU | 指标名称，§4.1 | 官方 `utils/eval.py` 返回 Smeasure、wFmeasure、MAE、Em，没有论文 F1/IoU 入口 | 本轮原尺寸、阈值 >0.5、empty/empty=1、逐图计算再均值、评估 seed 0，均为冻结补全 |
| Checkpoint | 未说明选择策略 | 官方每 epoch 对测试 Mix 做时间集成 MAE，并选 best | 本轮固定末 epoch 99 为主结果；best-on-test MAE 仅为诊断记录 |

对应的固定源码证据：[352 配置](https://github.com/QixianHao/DcDsDiff-and-GIT10K/blob/d2ad59fc218727c2148328e73c3d7a9fcb2b8bde/config/DcDsDiff_352x352.yaml)、[训练入口](https://github.com/QixianHao/DcDsDiff-and-GIT10K/blob/d2ad59fc218727c2148328e73c3d7a9fcb2b8bde/train.py)、[数据集](https://github.com/QixianHao/DcDsDiff-and-GIT10K/blob/d2ad59fc218727c2148328e73c3d7a9fcb2b8bde/dataset/data_val.py)、[扩散包装器](https://github.com/QixianHao/DcDsDiff-and-GIT10K/blob/d2ad59fc218727c2148328e73c3d7a9fcb2b8bde/model/SimpleDiffSef.py)、[时间集成](https://github.com/QixianHao/DcDsDiff-and-GIT10K/blob/d2ad59fc218727c2148328e73c3d7a9fcb2b8bde/model/train_val_forward.py)、[官方评估器](https://github.com/QixianHao/DcDsDiff-and-GIT10K/blob/d2ad59fc218727c2148328e73c3d7a9fcb2b8bde/utils/eval.py)。

## 3. 架构冲突和 `paper-aligned-v1`

论文 §3.4 / PDF p4 / 出版 p1074 的 Eq.10 下方明确把 Spatial Attention 写为沿通道的 global max pooling、3×3 convolution、sigmoid。官方 `model/net.py:884` 则是通道平均值与最大值拼接、双通道 7×7 convolution、sigmoid。

同节 Eq.11 下方把 Channel Attention 写为 global max pooling、1×1 convolution、ReLU、sigmoid。官方 `model/net.py:865` 使用平均池化与最大池化两路、共享两层瓶颈 1×1 MLP，再相加后 sigmoid。

本轮 SA 修改为 `max_channel→Conv(1,1,3,padding=1)→sigmoid`；CA 修改为 `AdaptiveMaxPool(1)→Conv(C,C,1)→ReLU→sigmoid`。CA 的 C→C 是根据输出需要逐通道重标定、且正文仅写一层卷积作出的维度补全。卷积 bias=False 沿用原实现约定。两项修改会改变可训练参数形状，因此 `paper-aligned-v1` 是一个明确的新架构版本，不能静默加载旧 DcDsDiff 训练 checkpoint。预训练 PVT 特征权重仍可复用。

`paper-aligned-v1` 表示修复已识别的明确冲突并冻结必要补全，不表示论文已经给出了每一行实现。例如 PVT 时间 token、MMFF 中间通道缩减、MSFF 拼接后的 1×1 投影、MSIE 的 gamma 参数化及归一化层仍来自官方代码。本文没有对这些未完全指定的细节重新设计一个新模型。

## 4. 辅助数据与评估的不可消除缺口

1. **作者原划分与生成器映射未知。** 论文 9:1、每组 2,250/250 是明确报告的数量，发布文件的真实组别和原始 ID 列表无法恢复。本轮先 natural-sort stems，再按相同解码 RGB SHA256 绑定内容组，用 Random(42) 打乱组序，整组填满训练 9,000，余下测试 1,000。保留全部 10,000 样本、不拆开同内容组，并保留不可变 manifest、源摘要、实际前缀数量和交集审计。不能从 RBN/EI/Flux/e/t/z 等前缀自行推断论文的四组映射，结果标题须注明内容组绑定的重建划分。
2. **HFVG 操作不完整。** 阈值 0.5 未定义为圆形/方形滤波、半径占宽/高/对角线比例。论文 Eq.9 未说明 IFFT 后取 real/abs、负值处理、截断和 PNG 量化。因此现有圆形高通、半径 `0.5×min(H,W)/2`、`abs(real(IFFT))×10`、截断并 uint8 量化，是可复现的重建假设。
3. **Detail 公式不完整。** 当前重建采用 3×3 椭圆核膨胀减腐蚀取边界，OpenCV L2 距离变换、半径 15 像素线性衰减。论文只支持“距边缘越远响应越低”的设计意图，并未公开这组公式或参数。
4. **尺寸和灰度细节。** 原始 image/mask 个别尺寸不同，需要在数据登记中计数。本轮各路独立 bilinear resize 到 352 对齐；训练保留灰度 soft mask。正式评估的原尺寸和 GT 二值阈值应写清，例如原始 8-bit mask `>127.5`。不能把工程补全描述成论文原操作。
5. **训练/推理 trace 分布冲突。** 官方训练 Dataset 产生 [0,1] trace，再由 diffusion.forward 映射到 [-1,1]；官方测试却走 ImageNet normalization。修复测试为 Resize+ToTensor+Normalize(0.5,0.5)，得到与训练等价的 [-1,1]。RGB 条件仍采用原 ImageNet normalization。
6. **F1/IoU 口径未知。** 单图指标平均和整数据集混淆矩阵池化不等价；不能根据表格倒推出唯一口径，也不能用 wFmeasure 代替 F1。补全口径必须在观察正式结果前固定，同时报告全局、实际文件前缀分组及有效样本数；不能把未验证的前缀分组直接与 Table 1/2 列名对齐。
7. **随机采样与最终 epoch。** 扩散推理包含随机噪声。固定种子与 batch/样本顺序是复核结果的必要条件。若最终评估与训练中的验证采用不同 batch 大小，随机数分配可以变化；必须记录为运行条件，不能无说明地混用多个结果。

上述缺口是严格数值复现的限制，不是阻止运行已声明重建基线的理由。本轮已按用户授权继续自建划分与重建辅助图，并保留这些边界。

首次正式训练前的数据审计拒绝了按单文件全局 shuffle 的候选划分：发布数据存在相同解码 RGB 图像，不同文件名仍可能造成 train/test 内容泄漏。因此在任何正式训练和分数观察之前，修正为上述内容组绑定划分。这次修正由输入内容泄漏检查触发，不是根据测试成绩选择数据集；具体重复组数、各类标注统计和最终无交集结果以正式数据准备 receipt 为准。该检查只排除完全相同的解码 RGB 内容，不能证明没有同源图像、语义相关样本或变换后的近重复。

## 5. 论文目标值

下列数值是论文值，未填入任何本地训练结果。Table 1 Scheme 4 提供三位小数；Table 2 完整模型为较粗的两位小数展示。

| 子集 | 测试数 | Table 1 F1 | Table 1 IoU | Table 2 F1 | Table 2 IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| BN | 250 | 0.866 | 0.782 | 0.87 | 0.78 |
| PE | 250 | 0.854 | 0.803 | 0.85 | 0.80 |
| IA | 250 | 0.818 | 0.738 | 0.82 | 0.74 |
| PP | 250 | 0.919 | 0.882 | 0.92 | 0.88 |

Table 1 四组算术均值为 F1 `0.86425`、IoU `0.80125`，是本审查计算的汇总，**不是论文额外报告的 Mix 指标**。

论文还包含以下独立实验，不能从主模型结果宣称已复现：

| 实验 | 论文范围 | DcDsDiff 目标 |
| --- | --- | --- |
| Table 1 消融 | Base、+DSDN、+HFVG、+MM-MSFF | 4 个不同模型配置，首轮只运行完整模型 |
| Table 3 泛化 | 单生成器训练，另外三组测试 | BN→PE/IA/PP: .646/.575/.745；PE→BN/IA/PP: .439/.341/.798；IA→BN/PE/PP: .566/.389/.476；PP→BN/PE/IA: .156/.385/.053，均为 F1 |
| Table 4 扩展性 | RLS/IMD/Nist16/DEF/AUTO 各自训练测试，非跨库混合 | F1 .583/.495/.936/.869/.949 |
| Fig.7 鲁棒性 | GIT10K 混合训练，测试加高斯噪声/gamma correction | 图形结果；本文不从曲线估出伪精确数值 |

## 6. 本轮修复及验证边界

- `dataset/data_val.py`：按 exact stem 四路严格配对；重复 stem、缺失、额外样本均 fail closed；加入同步水平翻转；修复 trace 范围；清理没有执行过的增强代码；保留可调用的数据集接口和原尺寸评估 GT。
- `model/loss.py`：`reduce='none'` 改为 `reduction='none'`。通过小型 CPU 复现实验证明，旧写法返回标量，使逐像素权重失效；修复结果与独立 WBCE+WIoU 公式一致，梯度有限。
- `model/net.py`：SA/CA 论文对齐；固定版本值；预训练默认只从 `pretrained_path`、`DCDSDIFF_PRETRAINED` 或项目 `pretrained_weights/pvt_v2_b2.pth` 读取。自动网络下载默认关闭；只有显式启用下载且未给本地路径时才允许 HF 下载。
- `model/train_val_forward.py`：只保留实际调用的双流入口；按交替 history 分离 mask/detail，并验证步数和形状。mask 的时间均值、原尺寸插值、min-max 与正值多数门控数学保持原样。detail 删除把均值乘以所有采样步而广播成 T 通道的错误，返回一张均值 detail 图；该 detail 返回值不参与当前 mask F1/IoU 指标。小型 CPU 检查确认 10 步 mask 与原公式逐值相等、detail 为单通道，并验证非 10 步接口的长度处理。
- 两个 PVT 分支必须完整覆盖所有特征主干 keys，逐 key 验证 tensor 形状；只允许新初始化的时间嵌入/未使用 mask projection 缺失，允许忽略原分类 head；来源路径、SHA256、覆盖率写入 `pretrained_load_report`。这是预训练覆盖检查，不是论文效果验证。
- 本地 CPU 不变量检查通过：损失与梯度、SA/CA 计算与梯度、四路同步翻转、训练/测试 trace 等价、额外 mask 拒绝。测试脚本只作为临时材料，正式代码不保留测试入口。
- 新服务器环境的完整模型 GPU 验证已通过，见 [validation_gpu.json](validation_gpu.json)：禁用 user-site 的 `dcdsdiff` 环境、Torch 2.1.2+cu121、OpenCV 4.9.0；两个 PVT 特征分支均 332/332 keys 完整加载；真实 6 图、352×352、FP32 一步 forward/backward 与 AdamW 更新完成，双 PVT、mask/detail 头及 SA/CA 代表参数均更新，全部实际梯度有限。forward/backward 约 1.04 秒，峰值 allocated 约 10.40 GiB。
- GPU smoke 的 8 张原始样本及 d/t 均在新锁环境现场独立准备，属于临时 6 训练/2 测试集，不代表正式 9,000/1,000 数据准备已通过。一步后的临时模型通过真正 `tools/evaluate_reproduction.py` CLI 完成两图 10 步评估，验证输出尺寸/有限值、样本覆盖和 checkpoint hash。临时 checkpoint 的 epoch 0/next_epoch 1 是接口测试元数据，实际只更新一次、没有完成一个正式 epoch，不能冒充 epoch 99 科研结果。测试进程退出后确认 GPU 上无其残留进程。

## 7. 长期基线约束

每次创新实验应绑定：架构版本、Git commit、数据 split/辅助图生成 manifest、环境导出、PVT checkpoint SHA256、训练与推理配置、RUN_ID、选择规则、最终评估 receipt。改变任意科学口径都应产生新配置/新实验标识，不能重写 `paper-aligned-v1` 的历史含义。旧权重可以作为已登记的迁移实验来源，但不能默认为本轮 fresh reproduction 的初始化。

正式结论应分别报告工程状态与科研状态：环境和训练正常完成、论文设置对齐程度、与论文表格的差值、数据/辅助图/评估仍存在的补全假设。分数达到或未达到目标都保留完整结果；不根据最终测试分数追加 checkpoint、阈值或 seed 搜索。
