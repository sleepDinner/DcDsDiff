# DcDsDiff / GIT10K

基于 [IJCAI 2025 论文](https://www.ijcai.org/proceedings/2025/0120.pdf) 与 [作者源码](https://github.com/QixianHao/DcDsDiff-and-GIT10K/tree/d2ad59fc218727c2148328e73c3d7a9fcb2b8bde) 的可追溯复现与创新实验项目。

本轮基线是 **GIT10K-PAPER-RECON-V1**：按论文明确参数修正实现，使用明确记录的 9,000/1,000 自建划分。作者未提供完整划分及辅助图生成细节，因此不能称为官方同划分精确复现。主成绩固定使用 epoch 99，不进行事后 checkpoint/阈值搜索。训练运行中不代表论文成绩已复现。

先读 [复现协议](docs/reproduction_protocol.md)、[论文核对](docs/paper_protocol_audit.md)、[代码审查](docs/code_audit.md)、[实验台账](experiment_ledger.md)。旧项目审查保存在 [历史记录](docs/history/pre-rebuild-audit.md)，清理记录见 [cleanup](docs/cleanup.md)。

## 服务器布局

| 内容 | 路径 |
|---|---|
| 新代码仓库 | `/data1/hl/DcDsDiff-and-GIT10K` |
| 专用 Conda 环境 | `/data0/hl/conda_envs/dcdsdiff` |
| 只读原始数据 | `/data0/hl/DcDsDiff-and-GIT10K/GIT10K` |
| 处理数据与完整清单 | `data/git10k-recon-v1` |
| ImageNet PVT 权重 | `pretrained_weights/pvt_v2_b2.pth` |
| 实验代码快照、checkpoint、指标 | `runs/<RUN_ID>` |
| 安装日志、缓存、项目锁 | `runtime` |

旧服务器目录仅作为资源来源。原图/mask 用只读链接复用，辅助图在新目录生成，预训练 PVT 文件复制后校验 SHA-256。历史训练权重不用于新基线初始化。

## 环境和数据准备

在服务器新仓库目录执行，始终使用专用环境：

```bash
bash environment/create.sh
/data0/hl/conda_envs/dcdsdiff/bin/python tools/prepare_reproduction_data.py \
  --source /data0/hl/DcDsDiff-and-GIT10K/GIT10K \
  --output data/git10k-recon-v1 --seed 42 --workers 8
```

环境安装不会覆盖已有前缀；完成标志、版本锁与 `pip check` 位于 `runtime/bootstrap`。正式入口强制 `python -s` 排除服务器用户级包。数据准备不会覆盖已有输出；像素完全相同的原图分到同一侧，并保持 9,000/1,000。完整图像内容、mask、辅助图哈希及尺寸异常写入 manifest/receipt，检查通过才允许正式启动。

## 启动、查看、恢复

提交并推送代码，服务器 `git pull --ff-only` 后，在新仓库目录执行：

```bash
PY=/data0/hl/conda_envs/dcdsdiff/bin/python
$PY tools/manage_experiment.py launch --run-id YOUR_RUN_ID --gpu 0
$PY tools/manage_experiment.py status --run-id YOUR_RUN_ID
# 只在需要暂停本项目当前实验时使用：
$PY tools/manage_experiment.py stop --run-id YOUR_RUN_ID
# 从最后完整 epoch 恢复同一份源码/配置/优化器/随机状态：
$PY tools/manage_experiment.py resume --run-id YOUR_RUN_ID --gpu 0
```

控制器脱离 SSH 会话，持有 GPU 与 run 文件锁；每 60 秒更新轻量状态，按注册协议自动执行训练后评估。新控制器识别原 GIT10K 控制器持有的旧锁及其 GPU，允许另一张空闲卡并行训练。`status` 会检查 PID 身份并同时显示训练进度。失败会保存原因并停在原实验，不自动修改科学参数重试。`stop` 只终止本控制器持有的训练进程组；恢复使用原注册 GPU，未完成 epoch 在恢复时重跑。

每个 run 保存 `provenance.json`、`source/`、`environment.freeze.txt`、`resolved_config.yaml`、`metrics.jsonl`、`training_status.json` 和 `controller_status.json`。训练每 10 epoch 保存归档，另保留 last、诊断 best 和最终 checkpoint。最终评估产生 `evaluation/{results.json,per_image.csv,report.md}`；仅 `controller_status.json=COMPLETED` 表示训练与固定评估均完成。最终结果需同步回本地后更新台账并提交；大型权重留在服务器。

## 后续创新实验

在 `codex/<idea>` 分支下新增 `config/experiments/<idea>.yaml`，继承基线，使用新的 `protocol_id`，记录唯一变量、假设和对照。模型变体用独立类/显式版本，禁止改写已冻结基线或运行中的 `runs/<RUN_ID>/source`。同一 100-epoch/10-step 协议的变体可用：

```bash
$PY tools/manage_experiment.py launch --run-id NEW_AUTHORIZED_RUN_ID --gpu 0 \
  --config config/experiments/your_idea.yaml
```

其他训练终点、数据集、种子或评估策略需要先登记新协议并适配控制器，不能套用本轮结果标签。重模型验证在服务器环境进行；临时测试脚本和测试 checkpoint 结束后移除，Git 只保留验证结论与小型收据。数据、权重、缓存和恢复备份均已加入 `.gitignore`。

## CASIA2 与八数据集扩展

**2026-09-15 最新变更：GPU 1 已按用户要求提前结束，完整完成 epoch 0–84（85 轮）；使用已有 epoch-52 best 进行八集评估，不再恢复训练。详见 [提前结束与八集测试](docs/casia2_early_stop.md)。以下 100-epoch 续训描述保留为历史。GPU 0 的固定末轮和八集 best 评估均已完成。

**当前 GPU 1 协议已修改**：每轮仅测试 Casiav1、Columbia、NIST16（1,664 张），仍以逐图平均 MAE 选择 best；训练结束后对全部八集（4,295 张）报告 F1。旧 A 已停止，使用最后完整 epoch-2 状态在 `DCDSDIFF-CASIA2-SEL3-ALL8-20260914-B` 继续至 epoch 99。旧 best 单独保留，新的 best 从变更后的轮次开始选择。启动/恢复说明见 [三集选择与检查点续训](docs/casia2_sel3_continuation.md)。以下 All8-per-epoch 配置与 A 命令保留为历史协议，不要重新启动或恢复 A。

2026-09-14 用户授权的新实验见 [CASIA2/All8 协议](docs/casia2_all8_protocol.md)。训练使用 CASIA2 的 5,123 对 Tp/Gt；测试使用指定八集，共 4,295 张。模型和训练参数继承原基线。数据路径、配对后缀和数量固定在 `config/benchmark_all8.json`，处理数据为 `data/casia2-all8-v1`，完整清单提交到 `manifests/casia2-all8-v1.csv`。

在尚未建立此数据快照的新服务器 checkout 中，先执行以下准备与清单比对；已有快照直接复用，启动器会核验其文件哈希：

```bash
$PY tools/prepare_benchmark_data.py --workers 8
cmp data/casia2-all8-v1/manifest.csv manifests/casia2-all8-v1.csv
```

```bash
$PY tools/manage_experiment.py launch --run-id DCDSDIFF-CASIA2-ALL8-20260914-A \
  --gpu 1 --config config/experiments/casia2_all8.yaml
$PY tools/manage_experiment.py status --run-id DCDSDIFF-CASIA2-ALL8-20260914-A

# 只登记一次原实验结束后的附加评估，不会启动另一轮训练：
$PY tools/queue_benchmark.py queue --run-id DCDSDIFF-GIT10K-RECON-20260914-A
$PY tools/queue_benchmark.py status --run-id DCDSDIFF-GIT10K-RECON-20260914-A
```

历史 CASIA2 A 的 best 由 All8 MAE 选择；当前续训 B 的 best 由三集 MAE 选择，完成总计 100 epochs 后直接使用已有 `model-best.pt` 输出八集结果。原 GIT10K 的 best 仍由原 GIT10K Mix MAE 选择，附加评估不使用 All8 重新挑权重，也不更改原 final99 复现终点。两个评估都直接读取原文件，不创建别名。

CASIA2 结果在 `runs/<CASIA_RUN_ID>/evaluation`；原实验的附加结果在 `runs/<GIT_RUN_ID>/followups/all8-best/evaluation`。后者有独立 `status.json`、控制器身份及评估代码快照，会等待原控制器完成退出后使用 GPU 0。`queue_benchmark.py resume/stop` 只恢复或停止该附加评估控制器。八集报告保留每集 F1/IoU/MAE、等数据集宏均值及逐图加权均值，并记录实际 checkpoint 哈希、epoch 和选择集。

作者原始数据入口：[夸克网盘](https://pan.quark.cn/s/0d6b5c9d7344)，提取码 `81hq`。
