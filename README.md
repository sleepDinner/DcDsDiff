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

控制器脱离 SSH 会话，持有项目文件锁；每 60 秒更新轻量状态，训练结束后自动运行固定最终评估。`status` 会检查 PID 身份并同时显示训练进度。失败会保存原因并停在原实验，不自动修改科学参数重试。`stop` 只终止本控制器持有的训练进程组；未完成 epoch 在恢复时重跑。

每个 run 保存 `provenance.json`、`source/`、`environment.freeze.txt`、`resolved_config.yaml`、`metrics.jsonl`、`training_status.json` 和 `controller_status.json`。训练每 10 epoch 保存归档，另保留 last、诊断 best 和最终 checkpoint。最终评估产生 `evaluation/{results.json,per_image.csv,report.md}`；仅 `controller_status.json=COMPLETED` 表示训练与固定评估均完成。最终结果需同步回本地后更新台账并提交；大型权重留在服务器。

## 后续创新实验

在 `codex/<idea>` 分支下新增 `config/experiments/<idea>.yaml`，继承基线，使用新的 `protocol_id`，记录唯一变量、假设和对照。模型变体用独立类/显式版本，禁止改写已冻结基线或运行中的 `runs/<RUN_ID>/source`。同一 100-epoch/10-step 协议的变体可用：

```bash
$PY tools/manage_experiment.py launch --run-id NEW_AUTHORIZED_RUN_ID --gpu 0 \
  --config config/experiments/your_idea.yaml
```

其他训练终点、数据集、种子或评估策略需要先登记新协议并适配控制器，不能套用本轮结果标签。重模型验证在服务器环境进行；临时测试脚本和测试 checkpoint 结束后移除，Git 只保留验证结论与小型收据。数据、权重、缓存和恢复备份均已加入 `.gitignore`。

作者原始数据入口：[夸克网盘](https://pan.quark.cn/s/0d6b5c9d7344)，提取码 `81hq`。
