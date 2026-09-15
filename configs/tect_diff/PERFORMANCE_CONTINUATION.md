# REFNORM-V2 性能优化续训

2026-09-15 同一会话，用户追加要求检查是否因服务器资源利用不足导致训练慢，有则优化、尽可能利用服务器配置。本次只优化实测同步开销；[科学配置](full_r512_s42_refnorm_v2.json) 的全部字段保持相同。

## 运行继承

- B：`TECT-DIFF-FULL-R512-S42-REFNORM-V2-20260915-B`，初始执行提交 `100d46edb1f73dd59fa0d57277e07c2925c53278`。已完成参考 epoch0–1，共1496次更新；两轮健康检查通过。B为有界性能对照暂停，原权重与源码保留。
- C：`TECT-DIFF-FULL-R512-S42-REFNORM-V2-PERF-20260915-C`，从 B 的 `reference_last.pth` 完整 epoch1 接续 epoch2–19，再执行校准及全部主训练 epoch0–99。
- C 的新源码快照来自发布提交。导入器仅更换 checkpoint 中的 `source_commit`，逐项重载验证其余字段完全一致，包括模型、优化器 scalar step/moments、scheduler、scaler、两 rank RNG、sampler、history、初始化 hash 和 manifest hash。原 checkpoint 文件 hash 不变；旧B写 hold，禁止同时续跑。
- 这是明确记录的源码版本边界，称为 `state_preserving_source_version_continuation`，不是同源码 exact resume，不是第二个独立科学实验，也没有重做已完成的两轮或增加参考/主训练轮数。
- C之后的同版本恢复仍严格要求本身的源码、配置、数据和完整恢复状态一致。旧A保持停止。

## 测量与改动

服务器：2×RTX4090 24GiB，双路 Xeon Silver4314、32物理/64逻辑CPU、约125.5GiB内存，两GPU跨NUMA。持续参考训练采样中GPU利用率约34%/40%，进程占用显存约1.9GiB；输入读取进程有空余，单纯增加worker或占满内存没有已验证收益。

CPU NUMA绑定试验约84.68→85.76 images/s，变化太小，不采用，线程调度已恢复。显存占用率不直接等于吞吐瓶颈，也不以占满资源为优化指标。

采用两处保持数学和数据流的修改：

1. 梯度逐参数 finite 标志在GPU上合并，保留同一 int64 all-rank MIN 和更新前失败阻断，减少 Python `all(cuda_tensor)` 引起的逐参数主机同步。
2. 参考健康统计保留每图FP32 moment、FP64累计、相同四尺度、同样样本数和双rank归约。登记 micro2 内尺度ID互异，用 `index_add_` 代替多次动态布尔索引；不改变健康阈值或统计定义。

不改batch/microbatch/累积、输入大小、训练顺序/增强/noise RNG、精度、模型、loss、LR、数据、冻结机制、校准、All8预算、选模或最终终点。没有启用全量缓存、torch.compile或修改环境依赖。

有界512/BF16/双rank对照使用64张已登记真实训练角色图，固定输入驻留GPU，每种实现两个40-update序列，前8次预热。修正诊断脚本的 optimizer state 深拷贝后，对照中模型参数hash、优化器状态hash、CUDA RNG hash、全部逐步loss/梯度范数和健康moment累计逐值相同。初次未深拷贝的诊断无效，不作为结论；所有诊断权重仅在内存中，未写正式checkpoint。

单独统计改写提速约2%，组合优化为小幅改善；记录原始窗口及正式续训后的实测，不把驻留输入基准当作全程加速比。见 `analysis_reports/tect_diff/refnorm_v2_performance.json`。方向依据亦与 [PyTorch performance tuning](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html) 关于避免CPU–GPU同步的说明一致。

正式C的step1550–2100窗口为0.18651秒/update、85.78 images/s；B的step100–450为0.18894秒/update、84.68 images/s，观测仅约1.3%改善，不能据此宣称显著全程提速。同期GPU平均利用率32.8%/41.7%，各1896MiB显存；CPU整体忙碌约7.3%，I/O等待约0.003%。参考网络在固定micro2下仍未充分占满GPU，当前证据不支持以增加读取进程或占满RAM来解决；主模型实际资源与吞吐由前期定时检查继续核对。

## 操作与监督

发布后在服务器运行：

```bash
/data0/hl/conda_envs/dcdsdiff/bin/python -B -s scripts/tect_diff/controller.py launch \
  --config configs/tect_diff/full_r512_s42_refnorm_v2.json \
  --run-id TECT-DIFF-FULL-R512-S42-REFNORM-V2-PERF-20260915-C \
  --reference-parent /data1/hl/DcDsDiff-and-GIT10K/runs/TECT-DIFF-FULL-R512-S42-REFNORM-V2-20260915-B
```

导入失败会保留新run的操作hold，不能用半完成导入启动。控制器仍先运行新快照的双卡preflight，再从导入状态续训。临时测试目录在realpath/marker/无符号链接核验后清理，保留小型验证收据。

原30分钟定时任务转为检查C的前期健康与实际吞吐，继承B已经完成的参考轮次。参考全部20轮、最终权重检查、校准和主训练前三轮均正常后暂停定时监督，健康训练继续至epoch99。每次检查仅少量状态/指标/日志，不把全量资源采样变为常驻本地轮询。

selection_protocol=test_selected：已有All8参与主训练选模；工程提速、参考学习正常都不等于定位成绩或创新收益。
