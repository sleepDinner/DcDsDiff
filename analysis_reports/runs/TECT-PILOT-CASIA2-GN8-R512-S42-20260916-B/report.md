# TECT-PILOT-CASIA2-GN8-R512-S42-20260916-B

状态：**COMPLETED**；结果：**NO_GO**。

selection_protocol=test_selected：Casiav1/Columbia 用于开发、checkpoint 选择和推进判断，不是独立泛化评估。READY_FOR_FULL 只表示小规模验证通过，完整训练尚未执行。

## 数据与训练边界

训练：固定 CASIA2 子集；逐轮测试：固定 Casiav1/Columbia 子集。
连续健康门槛通过后，才对同一 checkpoint 执行一次完整两集确认；不在此任务启动完整训练。
源码：`2ea2f5b855ef4d917fae4265e9e6d2ca7c1a6382`；配置：`174434a273212505913ed20b5469e9309776e656bf07f2da30e94eee8b9c36b8`。

## 逐轮 Pixel-F1

| Epoch | Casiav1 | Columbia | Average Test2 |
| ---: | ---: | ---: | ---: |
| 1 | 0.115193 | 0.167693 | 0.141443 |
| 2 | 0.230497 | 0.232411 | 0.231454 |
| 3 | 0.297329 | 0.234444 | 0.265886 |
| 4 | 0.330409 | 0.519150 | 0.424779 |
| 5 | 0.305634 | 0.180750 | 0.243192 |
| 6 | 0.355393 | 0.385013 | 0.370203 |
| 7 | 0.419469 | 0.447342 | 0.433405 |
| 8 | 0.428355 | 0.418729 | 0.423542 |
| 9 | 0.400106 | 0.440326 | 0.420216 |
| 10 | 0.403324 | 0.278310 | 0.340817 |

详细记录：[metrics_per_epoch.jsonl](metrics_per_epoch.jsonl)。

完整结果与确认收据：[report.json](report.json)。服务器 run：`/data1/hl/DcDsDiff-and-GIT10K/runs/TECT-PILOT-CASIA2-GN8-R512-S42-20260916-B`。
权重留在服务器；已有 BN 运行与本 GN 架构不能作为相同协议的数值对照。
