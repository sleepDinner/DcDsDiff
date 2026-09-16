# TECT-PILOT-CASIA2-GN8-N8192-R512-S42-20260916-C

状态：**COMPLETED**；结果：**NO_GO**。

selection_protocol=test_selected：Casiav1/Columbia 用于开发、checkpoint 选择和推进判断，不是独立泛化评估。READY_FOR_FULL 只表示小规模验证通过，完整训练尚未执行。

## 数据与训练边界

训练：固定 CASIA2 子集；逐轮测试：固定 Casiav1/Columbia 子集。
连续健康门槛通过后，才对同一 checkpoint 执行一次完整两集确认；不在此任务启动完整训练。
源码：`02691917da17016f64f26ef820d5ccfa175e18ab`；配置：`c42769f4f4fedc18f5ae21ae4c6525bb551a416e2e5ffd98ed067bc6877bd702`。

## 逐轮 Pixel-F1

| Epoch | Casiav1 | Columbia | Average Test2 |
| ---: | ---: | ---: | ---: |
| 1 | 0.273522 | 0.323797 | 0.298659 |
| 2 | 0.306777 | 0.566251 | 0.436514 |
| 3 | 0.418828 | 0.356285 | 0.387557 |
| 4 | 0.470221 | 0.518329 | 0.494275 |
| 5 | 0.450471 | 0.564519 | 0.507495 |
| 6 | 0.362152 | 0.090635 | 0.226393 |
| 7 | 0.366677 | 0.316397 | 0.341537 |
| 8 | 0.381515 | 0.426258 | 0.403887 |
| 9 | 0.467807 | 0.382786 | 0.425297 |
| 10 | 0.341252 | 0.129035 | 0.235143 |

详细记录：[metrics_per_epoch.jsonl](metrics_per_epoch.jsonl)。

完整结果与确认收据：[report.json](report.json)。服务器 run：`/data1/hl/DcDsDiff-and-GIT10K/runs/TECT-PILOT-CASIA2-GN8-N8192-R512-S42-20260916-C`。
权重留在服务器；已有 BN 运行与本 GN 架构不能作为相同协议的数值对照。
