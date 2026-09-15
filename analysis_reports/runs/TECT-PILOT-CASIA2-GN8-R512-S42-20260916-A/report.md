# TECT-PILOT-CASIA2-GN8-R512-S42-20260916-A

状态：**FAILED**；结果：**HOLD**。

selection_protocol=test_selected：Casiav1/Columbia 用于开发、checkpoint 选择和推进判断，不是独立泛化评估。READY_FOR_FULL 只表示小规模验证通过，完整训练尚未执行。

## 数据与训练边界

训练：固定 CASIA2 子集；逐轮测试：固定 Casiav1/Columbia 子集。
连续健康门槛通过后，才对同一 checkpoint 执行一次完整两集确认；不在此任务启动完整训练。
源码：`adf3102c329a5bc019355f14cb142c6de916acdc`；配置：`0f9c2112331c542669cd27c6a83566ab78dbb7ea34ecdaebe80d8d911601e8cb`。

## 逐轮 Pixel-F1

| Epoch | Casiav1 | Columbia | Average Test2 |
| ---: | ---: | ---: | ---: |

尚无完成的训练及测试轮次。

详细记录：[metrics_per_epoch.jsonl](metrics_per_epoch.jsonl)。

完整结果与确认收据：[report.json](report.json)。服务器 run：`/data1/hl/DcDsDiff-and-GIT10K/runs/TECT-PILOT-CASIA2-GN8-R512-S42-20260916-A`。
权重留在服务器；已有 BN 运行与本 GN 架构不能作为相同协议的数值对照。

## 停止原因

ValueError: Data audit blocked by 2 semantic/test errors; see /data1/hl/DcDsDiff-and-GIT10K/cache/tect_diff/pilot-data-v1/audit/audit_errors.json
