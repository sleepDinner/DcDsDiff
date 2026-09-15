# TECT-Diff experiment ledger

按八测试集逐 epoch F1 选择 checkpoint；这些测试集参与模型选择，不是独立泛化评估。selection_protocol=test_selected。固定终点 final.pth 与 test-selected best.pth 分开报告。

| Run | Status | Stage | Resolution | Best epoch / All8 F1 | Final epoch / All8 F1 |
| --- | --- | --- | ---: | --- | --- |
| [TECT-DIFF-FULL-R512-S42-20260915-A](runs/TECT-DIFF-FULL-R512-S42-20260915-A/report.md) | INTERRUPTED | REFERENCE | 512 |  /  |  /  |
| [TECT-DIFF-FULL-R512-S42-REFNORM-V2-20260915-B](runs/TECT-DIFF-FULL-R512-S42-REFNORM-V2-20260915-B/report.md) | INTERRUPTED | REFERENCE | 512 |  /  |  /  |
| [TECT-DIFF-FULL-R512-S42-REFNORM-V2-PERF-20260915-C](runs/TECT-DIFF-FULL-R512-S42-REFNORM-V2-PERF-20260915-C/report.md) | INTERRUPTED | MAIN | 512 |  / None |  / None |
