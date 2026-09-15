# TECT-DIFF-FULL-R512-S42-20260915-A

状态：**RUNNING**；阶段：REFERENCE。尚未完成正式主训练与固定终点汇总。

按八测试集逐 epoch F1 选择 checkpoint；这些测试集参与模型选择，不是独立泛化评估。selection_protocol=test_selected。固定终点 final.pth 与 test-selected best.pth 分开报告。

## Provenance

- Source commit: `4b4b785103f10ec9d9281cc8af095e3f91609433`
- Config SHA256: `a4f51026de58c24351b69ea9b60cf3db83ef22e1c1b12bc6db5c410ae6ef6a12`
- Environment: `/data0/hl/conda_envs/dcdsdiff`
- GPUs: [0, 1]; input 512 × 512; seed 42.
- Reference source mode: `authentic`.
- 全部有效训练图参加定位训练；参考拟合和校准仅使用训练内角色，不划分选模验证集。
- 观测锚定的 Image 加噪—去噪与 Mask 反向生成；不声称恢复篡改者的实际生成轨迹。

## Data and training-only fitting

```json
{
  "counts": {
    "train_count": 48612,
    "reference_count": 11964,
    "calibration_count": 2048,
    "test_counts": {
      "Casiav1": 920,
      "Columbia": 180,
      "DSO-1": 100,
      "IMD2020": 2010,
      "Korus": 220,
      "NIST16": 564,
      "coverage": 100,
      "wild": 201
    },
    "excluded_train_count": 17
  },
  "exclusions": {
    "excluded_train_count": 17,
    "rule": "exclude train records with exact decoded RGB or known namespaced source IDs present in tests; quarantine image-mask dimension mismatches as invalid training pairs; test anomalies block; no source files changed",
    "exact_rgb_overlap_count": 0,
    "known_source_overlap_count": 0,
    "records": 17,
    "counts_may_overlap": true
  },
  "manifest_hashes": {
    "calibration": "a22183506365f042e141c16b09a8c9e369c3f7886864c2a1d5326a4ada5e38c5",
    "reference": "0a501e3f0570b245ac594941fcbf2328d1bed195835825de8b7a758f62a13f79",
    "test_Casiav1": "a51508ac9da6618b685c807415674a78cd99b0681af7d4dae59b9947d187bc73",
    "test_Columbia": "dfcd1d818b0ec1250023b94e1b92c7292fb3ed107a47cbcf19d2050742594307",
    "test_DSO-1": "5265d478ca803e135518f2f9f5b0ecfadaaedf5a54dda022db310e92ee985fd6",
    "test_IMD2020": "5a74e0978c0ac97160eff982abc03ade617844dec53355671ace86b8a05eb766",
    "test_Korus": "22af2a4ce976c31a2881a87090878ecfbf9e08dc25a6baba3c9bbd1bfe38c222",
    "test_NIST16": "33c9b85a5facdc76722fc64a36f557dff457ad72f45530250e970e1c9e649a2f",
    "test_coverage": "e37249f77cd4de583a77a4d2887ffcb5fc0623fe31174535ab3b4c42dafa1783",
    "test_wild": "25a688eb600bc1dd9720c18c8eaff6fb6877549b3a041a691f3d17b14021fcb3",
    "train": "ba5c7de4a3988f1abe6fa34dd3d2424abd2712fde4754a2adab6aa9637ea74cc"
  },
  "stage_budgets": {
    "reference_epochs": 20,
    "calibration_max_images": 2048,
    "main_epochs": 100
  }
}
```

## Actual progress

```json
{
  "amp_skipped": false,
  "elapsed_seconds": 286.0782602503896,
  "epoch": 2,
  "gradient_norm": 1.8476111888885498,
  "loss": 0.15094396471977234,
  "micro_batch": 216,
  "micro_batches_per_epoch": 2991,
  "optimizer_step": 1550,
  "peak_memory_bytes": 970050048,
  "pid": 448690,
  "rank": 0,
  "stage": "REFERENCE_TRAINING",
  "updated_at": "2026-09-15T08:24:05Z"
}
```

### Test-selected best

尚无此终点的完整 checkpoint / 八集结果。

### Fixed final endpoint

尚无此终点的完整 checkpoint / 八集结果。

## Phase receipts and costs

```json
{
  "reference": {},
  "calibration": {},
  "costs": {
    "preflight_seconds": 60.00244477391243,
    "preparation_seconds": 10.452559020370245
  }
}
```

## Mechanisms and limitations

参考参数与训练内统计冻结；Image loss 必须非零启用。梯度/异常存在仅说明工程路径可运行，不代表方法有效。
q 全零、gamma 退化、Image 退化或控制损害定位以 diagnostics_summary.json 记录的实际诊断为准；缺失项仍待完成。
以下为固定低频诊断样本的统计，不能冒充所有训练像素的总体统计。

```json
{
  "flags": {
    "all_observed_q_zero": null,
    "all_observed_gamma_below_1e_minus4": null,
    "all_observed_image_loss_below_1e_minus12": null,
    "all_observed_joint_ref_mse_below_1e_minus12": null
  },
  "peak_allocated_bytes_by_rank": {
    "0": 970050048,
    "1": 970050048
  },
  "sample_profiles": [],
  "sampled_training": {}
}
```

Exact hashes and available name-level source IDs cannot exclude transformed or unrecognized same-source leakage; CASIA1 and CASIA2 IDs are separate namespaces
无同协议完整 baseline，只报告 TECT-Diff 自身结果，不声称超过 DcDsDiff。

## Reproduce / recover

```bash
/data0/hl/conda_envs/dcdsdiff/bin/python -s /data1/hl/DcDsDiff-and-GIT10K/runs/TECT-DIFF-FULL-R512-S42-20260915-A/source/scripts/tect_diff/controller.py status --run-dir /data1/hl/DcDsDiff-and-GIT10K/runs/TECT-DIFF-FULL-R512-S42-20260915-A
/data0/hl/conda_envs/dcdsdiff/bin/python -s /data1/hl/DcDsDiff-and-GIT10K/runs/TECT-DIFF-FULL-R512-S42-20260915-A/source/scripts/tect_diff/controller.py resume --run-dir /data1/hl/DcDsDiff-and-GIT10K/runs/TECT-DIFF-FULL-R512-S42-20260915-A
```

Source snapshot: `/data1/hl/DcDsDiff-and-GIT10K/runs/TECT-DIFF-FULL-R512-S42-20260915-A/source`; logs/status/checkpoints: `/data1/hl/DcDsDiff-and-GIT10K/runs/TECT-DIFF-FULL-R512-S42-20260915-A`.
模型权重、训练原图和诊断可视化不上传 GitHub。
