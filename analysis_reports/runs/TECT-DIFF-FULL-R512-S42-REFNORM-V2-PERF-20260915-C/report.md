# TECT-DIFF-FULL-R512-S42-REFNORM-V2-PERF-20260915-C

状态：**RUNNING**；阶段：REFERENCE。尚未完成正式主训练与固定终点汇总。

按八测试集逐 epoch F1 选择 checkpoint；这些测试集参与模型选择，不是独立泛化评估。selection_protocol=test_selected。固定终点 final.pth 与 test-selected best.pth 分开报告。

## Provenance

- Source commit: `f9c1b90d2fc19c914dd3998f49c2c6665d580ca2`
- Config SHA256: `0b0358aa4908b852bb53113467184483fbb11d4be517eb58548ba80757052f88`
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
  "elapsed_seconds": 309.9320814497769,
  "epoch": 4,
  "gradient_norm": 0.870252251625061,
  "loss": 0.10220833122730255,
  "micro_batch": 632,
  "micro_batches_per_epoch": 2991,
  "optimizer_step": 3150,
  "peak_memory_bytes": 990669824,
  "pid": 522403,
  "rank": 0,
  "stage": "REFERENCE_TRAINING",
  "updated_at": "2026-09-15T10:06:58Z"
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
  "reference_health": {
    "epoch": 3,
    "mse_by_scale": [
      0.08950342306451868,
      0.11432038330037392,
      0.15390143608247822,
      0.21864565773996292
    ],
    "mse_ratio_by_scale": [
      0.08950287163198192,
      0.11432187579199653,
      0.1538993436766898,
      0.21864087570621837
    ],
    "optimizer_step": 2992,
    "padded_training_samples": 0,
    "prediction_energy_by_scale": [
      0.8921338790278497,
      0.865841437306782,
      0.8295666980568042,
      0.7506751710357813
    ],
    "prediction_energy_ratio_by_scale": [
      0.8921283825716115,
      0.8658527411622658,
      0.8295554194735428,
      0.7506587529004467
    ],
    "prediction_noise_correlation_by_scale": [
      0.9542492767014854,
      0.9411650296562266,
      0.9198821432141405,
      0.8841227420873415
    ],
    "prediction_noise_cross_by_scale": [
      0.9013183092600683,
      0.8757540012274806,
      0.8378394307479616,
      0.7660256942073385
    ],
    "reference_health": {
      "all_scales_pass_final_margin": true,
      "checkpoint_selection": "fixed_final_only_no_reselection",
      "collapse_detected_epoch": null,
      "collapse_patience_epochs": 2,
      "collapse_ratio_threshold": 0.98,
      "consecutive_collapsed_epochs": 0,
      "correlation_definition": "uncentered_cross_over_sqrt_prediction_energy_times_noise_energy",
      "failed": false,
      "failure_reason": null,
      "final_endpoint_checked": false,
      "final_ratio_threshold": 0.98,
      "first_learned_epoch": 0,
      "learning_ratio_threshold": 0.98,
      "policy_id": "REFERENCE-NOISE-HEALTH-v1",
      "state": "LEARNED",
      "statistics_source": "training_images_only_all_rank_per_image_mean_sums"
    },
    "samples_by_scale": [
      2991,
      2991,
      2991,
      2991
    ],
    "zero_predictor_mse_by_scale": [
      1.0000061610597146,
      0.9999869448291303,
      1.0000135959370484,
      1.0000218716364409
    ]
  },
  "reference_final_health": {},
  "reference_continuation": {
    "all_other_state_exactly_equal": true,
    "at": "2026-09-15T10:00:33Z",
    "changed_checkpoint_fields": [
      "source_commit"
    ],
    "changed_python_files": [
      "scripts/tect_diff/controller.py",
      "scripts/tect_diff/reference_continuation.py",
      "scripts/tect_diff/report.py",
      "scripts/tect_diff/worker.py"
    ],
    "child_checkpoint_sha256": "70699e63742a27866d08ccff38ceda06e2639e1f75fe83df24b8ba1d92de64be",
    "config_hash": "0b0358aa4908b852bb53113467184483fbb11d4be517eb58548ba80757052f88",
    "epoch": 1,
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
    "mode": "state_preserving_source_version_continuation",
    "next_epoch": 2,
    "optimizer_step": 1496,
    "parent_checkpoint_sha256": "df51ace1ebccc1bd9177b83e164b4d9bac9a8eeeaef0c606ede5bcf60852d220",
    "parent_run": "TECT-DIFF-FULL-R512-S42-REFNORM-V2-20260915-B",
    "parent_source_commit": "100d46edb1f73dd59fa0d57277e07c2925c53278",
    "selection_protocol": "test_selected",
    "source_commit": "f9c1b90d2fc19c914dd3998f49c2c6665d580ca2",
    "status": "COMPLETED"
  },
  "operational_hold": {
    "active": false,
    "at": "2026-09-15T10:00:33Z",
    "reason": "Reference state import verified; see reference_continuation.json"
  },
  "calibration": {},
  "costs": {
    "preflight_seconds": 60.00258359313011
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
    "0": 990669824,
    "1": 990669824
  },
  "sample_profiles": [],
  "sampled_training": {}
}
```

Exact hashes and available name-level source IDs cannot exclude transformed or unrecognized same-source leakage; CASIA1 and CASIA2 IDs are separate namespaces
无同协议完整 baseline，只报告 TECT-Diff 自身结果，不声称超过 DcDsDiff。

## Reproduce / recover

```bash
/data0/hl/conda_envs/dcdsdiff/bin/python -s /data1/hl/DcDsDiff-and-GIT10K/runs/TECT-DIFF-FULL-R512-S42-REFNORM-V2-PERF-20260915-C/source/scripts/tect_diff/controller.py status --run-dir /data1/hl/DcDsDiff-and-GIT10K/runs/TECT-DIFF-FULL-R512-S42-REFNORM-V2-PERF-20260915-C
/data0/hl/conda_envs/dcdsdiff/bin/python -s /data1/hl/DcDsDiff-and-GIT10K/runs/TECT-DIFF-FULL-R512-S42-REFNORM-V2-PERF-20260915-C/source/scripts/tect_diff/controller.py resume --run-dir /data1/hl/DcDsDiff-and-GIT10K/runs/TECT-DIFF-FULL-R512-S42-REFNORM-V2-PERF-20260915-C
```

Source snapshot: `/data1/hl/DcDsDiff-and-GIT10K/runs/TECT-DIFF-FULL-R512-S42-REFNORM-V2-PERF-20260915-C/source`; logs/status/checkpoints: `/data1/hl/DcDsDiff-and-GIT10K/runs/TECT-DIFF-FULL-R512-S42-REFNORM-V2-PERF-20260915-C`.
模型权重、训练原图和诊断可视化不上传 GitHub。
