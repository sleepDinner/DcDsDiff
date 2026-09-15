# TECT-DIFF-FULL-R512-S42-REFNORM-V2-AMPFIX-20260915-D

状态：**RUNNING**；阶段：MAIN。尚未完成正式主训练与固定终点汇总。

按八测试集逐 epoch F1 选择 checkpoint；这些测试集参与模型选择，不是独立泛化评估。selection_protocol=test_selected。固定终点 final.pth 与 test-selected best.pth 分开报告。

## Provenance

- Source commit: `84fcf47e5721317d99fe186feec08d365fc3f72c`
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
  "A_mean": 0.18292422592639923,
  "A_negative_fraction": 0.19117355346679688,
  "amp_skipped": false,
  "control_logit_change": 0.00020316551672294736,
  "elapsed_seconds": 459.9378041625023,
  "ell_mean": -0.006009506527334452,
  "epoch": 0,
  "gamma": 0.03407493233680725,
  "gradient_norm": 0.8506653904914856,
  "gradient_sync": {
    "at": "2026-09-15T11:59:55Z",
    "coverage_pass_by_rank": [
      true,
      true
    ],
    "gradient_norm_by_rank": [
      0.8506653904914856,
      0.8506653904914856
    ],
    "gradient_tensors_by_rank": [
      856,
      856
    ],
    "passed": true
  },
  "image_loss": 0.05114854499697685,
  "joint_ref_mse": 0.0007977158529683948,
  "loss": 1.4046125411987305,
  "mask_loss": 1.3534640073776245,
  "micro_batch": 700,
  "micro_batches_per_epoch": 12153,
  "optimizer_step": 350,
  "peak_memory_bytes": 7110110208,
  "pid": 600611,
  "prefix": 2,
  "q_mean": 0.21191680431365967,
  "q_zero_fraction": 0.0233001708984375,
  "rank": 0,
  "sampling_step": 3,
  "stage": "MAIN_TRAINING",
  "updated_at": "2026-09-15T11:59:55Z"
}
```

### Test-selected best

尚无此终点的完整 checkpoint / 八集结果。

### Fixed final endpoint

尚无此终点的完整 checkpoint / 八集结果。

## Phase receipts and costs

```json
{
  "reference": {
    "architecture_version": "tect-reference-pixel-unet-v2-stable-paths",
    "artifact_path": "/data1/hl/DcDsDiff-and-GIT10K/artifacts/reference/TECT-DIFF-FULL-R512-S42-REFNORM-V2-PERF-20260915-C/reference_final.pth",
    "completed_at": "2026-09-15T10:43:15Z",
    "config_hash": "0b0358aa4908b852bb53113467184483fbb11d4be517eb58548ba80757052f88",
    "elapsed_seconds": 2486.857013281435,
    "epoch": 19,
    "final_health_probe": {
      "architecture_version": "tect-reference-pixel-unet-v2-stable-paths",
      "checkpoint_selection": "fixed_final_only_no_reselection",
      "ddp_padding": false,
      "image_count": 16,
      "image_noise_observations_per_scale": 32,
      "lambdas": [
        2,
        3,
        4,
        5
      ],
      "metrics": {
        "epoch": 19,
        "mse_by_scale": [
          0.03451702056918293,
          0.04682891140691936,
          0.06367181369569153,
          0.08684402983635664
        ],
        "mse_ratio_by_scale": [
          0.03450432840680501,
          0.04681169207171148,
          0.06364840110141204,
          0.08681209664774242
        ],
        "prediction_energy_by_scale": [
          0.9665694944560528,
          0.9556144587695599,
          0.9413436762988567,
          0.9262109212577343
        ],
        "prediction_energy_ratio_by_scale": [
          0.966214079742648,
          0.9552630723033645,
          0.9409975373043548,
          0.9258703466885808
        ],
        "prediction_noise_correlation_by_scale": [
          0.9825964555943345,
          0.9763142073464643,
          0.9676555887442707,
          0.9556313676432615
        ],
        "prediction_noise_cross_by_scale": [
          0.9662101529538631,
          0.9545766934752464,
          0.9390198551118374,
          0.919867355376482
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
          "final_endpoint_checked": true,
          "final_ratio_threshold": 0.98,
          "first_learned_epoch": 19,
          "learning_ratio_threshold": 0.98,
          "policy_id": "REFERENCE-NOISE-HEALTH-v1",
          "state": "LEARNED",
          "statistics_source": "fixed_final_weights_training_reference_probe"
        },
        "samples_by_scale": [
          16,
          16,
          16,
          16
        ],
        "zero_predictor_mse_by_scale": [
          1.0003678426146507,
          1.0003678426146507,
          1.0003678426146507,
          1.0003678426146507
        ]
      },
      "micro_batch": 2,
      "model_parameter_hash": "7fcf803535527efb7b42979e6ffa9f9454e7b308e28dafa3dac6a8744730d828",
      "noise_purposes": [
        "reference-final-probe-replica-0",
        "reference-final-probe-replica-1"
      ],
      "parameters_unchanged": true,
      "probe_id": "REFERENCE-FIXED-FINAL-TRAINING-PROBE-v1",
      "reference_manifest_sha256": "0a501e3f0570b245ac594941fcbf2328d1bed195835825de8b7a758f62a13f79",
      "replicas": 2,
      "resolution": 512,
      "rng_and_modes_restored": true,
      "scope": "training_reference_images_only_no_validation_split_or_test_data",
      "seed": 42,
      "selected_reference_ids": [
        "FinalTrainData:Au_ani_00002",
        "FinalTrainData:Au_ani_00003",
        "FinalTrainData:Au_ani_00004",
        "FinalTrainData:Au_ani_00006",
        "FinalTrainData:Au_ani_00011",
        "FinalTrainData:Au_ani_00013",
        "FinalTrainData:Au_ani_00017",
        "FinalTrainData:Au_ani_00018",
        "FinalTrainData:Au_ani_00019",
        "FinalTrainData:Au_ani_00020",
        "FinalTrainData:Au_ani_00022",
        "FinalTrainData:Au_ani_00024",
        "FinalTrainData:Au_ani_00025",
        "FinalTrainData:Au_ani_00027",
        "FinalTrainData:Au_ani_00028",
        "FinalTrainData:Au_ani_00029"
      ],
      "selection": "first_16_in_frozen_reference_manifest_order",
      "shared_noise_across_scales": true,
      "training_augmentation": false
    },
    "final_parameter_hash": "7fcf803535527efb7b42979e6ffa9f9454e7b308e28dafa3dac6a8744730d828",
    "health_metrics": {
      "epoch": 19,
      "mse_by_scale": [
        0.04705284501944984,
        0.06933950953775925,
        0.10231307664632658,
        0.15306649952740803
      ],
      "mse_ratio_by_scale": [
        0.04705354736842748,
        0.06933632493333626,
        0.10231176212651907,
        0.15307022817873817
      ],
      "optimizer_step": 14960,
      "padded_training_samples": 0,
      "prediction_energy_by_scale": [
        0.949628342955617,
        0.9273398385841845,
        0.894701914498735,
        0.8402182572822992
      ],
      "prediction_energy_ratio_by_scale": [
        0.9496425178794826,
        0.9272972479951933,
        0.8946904193563058,
        0.8402387247322788
      ],
      "prediction_noise_correlation_by_scale": [
        0.9761912373069056,
        0.9647106996623951,
        0.9474654366732265,
        0.9202951126132125
      ],
      "prediction_noise_cross_by_scale": [
        0.9512802883957748,
        0.9290231312449183,
        0.896200845330027,
        0.8435637002424108
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
        "final_endpoint_checked": true,
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
        0.9999850734106795,
        1.0000459298127793,
        1.000012848178745,
        0.9999756409108779
      ]
    },
    "initial_parameter_hash": "6f127a7e0904fd481bf2db1dbf5449e55eb49c352513eab035be5670f6236bec",
    "mean_epsilon_mse": 0.09294298268273593,
    "optimizer_step": 14960,
    "protocol_id": "TECT-DIFF-FULL-R512-S42-REFNORM-V2",
    "reference_source_mode": "authentic",
    "sample_count": 11964,
    "sha256": "800f50c392a275fde3ec249c6f52275c6db79813e6f8b7e8b534f9ddb52a523f",
    "status": "COMPLETED",
    "zero_predictor_mse": 1.0000048730782705
  },
  "reference_health": {
    "epoch": 19,
    "mse_by_scale": [
      0.04705284501944984,
      0.06933950953775925,
      0.10231307664632658,
      0.15306649952740803
    ],
    "mse_ratio_by_scale": [
      0.04705354736842748,
      0.06933632493333626,
      0.10231176212651907,
      0.15307022817873817
    ],
    "optimizer_step": 14960,
    "padded_training_samples": 0,
    "prediction_energy_by_scale": [
      0.949628342955617,
      0.9273398385841845,
      0.894701914498735,
      0.8402182572822992
    ],
    "prediction_energy_ratio_by_scale": [
      0.9496425178794826,
      0.9272972479951933,
      0.8946904193563058,
      0.8402387247322788
    ],
    "prediction_noise_correlation_by_scale": [
      0.9761912373069056,
      0.9647106996623951,
      0.9474654366732265,
      0.9202951126132125
    ],
    "prediction_noise_cross_by_scale": [
      0.9512802883957748,
      0.9290231312449183,
      0.896200845330027,
      0.8435637002424108
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
      "final_endpoint_checked": true,
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
      0.9999850734106795,
      1.0000459298127793,
      1.000012848178745,
      0.9999756409108779
    ]
  },
  "reference_final_health": {
    "architecture_version": "tect-reference-pixel-unet-v2-stable-paths",
    "checkpoint_selection": "fixed_final_only_no_reselection",
    "ddp_padding": false,
    "image_count": 16,
    "image_noise_observations_per_scale": 32,
    "lambdas": [
      2,
      3,
      4,
      5
    ],
    "metrics": {
      "epoch": 19,
      "mse_by_scale": [
        0.03451702056918293,
        0.04682891140691936,
        0.06367181369569153,
        0.08684402983635664
      ],
      "mse_ratio_by_scale": [
        0.03450432840680501,
        0.04681169207171148,
        0.06364840110141204,
        0.08681209664774242
      ],
      "prediction_energy_by_scale": [
        0.9665694944560528,
        0.9556144587695599,
        0.9413436762988567,
        0.9262109212577343
      ],
      "prediction_energy_ratio_by_scale": [
        0.966214079742648,
        0.9552630723033645,
        0.9409975373043548,
        0.9258703466885808
      ],
      "prediction_noise_correlation_by_scale": [
        0.9825964555943345,
        0.9763142073464643,
        0.9676555887442707,
        0.9556313676432615
      ],
      "prediction_noise_cross_by_scale": [
        0.9662101529538631,
        0.9545766934752464,
        0.9390198551118374,
        0.919867355376482
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
        "final_endpoint_checked": true,
        "final_ratio_threshold": 0.98,
        "first_learned_epoch": 19,
        "learning_ratio_threshold": 0.98,
        "policy_id": "REFERENCE-NOISE-HEALTH-v1",
        "state": "LEARNED",
        "statistics_source": "fixed_final_weights_training_reference_probe"
      },
      "samples_by_scale": [
        16,
        16,
        16,
        16
      ],
      "zero_predictor_mse_by_scale": [
        1.0003678426146507,
        1.0003678426146507,
        1.0003678426146507,
        1.0003678426146507
      ]
    },
    "micro_batch": 2,
    "model_parameter_hash": "7fcf803535527efb7b42979e6ffa9f9454e7b308e28dafa3dac6a8744730d828",
    "noise_purposes": [
      "reference-final-probe-replica-0",
      "reference-final-probe-replica-1"
    ],
    "parameters_unchanged": true,
    "probe_id": "REFERENCE-FIXED-FINAL-TRAINING-PROBE-v1",
    "reference_manifest_sha256": "0a501e3f0570b245ac594941fcbf2328d1bed195835825de8b7a758f62a13f79",
    "replicas": 2,
    "resolution": 512,
    "rng_and_modes_restored": true,
    "scope": "training_reference_images_only_no_validation_split_or_test_data",
    "seed": 42,
    "selected_reference_ids": [
      "FinalTrainData:Au_ani_00002",
      "FinalTrainData:Au_ani_00003",
      "FinalTrainData:Au_ani_00004",
      "FinalTrainData:Au_ani_00006",
      "FinalTrainData:Au_ani_00011",
      "FinalTrainData:Au_ani_00013",
      "FinalTrainData:Au_ani_00017",
      "FinalTrainData:Au_ani_00018",
      "FinalTrainData:Au_ani_00019",
      "FinalTrainData:Au_ani_00020",
      "FinalTrainData:Au_ani_00022",
      "FinalTrainData:Au_ani_00024",
      "FinalTrainData:Au_ani_00025",
      "FinalTrainData:Au_ani_00027",
      "FinalTrainData:Au_ani_00028",
      "FinalTrainData:Au_ani_00029"
    ],
    "selection": "first_16_in_frozen_reference_manifest_order",
    "shared_noise_across_scales": true,
    "training_augmentation": false
  },
  "reference_continuation": {},
  "artifact_continuation": {
    "at": "2026-09-15T11:51:01Z",
    "calibration_refitted": false,
    "calibration_sha256": "bb349a143385b25798fe0a9c7c43904ba08c26f38ed7a6aee991add7ed16f59c",
    "changed_python_files": [
      "model/tect_diff/amp_context.py",
      "model/tect_diff/diffusion.py",
      "scripts/tect_diff/artifact_continuation.py",
      "scripts/tect_diff/controller.py",
      "scripts/tect_diff/diagnostics.py",
      "scripts/tect_diff/gradient_safety.py",
      "scripts/tect_diff/report.py",
      "scripts/tect_diff/test_artifact_continuation.py",
      "scripts/tect_diff/test_main_gradient_safety.py",
      "scripts/tect_diff/worker.py"
    ],
    "config_hash": "0b0358aa4908b852bb53113467184483fbb11d4be517eb58548ba80757052f88",
    "copied_receipt_sha256": {
      "calibration_receipt.json": "89562ed608254b9d99412b5b914b9a5266d2442cca6f9949d26d4f351829a99b",
      "data_bundle.json": "157cdc70e1a317203a08f02a35b075c2ec7de277de70b71febc21997b6045186",
      "reference_final_health.json": "c1600fc6eda466b211629d60c2596cb107755719384bdcd90b3184099a1cbf9a",
      "reference_health.json": "36f3010c2377b87715be27e14670cffd2139ae18d0ccfc704c3c12f96207ddda",
      "reference_metrics.jsonl": "d69517a38211e9edb29cc410bb3ffbde01aa60bc7c1846f401e9cb5eaed0ff15",
      "reference_receipt.json": "fdf3793d063d5a529a470307f4bb10ccca1d89797c04be9a0de21e503b303ff1"
    },
    "main_seed": 42,
    "main_start_epoch": 0,
    "main_start_optimizer_step": 0,
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
    "mode": "reuse_completed_reference_calibration_restart_invalid_main",
    "parent_main_checkpoint_reused": false,
    "parent_partial_main_valid": false,
    "parent_run": "TECT-DIFF-FULL-R512-S42-REFNORM-V2-PERF-20260915-C",
    "parent_source_commit": "f9c1b90d2fc19c914dd3998f49c2c6665d580ca2",
    "reference_epochs_retrained": 0,
    "reference_sha256": "800f50c392a275fde3ec249c6f52275c6db79813e6f8b7e8b534f9ddb52a523f",
    "selection_protocol": "test_selected",
    "source_commit": "84fcf47e5721317d99fe186feec08d365fc3f72c",
    "status": "COMPLETED"
  },
  "operational_hold": {
    "active": false,
    "at": "2026-09-15T11:51:01Z",
    "reason": "Completed dependencies verified; corrected MAIN starts fresh; see artifact_continuation.json"
  },
  "calibration": {
    "artifact_hash": "bcb2f80690dff8632a5af39c3c3263f29460aede883f93335fbf45c36b470f43",
    "artifact_path": "/data1/hl/DcDsDiff-and-GIT10K/artifacts/calibration/TECT-DIFF-FULL-R512-S42-REFNORM-V2-PERF-20260915-C/calibration.pth",
    "completed_at": "2026-09-15T10:45:25Z",
    "config_hash": "0b0358aa4908b852bb53113467184483fbb11d4be517eb58548ba80757052f88",
    "elapsed_seconds": 96.11043768748641,
    "fit_receipt": {
      "content_fit": {
        "coefficient_l2": 0.8784816861152649,
        "fitting_pixels": 131072,
        "iterations": 4,
        "negative_pixels": 87245,
        "positive_pixels": 43827,
        "regularized_objective": 0.6183885335922241
      },
      "images": 2048,
      "normal_pixels": 262016,
      "normal_statistics": "only known GT=0 training query pixels",
      "sample_ids_sha256": "d08a7359e8725a2d45b7642c35975fa86ea35472d88e9f3070b4e4a2670af71f",
      "sampling": "up to 128 uniform pixels per available class per image; equal pixel weight; identical joint/content sample",
      "semantics": "estimated content-subtracted evidence; no conditional-density guarantee",
      "tampered_pixels": 130561
    },
    "prefix_metadata": {
      "1": {
        "A_quantiles": [
          -7.880748748779297,
          0.018113723024725914,
          0.12383788824081421,
          0.3908524513244629,
          61.65579605102539
        ],
        "W_normalization": "inverse shrunk variance divided by dimension",
        "W_sha256": "31e596978019c5eea3f77fd720a18058a76f1cd835bf6874f16836a6050caf04",
        "anomaly_scale": 0.13599398732185364,
        "control_enabled": false,
        "d0": 0.1414022594690323,
        "dimension": 9,
        "ell_mean": -0.000751599611248821,
        "ell_std": 0.04960409551858902,
        "joint_fit": {
          "coefficient_l2": 0.9058691263198853,
          "fitting_pixels": 131072,
          "iterations": 4,
          "negative_pixels": 87245,
          "positive_pixels": 43827,
          "regularized_objective": 0.6180851459503174
        },
        "negative_A_fraction": 0.204996719956398,
        "normal_trajectory_rows": 524032,
        "q_mean": 0.21458935737609863,
        "q_zero_fraction": 0.0,
        "v0": 0.2336273193359375
      },
      "2": {
        "A_quantiles": [
          -8.633238792419434,
          0.025318020954728127,
          0.10904906690120697,
          0.35280659794807434,
          52.047515869140625
        ],
        "W_normalization": "inverse shrunk variance divided by dimension",
        "W_sha256": "2b7499bec14dca40816894214005791a66a3b6c4e9a2310b607519b34bfd9c72",
        "anomaly_scale": 0.1128426045179367,
        "control_enabled": true,
        "d0": 0.1414022594690323,
        "dimension": 27,
        "ell_mean": -0.0009161768830381334,
        "ell_std": 0.05373438075184822,
        "joint_fit": {
          "coefficient_l2": 0.9088301062583923,
          "fitting_pixels": 131072,
          "iterations": 4,
          "negative_pixels": 87245,
          "positive_pixels": 43827,
          "regularized_objective": 0.6180295348167419
        },
        "negative_A_fraction": 0.16642339527606964,
        "normal_trajectory_rows": 524032,
        "q_mean": 0.2145639955997467,
        "q_zero_fraction": 0.0,
        "v0": 0.19703319668769836
      },
      "3": {
        "A_quantiles": [
          -6.8743896484375,
          0.029178602620959282,
          0.1082027405500412,
          0.34457486867904663,
          40.679908752441406
        ],
        "W_normalization": "inverse shrunk variance divided by dimension",
        "W_sha256": "b56377d1e5e030a8c695867bff70b968ad3583dd50f77e27ef1581f26de4357d",
        "anomaly_scale": 0.11086861789226532,
        "control_enabled": true,
        "d0": 0.1414022594690323,
        "dimension": 45,
        "ell_mean": -0.0010517120826989412,
        "ell_std": 0.05607052892446518,
        "joint_fit": {
          "coefficient_l2": 0.9096332788467407,
          "fitting_pixels": 131072,
          "iterations": 4,
          "negative_pixels": 87245,
          "positive_pixels": 43827,
          "regularized_objective": 0.6179935336112976
        },
        "negative_A_fraction": 0.14441242814064026,
        "normal_trajectory_rows": 524032,
        "q_mean": 0.21562372148036957,
        "q_zero_fraction": 0.0,
        "v0": 0.18079955875873566
      },
      "4": {
        "A_quantiles": [
          -6.197851181030273,
          0.03362632915377617,
          0.11223015189170837,
          0.33935195207595825,
          37.91045379638672
        ],
        "W_normalization": "inverse shrunk variance divided by dimension",
        "W_sha256": "fef4391de6ebf63e4cf5cb783625445e93630af6cd65e7c7e7da74807e985232",
        "anomaly_scale": 0.1134263351559639,
        "control_enabled": true,
        "d0": 0.1414022594690323,
        "dimension": 63,
        "ell_mean": -0.0010528421262279153,
        "ell_std": 0.055919308215379715,
        "joint_fit": {
          "coefficient_l2": 0.9076429605484009,
          "fitting_pixels": 131072,
          "iterations": 4,
          "negative_pixels": 87245,
          "positive_pixels": 43827,
          "regularized_objective": 0.6179834604263306
        },
        "negative_A_fraction": 0.12565432488918304,
        "normal_trajectory_rows": 524032,
        "q_mean": 0.21798691153526306,
        "q_zero_fraction": 0.0,
        "v0": 0.17477399110794067
      }
    },
    "protocol_id": "TECT-DIFF-FULL-R512-S42-REFNORM-V2",
    "sha256": "bb349a143385b25798fe0a9c7c43904ba08c26f38ed7a6aee991add7ed16f59c",
    "status": "COMPLETED"
  },
  "costs": {
    "preflight_seconds": 60.0024257004261
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
    "all_observed_q_zero": false,
    "all_observed_gamma_below_1e_minus4": false,
    "all_observed_image_loss_below_1e_minus12": false,
    "all_observed_joint_ref_mse_below_1e_minus12": false
  },
  "peak_allocated_bytes_by_rank": {
    "0": 7110110208,
    "1": 7116528128
  },
  "sample_profiles": [
    {
      "rank": 0,
      "cold_reference_and_measurement_seconds": 0.94218909740448,
      "cross_call_cache": false,
      "image_count": 1,
      "mask_ten_steps_with_reference_reuse_seconds": 0.5246180519461632,
      "reference_feature_requests": 20,
      "reference_feature_reuse_hits": 12,
      "reference_forwards": 8,
      "total_seconds": 1.4668071493506432
    },
    {
      "rank": 1,
      "cold_reference_and_measurement_seconds": 0.8202049024403095,
      "cross_call_cache": false,
      "image_count": 1,
      "mask_ten_steps_with_reference_reuse_seconds": 0.6377383694052696,
      "reference_feature_requests": 20,
      "reference_feature_reuse_hits": 12,
      "reference_forwards": 8,
      "total_seconds": 1.4579432718455791
    }
  ],
  "sampled_training": {
    "records": 8,
    "malformed_lines": 0,
    "metrics": {
      "A_mean": {
        "count": 8,
        "nonfinite_count": 0,
        "mean": 0.4623941648751497,
        "minimum": 0.18292422592639923,
        "maximum": 0.8028234839439392,
        "last": 0.18292422592639923,
        "kind": "scalar"
      },
      "A_negative_fraction": {
        "count": 8,
        "nonfinite_count": 0,
        "mean": 0.16966462135314944,
        "minimum": 0.09840583801269531,
        "maximum": 0.27401161193847656,
        "last": 0.19117355346679688,
        "kind": "scalar"
      },
      "amp_skipped": {
        "count": 8,
        "nonfinite_count": 0,
        "mean": 0.0,
        "minimum": 0,
        "maximum": 0,
        "last": 0,
        "kind": "boolean_frequency"
      },
      "control_logit_change": {
        "count": 8,
        "nonfinite_count": 0,
        "mean": 0.00026169620105065405,
        "minimum": 0.0,
        "maximum": 0.0005711392150260508,
        "last": 0.00020316551672294736,
        "kind": "scalar"
      },
      "ell_mean": {
        "count": 8,
        "nonfinite_count": 0,
        "mean": 0.0030589776288252333,
        "minimum": -0.012933689169585705,
        "maximum": 0.030825752764940262,
        "last": -0.006009506527334452,
        "kind": "scalar"
      },
      "epoch": {
        "count": 8,
        "nonfinite_count": 0,
        "mean": 0.0,
        "minimum": 0,
        "maximum": 0,
        "last": 0,
        "kind": "scalar"
      },
      "gamma": {
        "count": 8,
        "nonfinite_count": 0,
        "mean": 0.037924841977655895,
        "minimum": 0.0,
        "maximum": 0.0783112645149231,
        "last": 0.03407493233680725,
        "kind": "scalar"
      },
      "gradient_norm": {
        "count": 8,
        "nonfinite_count": 0,
        "mean": 1.1160809323191645,
        "minimum": 0.7197176814079285,
        "maximum": 3.1453287601470947,
        "last": 0.8506653904914856,
        "kind": "scalar"
      },
      "image_loss": {
        "count": 8,
        "nonfinite_count": 0,
        "mean": 0.11225769156590105,
        "minimum": 0.03354518860578537,
        "maximum": 0.19214731454849243,
        "last": 0.05114854499697685,
        "kind": "scalar"
      },
      "joint_ref_mse": {
        "count": 8,
        "nonfinite_count": 0,
        "mean": 0.01700487171183341,
        "minimum": 0.0007977158529683948,
        "maximum": 0.12436667084693909,
        "last": 0.0007977158529683948,
        "kind": "scalar"
      },
      "loss": {
        "count": 8,
        "nonfinite_count": 0,
        "mean": 1.4610667526721957,
        "minimum": 1.0817896127700806,
        "maximum": 1.7195851802825928,
        "last": 1.4046125411987305,
        "kind": "scalar"
      },
      "mask_loss": {
        "count": 8,
        "nonfinite_count": 0,
        "mean": 1.3488090634346013,
        "minimum": 0.9361345767974854,
        "maximum": 1.5605813264846802,
        "last": 1.3534640073776245,
        "kind": "scalar"
      },
      "optimizer_step": {
        "count": 8,
        "nonfinite_count": 0,
        "mean": 175.125,
        "minimum": 1,
        "maximum": 350,
        "last": 350,
        "kind": "scalar"
      },
      "peak_memory_bytes": {
        "count": 8,
        "nonfinite_count": 0,
        "mean": 7027214208.0,
        "minimum": 6453261312,
        "maximum": 7110110208,
        "last": 7110110208,
        "kind": "scalar"
      },
      "prefix": {
        "count": 8,
        "nonfinite_count": 0,
        "mean": 2.1250000000000004,
        "minimum": 1,
        "maximum": 3,
        "last": 2,
        "kind": "scalar"
      },
      "q_mean": {
        "count": 8,
        "nonfinite_count": 0,
        "mean": 0.20999204926192766,
        "minimum": 0.165652796626091,
        "maximum": 0.3143921494483948,
        "last": 0.21191680431365967,
        "kind": "scalar"
      },
      "q_zero_fraction": {
        "count": 8,
        "nonfinite_count": 0,
        "mean": 0.0233001708984375,
        "minimum": 0.0233001708984375,
        "maximum": 0.0233001708984375,
        "last": 0.0233001708984375,
        "kind": "scalar"
      },
      "sampling_step": {
        "count": 8,
        "nonfinite_count": 0,
        "mean": 3.6250000000000004,
        "minimum": 1,
        "maximum": 7,
        "last": 3,
        "kind": "scalar"
      }
    }
  }
}
```

Exact hashes and available name-level source IDs cannot exclude transformed or unrecognized same-source leakage; CASIA1 and CASIA2 IDs are separate namespaces
无同协议完整 baseline，只报告 TECT-Diff 自身结果，不声称超过 DcDsDiff。

## Reproduce / recover

```bash
/data0/hl/conda_envs/dcdsdiff/bin/python -s /data1/hl/DcDsDiff-and-GIT10K/runs/TECT-DIFF-FULL-R512-S42-REFNORM-V2-AMPFIX-20260915-D/source/scripts/tect_diff/controller.py status --run-dir /data1/hl/DcDsDiff-and-GIT10K/runs/TECT-DIFF-FULL-R512-S42-REFNORM-V2-AMPFIX-20260915-D
/data0/hl/conda_envs/dcdsdiff/bin/python -s /data1/hl/DcDsDiff-and-GIT10K/runs/TECT-DIFF-FULL-R512-S42-REFNORM-V2-AMPFIX-20260915-D/source/scripts/tect_diff/controller.py resume --run-dir /data1/hl/DcDsDiff-and-GIT10K/runs/TECT-DIFF-FULL-R512-S42-REFNORM-V2-AMPFIX-20260915-D
```

Source snapshot: `/data1/hl/DcDsDiff-and-GIT10K/runs/TECT-DIFF-FULL-R512-S42-REFNORM-V2-AMPFIX-20260915-D/source`; logs/status/checkpoints: `/data1/hl/DcDsDiff-and-GIT10K/runs/TECT-DIFF-FULL-R512-S42-REFNORM-V2-AMPFIX-20260915-D`.
模型权重、训练原图和诊断可视化不上传 GitHub。
