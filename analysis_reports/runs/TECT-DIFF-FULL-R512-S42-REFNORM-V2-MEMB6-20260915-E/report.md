# TECT-DIFF-FULL-R512-S42-REFNORM-V2-MEMB6-20260915-E

状态：**INTERRUPTED**；阶段：MAIN。尚未完成正式主训练与固定终点汇总。

按八测试集逐 epoch F1 选择 checkpoint；这些测试集参与模型选择，不是独立泛化评估。selection_protocol=test_selected。固定终点 final.pth 与 test-selected best.pth 分开报告。

## Provenance

- Source commit: `ad8fb6e5f3f6f295b1e94b87e28175a5fabc779c`
- Config SHA256: `63e53b3e50df4d4798d9310cb4ca2065efee1fbaea4fd2ba485fb88a37116275`
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
  "A_mean": 0.47452718019485474,
  "A_negative_fraction": 0.2145099639892578,
  "amp_skipped": false,
  "control_logit_change": 0.0,
  "elapsed_seconds": 15051.884952761233,
  "ell_mean": -0.004101177677512169,
  "epoch": 2,
  "gamma": 0.0,
  "gradient_norm": 1.0622400045394897,
  "gradient_sync": {
    "at": "2026-09-15T18:22:32Z",
    "coverage_pass_by_rank": [
      true,
      true
    ],
    "gradient_norm_by_rank": [
      1.0622400045394897,
      1.0622400045394897
    ],
    "gradient_tensors_by_rank": [
      856,
      856
    ],
    "passed": true
  },
  "image_loss": 0.04916076362133026,
  "joint_ref_mse": 0.0008083865395747125,
  "loss": 0.2870669960975647,
  "mask_loss": 0.23790624737739563,
  "micro_batch": 2098,
  "micro_batches_per_epoch": 4051,
  "optimizer_step": 10200,
  "peak_memory_bytes": 18007270400,
  "pid": 702761,
  "prefix": 1,
  "q_mean": 0.1888730525970459,
  "q_zero_fraction": 0.0233001708984375,
  "rank": 0,
  "sampling_step": 1,
  "stage": "MAIN_TRAINING",
  "updated_at": "2026-09-15T18:22:32Z"
}
```

### Test-selected best

Epoch: 0。
All8 macro Pixel-F1: 0.20318136。

| Dataset | Images | Pixel-F1 | IoU | Boundary-F1 |
| --- | ---: | ---: | ---: | ---: |
| Casiav1 | 920 | 0.15333971 | 0.088327892 | 0.12805718 |
| Columbia | 180 | 0.41030271 | 0.26502152 | 0.36023989 |
| DSO-1 | 100 | 0.23764159 | 0.14071884 | 0.13765774 |
| IMD2020 | 2010 | 0.13125169 | 0.075916097 | 0.064087755 |
| Korus | 220 | 0.10305755 | 0.05716893 | 0.058080212 |
| NIST16 | 564 | 0.12058792 | 0.073665316 | 0.046901143 |
| coverage | 100 | 0.19744292 | 0.11273626 | 0.016818541 |
| wild | 201 | 0.27182683 | 0.17122139 | 0.095730423 |

Server checkpoint: `/data1/hl/DcDsDiff-and-GIT10K/runs/TECT-DIFF-FULL-R512-S42-REFNORM-V2-MEMB6-20260915-E/best.pth`
SHA256: `NA`。权重仅保存在服务器。

### Fixed final endpoint

尚无此终点的完整 checkpoint / 八集结果。

## Phase receipts and costs

```json
{
  "reference": {
    "architecture_version": "tect-reference-pixel-unet-v2-stable-paths",
    "artifact_path": "/data1/hl/DcDsDiff-and-GIT10K/artifacts/reference/TECT-DIFF-FULL-R512-S42-REFNORM-V2-PERF-20260915-C/reference_final.pth",
    "completed_at": "2026-09-15T10:43:15Z",
    "config_hash": "63e53b3e50df4d4798d9310cb4ca2065efee1fbaea4fd2ba485fb88a37116275",
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
    "protocol_id": "TECT-DIFF-FULL-R512-S42-REFNORM-V2-MEMB6",
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
  "artifact_continuation": {},
  "batch_restart": {
    "artifact_bytes_and_metadata_unchanged": true,
    "at": "2026-09-15T14:10:24Z",
    "calibration_refitted": false,
    "calibration_sha256": "bb349a143385b25798fe0a9c7c43904ba08c26f38ed7a6aee991add7ed16f59c",
    "child_receipt_sha256": {
      "calibration_receipt.json": "d12117a50feff7896bfc4e8f5189a9d791ee3ef6da5e66a47170b294892673ce",
      "data_bundle.json": "157cdc70e1a317203a08f02a35b075c2ec7de277de70b71febc21997b6045186",
      "reference_final_health.json": "c1600fc6eda466b211629d60c2596cb107755719384bdcd90b3184099a1cbf9a",
      "reference_health.json": "36f3010c2377b87715be27e14670cffd2139ae18d0ccfc704c3c12f96207ddda",
      "reference_metrics.jsonl": "d69517a38211e9edb29cc410bb3ffbde01aa60bc7c1846f401e9cb5eaed0ff15",
      "reference_receipt.json": "031202aaee1b164bff5a0dbe9191ccc93f01a83b6e43b7e81af1befd93416b0a"
    },
    "config_delta": {
      "evaluation.metric_queue_limit": {
        "new": 8,
        "old": null
      },
      "evaluation.metric_workers": {
        "new": 4,
        "old": null
      },
      "evaluation.micro_batch": {
        "new": 8,
        "old": 1
      },
      "protocol_id": {
        "new": "TECT-DIFF-FULL-R512-S42-REFNORM-V2-MEMB6",
        "old": "TECT-DIFF-FULL-R512-S42-REFNORM-V2"
      },
      "training.accumulation_steps": {
        "new": 1,
        "old": 2
      },
      "training.global_batch": {
        "new": 12,
        "old": 8
      },
      "training.micro_batch": {
        "new": 6,
        "old": 2
      }
    },
    "config_hash": "63e53b3e50df4d4798d9310cb4ca2065efee1fbaea4fd2ba485fb88a37116275",
    "evaluation_numerical_policy": {
      "budget_basis": "predeclared empirical guard, not analytical end-to-end bound",
      "difference_scope": "Cross-batch binary and metric differences are disclosed, not required to be zero; CPU metrics on identical arrays remain exact",
      "fp32_control_budget": 1e-05,
      "policy_id": "BF16_BATCH_EXECUTION_V1",
      "probability_budget": 0.0078125
    },
    "evaluation_scheduling": {
      "metric_queue_limit": 8,
      "metric_workers": 4,
      "micro_batch": 8
    },
    "inherited_artifact_continuation": {
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
    "inherited_artifact_continuation_sha256": "a894f0943b03a2dd101174f6431e2c6976e3ccb6eacdebd9f00ab459b929e4d8",
    "main_batch": {
      "accumulation_steps": 1,
      "global_batch": 12,
      "micro_batch_per_rank": 6
    },
    "main_initialization": "fresh registered ImageNet weights",
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
    "mode": "authorized_microbatch_protocol_restart_fresh_main_reuse_fitting",
    "original_receipts": {
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
      }
    },
    "parent_config_hash": "0b0358aa4908b852bb53113467184483fbb11d4be517eb58548ba80757052f88",
    "parent_main_state_reused": false,
    "parent_protocol_id": "TECT-DIFF-FULL-R512-S42-REFNORM-V2",
    "parent_receipt_sha256": {
      "calibration_receipt.json": "89562ed608254b9d99412b5b914b9a5266d2442cca6f9949d26d4f351829a99b",
      "data_bundle.json": "157cdc70e1a317203a08f02a35b075c2ec7de277de70b71febc21997b6045186",
      "reference_final_health.json": "c1600fc6eda466b211629d60c2596cb107755719384bdcd90b3184099a1cbf9a",
      "reference_health.json": "36f3010c2377b87715be27e14670cffd2139ae18d0ccfc704c3c12f96207ddda",
      "reference_metrics.jsonl": "d69517a38211e9edb29cc410bb3ffbde01aa60bc7c1846f401e9cb5eaed0ff15",
      "reference_receipt.json": "fdf3793d063d5a529a470307f4bb10ccca1d89797c04be9a0de21e503b303ff1"
    },
    "parent_run": "TECT-DIFF-FULL-R512-S42-REFNORM-V2-AMPFIX-20260915-D",
    "parent_source_commit": "84fcf47e5721317d99fe186feec08d365fc3f72c",
    "protocol_id": "TECT-DIFF-FULL-R512-S42-REFNORM-V2-MEMB6",
    "rebound_receipt_fields": {
      "calibration_receipt.json": [
        "config_hash",
        "protocol_id"
      ],
      "reference_receipt.json": [
        "config_hash",
        "protocol_id"
      ]
    },
    "reference_epochs_retrained": 0,
    "reference_sha256": "800f50c392a275fde3ec249c6f52275c6db79813e6f8b7e8b534f9ddb52a523f",
    "selection_protocol": "test_selected",
    "source_commit": "ad8fb6e5f3f6f295b1e94b87e28175a5fabc779c",
    "status": "COMPLETED",
    "training_validation_scope": {
      "all_rank_gradients_synchronized": true,
      "final_parameter_and_optimizer_hashes_agree": true,
      "full_gradient_coverage_all_updates": true,
      "learning_rate_scaling": "none",
      "measured_updates": 32,
      "warmup_updates": 4
    },
    "validation": {
      "accepted_geometry_variant": "geometry_full_queries",
      "accumulation_steps": 1,
      "all_rank_gradients_synchronized": true,
      "at": "2026-09-15T14:07:11.294618+00:00",
      "calibration_sha256": "bb349a143385b25798fe0a9c7c43904ba08c26f38ed7a6aee991add7ed16f59c",
      "candidate_evaluation_pipeline_sha256": "602826a22fcecd49d5d7e2defdd753d6d454edcbd7dcc1f40cf4c362992329c4",
      "candidate_model_sha256": "11cdc9079d41693951f504a2831eb675ef891f381b8cf0a6b176f9de140f0781",
      "candidate_worker_sha256": "f05cc607c0c57eeeaf2933c56da54f81afc34b10284fb073b44db0caed143dee",
      "config_hash": "63e53b3e50df4d4798d9310cb4ca2065efee1fbaea4fd2ba485fb88a37116275",
      "cross_run_bitwise_training_equivalence": false,
      "evaluation_memory_headroom": {
        "context_scratch_allowance_bytes": 2147483648,
        "estimated_total_bytes": 9249661280,
        "limitation": "Conservative arithmetic allowance, not a measured formal evaluation peak.",
        "optimizer_and_bucket_allowance_multiplier": 2,
        "optimizer_state_bytes_per_rank": 530484656,
        "passed": true,
        "physical_gpu_bytes": 25757220864,
        "selected_evaluation_peak_bytes": 6041208320
      },
      "evaluation_validation": {
        "binary_disagreement_count": 2520,
        "finite": true,
        "fp32_batch_control_max_abs": 4.172325134277344e-07,
        "fp32_batch_control_passed": true,
        "full_id_coverage": true,
        "identical_array_cpu_metrics_exact": true,
        "max_boundary_f1_abs_difference": 0.003700950795189284,
        "max_iou_abs_difference": 3.45228609227588e-05,
        "max_mae_abs_difference": 4.6361632272673425e-05,
        "max_pixel_f1_abs_difference": 3.343034862968164e-05,
        "max_probability_abs_error": 0.007109403610229492,
        "metric_queue_limit": 8,
        "metric_workers": 4,
        "micro_batch": 8,
        "model_rng_unchanged": true,
        "numerical_policy": "BF16_BATCH_EXECUTION_V1",
        "passed": true,
        "unique_valid_pixels": 33508878
      },
      "evidence_forward_validation": {
        "all_global_ids_exact": true,
        "candidate_evidence_sha256": "11cdc9079d41693951f504a2831eb675ef891f381b8cf0a6b176f9de140f0781",
        "forward_original_repeat_bitwise": true,
        "global_count": 16,
        "limits": "Forward-only evidence isolation; no full training/backward equivalence or performance conclusion.",
        "reference_calibration_rng_unchanged": true,
        "scientific_result": false,
        "status": "COMPLETED",
        "variants": {
          "geometry_full_queries": {
            "all_forward_outputs_bitwise_equal": true
          },
          "geometry_only": {
            "all_forward_outputs_bitwise_equal": true
          }
        }
      },
      "evidence_forward_validation_path": "/data1/hl/DcDsDiff-and-GIT10K/runtime/tect-resource-recheck-20260915/evidence-forward-proof/aggregate.json",
      "evidence_forward_validation_sha256": "28849319d020f8fb5b112b20d2ca2a8588e6f7e55850d90a80352743943ccfda",
      "full_gradient_coverage": true,
      "global_batch": 12,
      "measured_speedup": 1.589831301767125,
      "measured_speedup_vs_micro4": 1.1067724531912888,
      "measurement_receipts": {
        "bf16-throughput/aggregate.json": {
          "server_path": "/data1/hl/DcDsDiff-and-GIT10K/runtime/tect-resource-recheck-20260915/bf16-throughput/aggregate.json",
          "sha256": "f3a48b51c19e375d5bf2aaa04bb8db6f1f80c6e3db2f183b1503ec197fef779b"
        },
        "cpu_validation_final.json": {
          "server_path": "/data1/hl/DcDsDiff-and-GIT10K/runtime/tect-resource-recheck-20260915/cpu_validation_final.json",
          "sha256": "0a37533b0b3b92c4cacb79cbd9bae096ffaeece0f4b5600a9dfbae4461cf3c49"
        },
        "equivalence/aggregate.json": {
          "server_path": "/data1/hl/DcDsDiff-and-GIT10K/runtime/tect-resource-recheck-20260915/equivalence/aggregate.json",
          "sha256": "5819f9d16a9d833c06c54304c71fa8f75140cf08b9daccbbf46d0f4c1695bd05"
        },
        "evaluation/aggregate.json": {
          "server_path": "/data1/hl/DcDsDiff-and-GIT10K/runtime/tect-resource-recheck-20260915/evaluation/aggregate.json",
          "sha256": "ec15f4fbc4cc305d1f6c1dfa33a031e60bb1095777e426c10ecabb14c3fb4886"
        },
        "fp32-control/aggregate.json": {
          "server_path": "/data1/hl/DcDsDiff-and-GIT10K/runtime/tect-resource-recheck-20260915/fp32-control/aggregate.json",
          "sha256": "b3500f4a5387a9f67b8b9399f0a5514dd42369f00ce7be693554dcba09cca707"
        },
        "micro4/aggregate.json": {
          "server_path": "/data1/hl/DcDsDiff-and-GIT10K/runtime/tect-resource-recheck-20260915/micro4/aggregate.json",
          "sha256": "7b61ad79d41eeddf33a9ff2a2769fd652de27e6158c2dadf4a89184cd515f219"
        },
        "micro6/aggregate.json": {
          "server_path": "/data1/hl/DcDsDiff-and-GIT10K/runtime/tect-resource-recheck-20260915/micro6/aggregate.json",
          "sha256": "f6b9be8ed0ab953df53078d512358b14e86272e946550a91900bf92baaeeda82"
        }
      },
      "micro6_no_oom": true,
      "micro_batch": 6,
      "parent_config_hash": "0b0358aa4908b852bb53113467184483fbb11d4be517eb58548ba80757052f88",
      "parent_run": "TECT-DIFF-FULL-R512-S42-REFNORM-V2-AMPFIX-20260915-D",
      "parent_source_commit": "84fcf47e5721317d99fe186feec08d365fc3f72c",
      "passed": true,
      "probe_finite": true,
      "reference_sha256": "800f50c392a275fde3ec249c6f52275c6db79813e6f8b7e8b534f9ddb52a523f",
      "scientific_result": false,
      "training_benchmark_path": "/data1/hl/DcDsDiff-and-GIT10K/runtime/tect-resource-recheck-20260915/micro6/aggregate.json",
      "training_benchmark_sha256": "f6b9be8ed0ab953df53078d512358b14e86272e946550a91900bf92baaeeda82"
    },
    "validation_source_path": "/data1/hl/DcDsDiff-and-GIT10K/runtime/tect-resource-recheck-20260915/batch_validation.json",
    "validation_source_sha256": "27d7c201f25bb44d06686d7d0732c962b060cbd186c5f69b914d6ca8396843d3"
  },
  "operational_hold": {
    "active": true,
    "at": "2026-09-15T18:22:35Z",
    "authority": "user authorized stop on confirmed training abnormality",
    "reason": "EARLY_MONITOR_CONFIRMED_FOREGROUND_SATURATION: epochs0/1 predicted-positive fractions 0.9983681757525111 and 0.9999929905032854; GT 0.07785399098752006. Preserve completed states; hold for bounded training-image engineering diagnosis. No automatic resume."
  },
  "calibration": {
    "artifact_hash": "bcb2f80690dff8632a5af39c3c3263f29460aede883f93335fbf45c36b470f43",
    "artifact_path": "/data1/hl/DcDsDiff-and-GIT10K/artifacts/calibration/TECT-DIFF-FULL-R512-S42-REFNORM-V2-PERF-20260915-C/calibration.pth",
    "completed_at": "2026-09-15T10:45:25Z",
    "config_hash": "63e53b3e50df4d4798d9310cb4ca2065efee1fbaea4fd2ba485fb88a37116275",
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
    "protocol_id": "TECT-DIFF-FULL-R512-S42-REFNORM-V2-MEMB6",
    "sha256": "bb349a143385b25798fe0a9c7c43904ba08c26f38ed7a6aee991add7ed16f59c",
    "status": "COMPLETED"
  },
  "costs": {
    "preflight_seconds": 60.00239287316799
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
    "0": 18007270400,
    "1": 18005599232
  },
  "sample_profiles": [
    {
      "rank": 0,
      "cold_reference_and_measurement_seconds": 0.8895149864256382,
      "cross_call_cache": false,
      "image_count": 1,
      "mask_ten_steps_with_reference_reuse_seconds": 0.5194689482450485,
      "reference_feature_requests": 20,
      "reference_feature_reuse_hits": 12,
      "reference_forwards": 8,
      "total_seconds": 1.4089839346706867
    },
    {
      "rank": 1,
      "cold_reference_and_measurement_seconds": 0.9106642790138721,
      "cross_call_cache": false,
      "image_count": 1,
      "mask_ten_steps_with_reference_reuse_seconds": 0.5248575173318386,
      "reference_feature_requests": 20,
      "reference_feature_reuse_hits": 12,
      "reference_forwards": 8,
      "total_seconds": 1.4355217963457108
    }
  ],
  "sampled_training": {
    "records": 205,
    "malformed_lines": 0,
    "metrics": {
      "A_mean": {
        "count": 205,
        "nonfinite_count": 0,
        "mean": 0.3738248872320828,
        "minimum": 0.1832752674818039,
        "maximum": 0.7256617546081543,
        "last": 0.47452718019485474,
        "kind": "scalar"
      },
      "A_negative_fraction": {
        "count": 205,
        "nonfinite_count": 0,
        "mean": 0.1565303942415774,
        "minimum": 0.10241063684225082,
        "maximum": 0.2903359830379486,
        "last": 0.2145099639892578,
        "kind": "scalar"
      },
      "amp_skipped": {
        "count": 205,
        "nonfinite_count": 0,
        "mean": 0.0,
        "minimum": 0,
        "maximum": 0,
        "last": 0,
        "kind": "boolean_frequency"
      },
      "control_logit_change": {
        "count": 205,
        "nonfinite_count": 0,
        "mean": 0.0005901918782733336,
        "minimum": 0.0,
        "maximum": 0.001784556545317173,
        "last": 0.0,
        "kind": "scalar"
      },
      "ell_mean": {
        "count": 205,
        "nonfinite_count": 0,
        "mean": -0.0006411536156076366,
        "minimum": -0.013927406631410122,
        "maximum": 0.011944061145186424,
        "last": -0.004101177677512169,
        "kind": "scalar"
      },
      "epoch": {
        "count": 205,
        "nonfinite_count": 0,
        "mean": 0.8048780487804876,
        "minimum": 0,
        "maximum": 2,
        "last": 2,
        "kind": "scalar"
      },
      "gamma": {
        "count": 205,
        "nonfinite_count": 0,
        "mean": 0.07292054448185897,
        "minimum": 0.0,
        "maximum": 0.14641058444976807,
        "last": 0.0,
        "kind": "scalar"
      },
      "gradient_norm": {
        "count": 205,
        "nonfinite_count": 0,
        "mean": 1.918152790353065,
        "minimum": 0.07430268824100494,
        "maximum": 23.59438705444336,
        "last": 1.0622400045394897,
        "kind": "scalar"
      },
      "image_loss": {
        "count": 205,
        "nonfinite_count": 0,
        "mean": 0.09804463729989242,
        "minimum": 0.027830395847558975,
        "maximum": 0.21575239300727844,
        "last": 0.04916076362133026,
        "kind": "scalar"
      },
      "joint_ref_mse": {
        "count": 205,
        "nonfinite_count": 0,
        "mean": 0.0025618829134079367,
        "minimum": 0.000497759145218879,
        "maximum": 0.14418530464172363,
        "last": 0.0008083865395747125,
        "kind": "scalar"
      },
      "loss": {
        "count": 205,
        "nonfinite_count": 0,
        "mean": 0.648535276295208,
        "minimum": 0.12424787133932114,
        "maximum": 1.779725193977356,
        "last": 0.2870669960975647,
        "kind": "scalar"
      },
      "mask_loss": {
        "count": 205,
        "nonfinite_count": 0,
        "mean": 0.5504906365943213,
        "minimum": 0.0072935777716338634,
        "maximum": 1.5639728307724,
        "last": 0.23790624737739563,
        "kind": "scalar"
      },
      "optimizer_step": {
        "count": 205,
        "nonfinite_count": 0,
        "mean": 5100.004878048781,
        "minimum": 1,
        "maximum": 10200,
        "last": 10200,
        "kind": "scalar"
      },
      "peak_memory_bytes": {
        "count": 205,
        "nonfinite_count": 0,
        "mean": 17998434099.20001,
        "minimum": 17215086080,
        "maximum": 18007270400,
        "last": 18007270400,
        "kind": "scalar"
      },
      "prefix": {
        "count": 205,
        "nonfinite_count": 0,
        "mean": 2.7512195121951217,
        "minimum": 1,
        "maximum": 4,
        "last": 1,
        "kind": "scalar"
      },
      "q_mean": {
        "count": 205,
        "nonfinite_count": 0,
        "mean": 0.23157466918956934,
        "minimum": 0.139212965965271,
        "maximum": 0.33851635456085205,
        "last": 0.1888730525970459,
        "kind": "scalar"
      },
      "q_zero_fraction": {
        "count": 205,
        "nonfinite_count": 0,
        "mean": 0.023300170898437497,
        "minimum": 0.0233001708984375,
        "maximum": 0.0233001708984375,
        "last": 0.0233001708984375,
        "kind": "scalar"
      },
      "sampling_step": {
        "count": 205,
        "nonfinite_count": 0,
        "mean": 5.541463414634144,
        "minimum": 0,
        "maximum": 9,
        "last": 1,
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
/data0/hl/conda_envs/dcdsdiff/bin/python -s /data1/hl/DcDsDiff-and-GIT10K/runs/TECT-DIFF-FULL-R512-S42-REFNORM-V2-MEMB6-20260915-E/source/scripts/tect_diff/controller.py status --run-dir /data1/hl/DcDsDiff-and-GIT10K/runs/TECT-DIFF-FULL-R512-S42-REFNORM-V2-MEMB6-20260915-E
# Resume is blocked by the recorded user-requested operational hold.
```

Source snapshot: `/data1/hl/DcDsDiff-and-GIT10K/runs/TECT-DIFF-FULL-R512-S42-REFNORM-V2-MEMB6-20260915-E/source`; logs/status/checkpoints: `/data1/hl/DcDsDiff-and-GIT10K/runs/TECT-DIFF-FULL-R512-S42-REFNORM-V2-MEMB6-20260915-E`.
模型权重、训练原图和诊断可视化不上传 GitHub。

## Failure

Controller terminated only its registered worker process group
