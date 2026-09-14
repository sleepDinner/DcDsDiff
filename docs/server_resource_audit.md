# Server resource audit — 2026-09-14

This inventory was obtained through read-only SSH queries on `imdl-server` (`amax`).
The old project, data, and environments were not modified during the audit.

| Resource | Observed state |
| --- | --- |
| Old checkout | `/data0/hl/DcDsDiff-and-GIT10K` |
| Old HEAD | `d60e4e893ad391a699fe5229a4be0903cfc3cbf3` |
| Old origin | `git@github.com:sleepDinner/DcDsDiff.git` |
| Old git status | Untracked `GIT10K/`, `eval_best/`, `logs/`, `nohup.out`; tracked tree clean |
| New checkout target | `/data1/hl/DcDsDiff-and-GIT10K`, absent before this task |
| New environment target | `/data0/hl/conda_envs/dcdsdiff`, absent before this task |
| Conda executable | `/opt/anaconda3/bin/conda` |
| Historical runtime | `/data0/hl/conda_envs/lzb/bin/python`, Python 3.10.20 |
| GPU | 2 × NVIDIA GeForce RTX 4090, 24,564 MiB each |
| GPU state at audit | Both 13 MiB used, utilization 0%; no user Python/torchrun training processes observed |
| NVIDIA driver | 535.54.03 |
| `/data0` free | 389 GiB displayed by `df -h`, 95% used |
| `/data1` free | 4.3 TiB displayed by `df -h`, 39% used |

GPU availability and free disk space are snapshots; execution must check current state.

## Reusable data and weights

`/data0/hl/DcDsDiff-and-GIT10K/GIT10K/` occupies 4.3 GiB and contains only `Image/` and `Mask/` directories. Each has 10,000 PNG files; all image and mask filename stems match. No split manifest or auxiliary-label directory was found inside this dataset directory. Reuse the original resource by reference rather than copying or editing it.

Image filename-prefix counts are BN=251, EI=2225, Flux=4, IA=2475, PP=2274, RBN=2322, e=200, t=50, z=199. These are observed naming groups, not independently verified semantic classes.

Eight evenly spaced filename samples were opened. Some masks contain intermediate gray values rather than only 0/255. Sample `IA (1016)` has an image of 368×496 pixels and a mask of 375×500 pixels. A complete integrity audit and an explicit geometry policy are necessary before preparing splits; silently dropping mismatched pairs would change the population. The eight-image audit is not a full image-decoding validation.

Reusable PVT-B2 initialization:

```text
/data0/hl/DcDsDiff-and-GIT10K/pretrained_weights/models--Anonymity--pvt_pretrained/snapshots/a11fd1f27a892002dde9a2050e423ce6c3491bb4/pvt_v2_b2.pth
sha256 80711cd1b37ffba12bec6c7a2a7c54efe2315ed635e6d2055fd51c3e909ede4d
```

The weight is approximately 97 MiB. The checksum records local provenance; it does not independently authenticate the upstream artifact.

## Historical runtime and experiment boundary

Historical logs show training against `/data0/hl/Diff_dataset/...` and a separate configuration for `/data0/hl/FinalTrainData_Diff/...`. They do not establish a completed GIT10K paper reproduction. An old `nohup.out` ends in an interrupted launch from the `lzb` environment. Existing predictions/logs must not be adopted as the new run's results.

Key historical package versions were obtained with `importlib.metadata`, without importing Torch or running a GPU operation: torch 2.1.2+cu121, torchvision 0.16.2+cu121, numpy 1.26.4, mmcv-lite 2.2.0, mmengine 0.10.7, timm 0.9.12, accelerate 1.13.0, einops 0.8.2, albumentations 1.3.1, Pillow 10.2.0, scipy 1.11.4, scikit-image 0.22.0, scikit-learn 1.3.2, omegaconf 2.3.0, wandb 0.27.2, numba 0.58.1, ema-pytorch 0.7.9, pytorch-fid 0.3.0, matplotlib 3.10.9, huggingface-hub 1.15.0, PyYAML 6.0.1, tqdm 4.66.1. The old environment contains overlapping opencv-python 4.9.0.80 and opencv-python-headless 4.8.1.78. Installed package metadata confirms MMCV/MMEngine require the former and Albumentations requires the latter; the new environment pins both to 4.9.0.80 and does not use GUI operations.

## Environment installation policy

Create a fresh Conda Python 3.10 environment at the requested prefix. Install CUDA 12.1 Torch wheels from the [official PyTorch wheel index](https://pytorch.org/get-started/previous-versions/), then the pinned direct dependencies in `environment/requirements.txt`. Only `mmcv.cnn.ConvModule` is imported by the model; [MMCV documents the lightweight distribution](https://mmcv.readthedocs.io/en/latest/get_started/installation.html), and the historical environment already used mmcv-lite.

Use project-owned `runtime/bootstrap/` for logs and installation receipts and `runtime/cache/` for Conda/Pip downloads. Record the final full pip freeze and `pip check` before considering the environment ready. Do not clone or change the old lzb environment.

`bash environment/create.sh` creates the environment when absent. If the prefix already exists, it verifies the READY receipt, direct-requirement hash, dependency-lock hash, live package-freeze hash, and `pip check`; it refuses to overwrite or silently repair an unverified existing prefix. The installer locks `runtime/bootstrap/environment.lock` and records completion/exit status. Formal training must require the JSON READY receipt rather than infer readiness from the Python executable merely existing.

The first download attempt was deliberately stopped after a complete cached PyTorch wheel was found. The cp310 Linux wheel was independently matched against the official CUDA 12.1 package index: SHA-256 `b2184b7729ef3b9b10065c074a37c1e603fd99f91e38376e25cb7ed6e1d54696`. Twenty additional distinct package wheels were found in the existing pip download cache. Their file hashes were matched against PyPI release metadata or the official PyTorch index before copying them into the new project cache. This reuses downloaded installers and still installs a fresh environment; no files were copied from the old environment's `site-packages`. The machine-readable sources and hashes are recorded in `runtime/bootstrap/torch-wheel-provenance.json` and `runtime/bootstrap/wheel-cache-provenance.json`. Recovery actions target only the installer process groups created by this task, with interruption receipts retained in the bootstrap directory.

## Completed environment verification

At 2026-09-14 11:03:08 UTC (19:03:08 Asia/Shanghai), the fresh environment passed checks for every pinned direct package, imports of Torch/TorchVision/Accelerate/Albumentations/OpenCV/NumPy/OmegaConf/MMCV ConvModule, CUDA availability, and `pip check` (`No broken requirements found.`). Python is 3.10.21; Torch is 2.1.2+cu121; TorchVision is 0.16.2+cu121; the CUDA runtime is 12.1 and two CUDA devices are visible. These checks establish environment readiness, not model-level correctness or paper-level reproduction.

The final JSON receipt is `runtime/bootstrap/environment.ready`, with direct-requirement SHA-256 `e2c4e5b58dca86f09f66aa5efafb0716f9dc9f8e63c46c213a3a72edb6637eb5`, dependency-lock SHA-256 `b846c703ddebceffa2efece65c08501b3656a1f2af78ef51e747c1ed42ce0300`, and observed pip-freeze SHA-256 `97c473a85a8cbbafe7047cab2d8e3b81a60f4026f1b60226ddbe0ecee714415f`. The portable version lock is `environment/requirements.lock.txt`; raw `pip freeze`, `pip check`, Conda explicit spec, recovery receipts, and installation logs remain in the server's project-local bootstrap directory. No test packages were added.
