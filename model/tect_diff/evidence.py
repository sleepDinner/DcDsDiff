"""Frozen, training-fitted TECT-Diff trajectory evidence.

The reference network is deliberately absent from this module. Its only response
input is Phi(epsilon_ref - epsilon_draw); neither matching nor inference accepts
a mask. Training masks are accepted only by CalibrationFitter.add(). All spatial
measurements retain the original pixel grid. Computation is explicitly FP32.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


MEASUREMENT_VERSION = "TECT-PHI9-TRAJECTORY-U2-v1"


@dataclass(frozen=True)
class EvidenceConfig:
    lambdas: tuple[float, ...] = (2.0, 3.0, 4.0, 5.0)
    replicas: int = 2
    eta: float = 1.0
    candidate_limit: int = 1024
    neighbors: int = 8
    exclusion_divisor: float = 8.0
    query_chunk: int = 2048
    invalid_border: int = 3
    content_window: int = 7
    match_temperature: float = 1.0
    content_std_floor: float = 0.001
    mu_ridge: float = 0.01
    variance_floor: float = 0.0001
    variance_shrinkage: float = 0.1
    logit_ridge: float = 0.01
    logit_iterations: int = 24
    pixels_per_class_per_image: int = 128
    max_fit_images: int = 2048
    max_logit_pixels: int = 131072
    seed: int = 42

    def __post_init__(self):
        object.__setattr__(self, "lambdas", tuple(self.lambdas))
        if self.replicas != 2 or tuple(self.lambdas) != (2.0, 3.0, 4.0, 5.0):
            raise ValueError("The registered FULL protocol requires R=2, lambda=[2,3,4,5]")
        if not 0 < self.neighbors <= self.candidate_limit <= 1024:
            raise ValueError("Invalid bounded candidate/neighbor count")
        if self.invalid_border < 3 or self.query_chunk < 1:
            raise ValueError("Measurement border must cover all fixed local kernels")
        if self.content_window != 7:
            raise ValueError("Registered fixed content mean/std window is 7x7")
        if self.max_fit_images > 2048 or self.max_fit_images < 1:
            raise ValueError("The fitting image budget must lie in [1,2048]")
        for name in ("match_temperature", "content_std_floor", "mu_ridge",
                     "variance_floor", "logit_ridge", "exclusion_divisor"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be strictly positive")
        if not 0 <= self.variance_shrinkage <= 1:
            raise ValueError("Invalid variance shrinkage")


def _fp32(device):
    return torch.autocast(device_type=torch.device(device).type, enabled=False)


def evidence_config(config: EvidenceConfig | Mapping | None = None) -> EvidenceConfig:
    """Accept the persisted full experiment config or a resolved evidence config."""
    if config is None or isinstance(config, EvidenceConfig):
        return config or EvidenceConfig()
    if "image_diffusion" not in config:
        return EvidenceConfig(**config)
    source = config["evidence"]
    names = {"knn_chunk": "query_chunk", "exclusion_radius_divisor": "exclusion_divisor",
             "ridge": "mu_ridge", "logit_l2": "logit_ridge", "logit_max_iter": "logit_iterations"}
    fields = set(EvidenceConfig.__dataclass_fields__)
    resolved = {names.get(key, key): value for key, value in source.items()
                if names.get(key, key) in fields}
    resolved.update(config["image_diffusion"])
    resolved["seed"] = config["seed"]
    resolved["max_fit_images"] = config["data"]["calibration_max_images"]
    if source.get("measurement_version") != MEASUREMENT_VERSION:
        raise ValueError("Experiment config measurement version differs from implementation")
    if source.get("ell_clip") != 5 or source.get("minimum_control_prefix") != 2:
        raise ValueError("This measurement implementation requires ell clip 5 and control prefix >=2")
    return EvidenceConfig(**resolved)


def _finite(value: Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"Non-finite {name}")


def _hash_value(value: Any, digest) -> None:
    if isinstance(value, Tensor):
        value = value.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    elif isinstance(value, Mapping):
        for key in sorted(value):
            digest.update(str(key).encode())
            _hash_value(value[key], digest)
    elif isinstance(value, (tuple, list)):
        for entry in value:
            _hash_value(entry, digest)
    else:
        digest.update(json.dumps(value, sort_keys=True, allow_nan=False).encode())


def tensor_tree_hash(value: Any) -> str:
    digest = hashlib.sha256()
    _hash_value(value, digest)
    return digest.hexdigest()


def calibration_hash(artifact: Mapping) -> str:
    return tensor_tree_hash({k: v for k, v in artifact.items() if k != "artifact_hash"})


def phi9(value: Tensor) -> Tensor:
    """RGB-major central dx/2, dy/2, and 4-neighbor Laplacian, reflect pad.

    Edge values exist for tensor convenience, but the outer 3 pixels are excluded
    from all reference candidates, fitting and control by the content context.
    """
    if value.ndim != 4 or value.shape[1] != 3 or min(value.shape[-2:]) < 3:
        raise ValueError("Phi requires Bx3xHxW RGB with H,W >= 3")
    with _fp32(value.device):
        kernel = value.new_tensor([
            [[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]],
            [[0, -0.5, 0], [0, 0, 0], [0, 0.5, 0]],
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
        ], dtype=torch.float32).unsqueeze(1).repeat(3, 1, 1, 1)
        return F.conv2d(F.pad(value.float(), (1, 1, 1, 1), mode="reflect"),
                        kernel, groups=3)


def measure(epsilon_ref: Tensor, epsilon_noise: Tensor) -> Tensor:
    if epsilon_ref.shape != epsilon_noise.shape:
        raise ValueError("Reference prediction and external noise must align")
    with torch.no_grad(), _fp32(epsilon_ref.device):
        return phi9(epsilon_ref.float() - epsilon_noise.float())


def content_descriptor(image_obs: Tensor) -> Tensor:
    """Six deterministic channels: RGB mean7, luma std7, mean3 |grad|/|lap|.

    The observed RGB input follows the image diffusion's [-1,1] convention. Only
    the descriptor maps it to [0,1]. No fitted or image-wise normalization occurs
    here; training-fitted channel center/std are applied before matching.
    """
    with torch.no_grad(), _fp32(image_obs.device):
        rgb = (image_obs.float() + 1.0) * 0.5
        local_mean = F.avg_pool2d(F.pad(rgb, (3, 3, 3, 3), mode="reflect"), 7, 1)
        luma = (rgb * rgb.new_tensor([0.299, 0.587, 0.114])[None, :, None, None]).sum(1, keepdim=True)
        mean = F.avg_pool2d(F.pad(luma, (3, 3, 3, 3), mode="reflect"), 7, 1)
        second = F.avg_pool2d(F.pad(luma.square(), (3, 3, 3, 3), mode="reflect"), 7, 1)
        std = (second - mean.square()).clamp_min(0).sqrt()
        derivatives = phi9(luma.expand(-1, 3, -1, -1))[:, :3]
        magnitude = (derivatives[:, 0:1].square() + derivatives[:, 1:2].square()).sqrt()
        lap = derivatives[:, 2:3].abs()
        texture = F.avg_pool2d(F.pad(torch.cat((magnitude, lap), 1),
                                   (1, 1, 1, 1), mode="reflect"), 3, 1)
        return torch.cat((local_mean, std, texture), 1)


def _positions(height: int, width: int, device) -> Tensor:
    yy, xx = torch.meshgrid(torch.arange(height, device=device),
                            torch.arange(width, device=device), indexing="ij")
    return torch.stack((yy, xx), -1).reshape(-1, 2).float()


def _grid_candidates(height: int, width: int, config: EvidenceConfig, device) -> Tensor:
    border = config.invalid_border
    if min(height, width) <= 2 * border:
        raise ValueError("Image too small for the registered measurement support")
    rows = min(height - 2 * border, max(1, int(math.sqrt(config.candidate_limit * height / width))))
    cols = min(width - 2 * border, config.candidate_limit // rows)
    y = torch.linspace(border, height - 1 - border, rows, device=device).round().long().unique()
    x = torch.linspace(border, width - 1 - border, cols, device=device).round().long().unique()
    return (y[:, None] * width + x[None, :]).reshape(-1)


def _edge_valid(height: int, width: int, border: int, device) -> Tensor:
    valid = torch.zeros(height, width, dtype=torch.bool, device=device)
    valid[border:height-border, border:width-border] = True
    return valid.reshape(-1)


def _knn(query_content: Tensor, candidate_content: Tensor, query_position: Tensor,
         candidate_position: Tensor, height: int, width: int, config: EvidenceConfig,
         query_valid: Tensor | None = None,
         spatial_exclusion: Tensor | None = None) -> dict[str, Tensor]:
    """Blocked content KNN; all positions are independent of labels/responses."""
    count = query_content.shape[0]
    candidate_count = candidate_content.shape[0]
    k = min(config.neighbors, candidate_count)
    if k == 0:
        raise ValueError("The fixed spatial grid contains no reference candidates")
    if spatial_exclusion is not None and (spatial_exclusion.shape != (count, candidate_count) or
            spatial_exclusion.dtype != torch.bool or spatial_exclusion.device != query_content.device):
        raise ValueError("Spatial exclusion cache does not match this query/candidate grid")
    chunks = {key: [] for key in ("indices", "weights", "match_distance", "availability")}
    radius2 = (max(height, width) / config.exclusion_divisor) ** 2
    with _fp32(query_content.device):
        candidate_norm = candidate_content.square().sum(-1)[None]
        for start in range(0, count, config.query_chunk):
            stop = min(start + config.query_chunk, count)
            qc = query_content[start:stop]
            distances = ((qc.square().sum(-1, keepdim=True) + candidate_norm -
                          2.0 * qc @ candidate_content.T) / qc.shape[-1]).clamp_min(0)
            if spatial_exclusion is None:
                spatial2 = (query_position[start:stop, None] - candidate_position[None]).square().sum(-1)
                excluded = spatial2 < radius2
            else:
                excluded = spatial_exclusion[start:stop]
            distances.masked_fill_(excluded, float("inf"))
            nearest, indices = distances.topk(k, largest=False, sorted=True)
            present = torch.isfinite(nearest)
            safe = torch.where(present, nearest, torch.zeros_like(nearest))
            # Stable unnormalized weights; missing neighbors receive exactly zero.
            minimum = nearest[:, :1]
            minimum = torch.where(torch.isfinite(minimum), minimum, torch.zeros_like(minimum))
            centered = torch.where(present, nearest - minimum, torch.zeros_like(nearest))
            weights = torch.exp(-centered / config.match_temperature) * present
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
            availability = present.sum(-1).float() / config.neighbors
            if query_valid is not None:
                availability = availability * query_valid[start:stop].float()
            chunks["indices"].append(indices)
            chunks["weights"].append(weights)
            chunks["match_distance"].append((weights * safe.sqrt()).sum(-1))
            chunks["availability"].append(availability)
    return {key: torch.cat(values) for key, values in chunks.items()}


def trajectory_vector(u: Tensor, prefix: int, config: EvidenceConfig) -> Tensor:
    """u shape ... x R x K x 9; output ... x R x (9*(2m-1))."""
    if not 1 <= prefix <= len(config.lambdas):
        raise ValueError("Invalid trajectory prefix")
    blocks = [u[..., :prefix, :].flatten(-2)]
    if prefix > 1:
        delta_lambda = u.new_tensor(config.lambdas[1:prefix]) - u.new_tensor(config.lambdas[:prefix-1])
        delta = config.eta * (u[..., 1:prefix, :] - u[..., :prefix-1, :]) / delta_lambda[:, None]
        blocks.append(delta.flatten(-2))
    return torch.cat(blocks, -1)


def cross_replica_u_stat(v: Tensor, weight: Tensor) -> Tensor:
    """Signed independent-replica U-statistic; replica axis is -2."""
    with _fp32(v.device):
        v, weight = v.float(), weight.float()
        replicas = v.shape[-2]
        if replicas < 2:
            raise ValueError("Independent replicas are required")
        if replicas == 2:
            return (v[..., 0, :] * weight * v[..., 1, :]).sum(-1)
        return ((v.sum(-2).square() - v.square().sum(-2)) * weight).sum(-1) / (replicas * (replicas - 1))


def _design(content: Tensor) -> Tensor:
    return torch.cat((torch.ones_like(content[:, :1]), content), -1)


def _joint_design(content: Tensor, signed_anomaly: Tensor) -> Tensor:
    z = signed_anomaly[:, None]
    return torch.cat((_design(content), z, z * content), -1)


def _regularized_logit(x: Tensor, y: Tensor, config: EvidenceConfig) -> tuple[Tensor, dict]:
    """Deterministic damped Newton logistic regression, same sampled distribution."""
    x, y = x.float(), y.float()
    beta = torch.zeros(x.shape[-1], dtype=torch.float32)
    penalty = torch.ones_like(beta) * config.logit_ridge
    penalty[0] = 1e-6
    n = x.shape[0]

    def objective(candidate):
        return F.binary_cross_entropy_with_logits(x @ candidate, y) + 0.5 * (penalty * candidate.square()).sum()

    completed = 0
    for iteration in range(config.logit_iterations):
        probability = torch.sigmoid(x @ beta)
        gradient = x.T @ (probability - y) / n + penalty * beta
        curvature = (probability * (1 - probability)).clamp_min(1e-5)
        hessian = x.T @ (x * curvature[:, None]) / n + torch.diag(penalty)
        direction = torch.linalg.solve(hessian, gradient)
        current = objective(beta)
        step = 1.0
        for _ in range(12):
            candidate = beta - step * direction
            if bool(objective(candidate) <= current):
                break
            step *= 0.5
        beta = candidate
        completed = iteration + 1
        if float((step * direction).abs().max()) < 1e-6:
            break
    _finite(beta, "logistic coefficients")
    return beta, {"iterations": completed, "regularized_objective": float(objective(beta)),
                  "positive_pixels": int(y.sum()), "negative_pixels": int((1-y).sum()),
                  "fitting_pixels": n, "coefficient_l2": float(beta.norm())}


class CalibrationFitter:
    """One reference pass, followed by CPU fitting on compact sampled responses.

    add() takes exactly one training image and all 2x4 Phi responses. It stores
    only up to 128 pixels per class and the <=1024 fixed candidate-grid pixels.
    The normal mu/W fit uses sampled query pixels labelled unedited; reference
    selection itself never uses those labels. No test-data API exists here.
    """
    def __init__(self, config: EvidenceConfig | Mapping | None = None):
        self.config = evidence_config(config)
        self.records: list[dict] = []
        self.ids: set[str] = set()
        self.generator = torch.Generator().manual_seed(self.config.seed)

    @torch.no_grad()
    def add(self, image_obs: Tensor, responses: Tensor, gt: Tensor,
            sample_id: str, valid: Tensor | None = None) -> dict:
        if sample_id in self.ids or len(self.records) >= self.config.max_fit_images:
            raise ValueError("Duplicate fitting ID or exceeded registered image budget")
        if image_obs.ndim != 4 or image_obs.shape[:2] != (1, 3):
            raise ValueError("Calibration collection uses one RGB training image at a time")
        height, width = image_obs.shape[-2:]
        if responses.shape != (2, 4, 1, 9, height, width):
            raise ValueError("Expected responses [R=2,K=4,B=1,9,H,W]")
        if gt.numel() != height * width:
            raise ValueError("Training annotation does not align with the input pixel grid")
        raw = content_descriptor(image_obs).squeeze(0).flatten(1).T.cpu()
        labels = gt.detach().float().reshape(-1).cpu()
        usable = _edge_valid(height, width, self.config.invalid_border, "cpu")
        if valid is not None:
            usable &= valid.detach().reshape(-1).bool().cpu()
        if not bool(((labels[usable] == 0) | (labels[usable] == 1)).all()):
            raise ValueError("Fitting requires established binary tamper labels; pass ignore pixels via valid")
        query_parts = []
        for category in (0, 1):
            pool = torch.where(usable & (labels == category))[0]
            order = torch.randperm(len(pool), generator=self.generator)
            query_parts.append(pool[order[:self.config.pixels_per_class_per_image]])
        query = torch.cat(query_parts)
        if len(query) == 0:
            raise ValueError(f"No valid fitting pixels in {sample_id}")
        candidates = _grid_candidates(height, width, self.config, "cpu")
        # Ignore labels affect fitting samples only. Image-internal references
        # remain the same unlabeled spatial grid at fit and inference time.
        response_grid = responses.detach().float().squeeze(2).flatten(-2).permute(3, 0, 1, 2)
        _finite(response_grid, "reference responses")
        positions = _positions(height, width, "cpu")
        record = {"sample_id": sample_id, "height": height, "width": width,
                  "content": raw[query], "candidate_content": raw[candidates],
                  "response": response_grid[query.to(response_grid.device)].cpu(),
                  "candidate_response": response_grid[candidates.to(response_grid.device)].cpu(),
                  "position": positions[query], "candidate_position": positions[candidates],
                  "labels": labels[query]}
        self.records.append(record)
        self.ids.add(sample_id)
        return {"images": len(self.records), "sample_id": sample_id,
                "normal_pixels": int((record["labels"] == 0).sum()),
                "tampered_pixels": int(record["labels"].sum()),
                "candidate_pixels": len(candidates)}

    @torch.no_grad()
    @torch.autocast(device_type="cpu", enabled=False)
    def finalize(self, metadata: Mapping[str, Any]) -> dict:
        required = ("training_manifest_sha256", "fit_manifest_sha256", "reference_sha256",
                    "input_resolution", "reference_trained", "train_root")
        if any(key not in metadata for key in required):
            raise ValueError(f"Fitting provenance requires {required}")
        if metadata["train_root"].rstrip("/") != "/data0/hl/FinalTrainData" or metadata["reference_trained"] is not True:
            raise ValueError("Only trained-reference responses from the authorized training root can fit FULL")
        for key in ("training_manifest_sha256", "fit_manifest_sha256", "reference_sha256"):
            if len(str(metadata[key])) != 64:
                raise ValueError(f"Missing SHA256 provenance: {key}")
        if not self.records:
            raise ValueError("An empty calibration artifact cannot enter FULL training")
        if any(record["height"] != metadata["input_resolution"] or record["width"] != metadata["input_resolution"]
               for record in self.records):
            raise ValueError("Calibration images and registered resolution differ")
        config = self.config
        raw = torch.cat([r["content"] for r in self.records])
        labels = torch.cat([r["labels"] for r in self.records])
        normal = labels == 0
        if int(normal.sum()) < 64 or int((~normal).sum()) < 64:
            raise ValueError("Calibration needs at least 64 known unedited and 64 edited training pixels")
        center = raw.mean(0)
        std = raw.std(0, unbiased=False).clamp_min(config.content_std_floor)
        content = (raw - center) / std
        x = _design(content[normal])
        response = torch.cat([r["response"] for r in self.records])[normal].mean(1)
        # mu shape K x (1+6) x 9. Only GT=0 query responses contribute.
        penalty = torch.eye(x.shape[-1]) * config.mu_ridge
        penalty[0, 0] = 1e-6
        mu = torch.linalg.solve(x.T @ x / len(x) + penalty,
                                x.T @ response.flatten(1) / len(x)).reshape(7, 4, 9).permute(1, 0, 2).contiguous()
        all_u, all_match, all_availability = [], [], []
        for record in self.records:
            c = (record["content"] - center) / std
            cc = (record["candidate_content"] - center) / std
            matching = _knn(c, cc, record["position"], record["candidate_position"],
                            record["height"], record["width"], config)
            residual = record["response"] - torch.einsum("nc,kcd->nkd", _design(c), mu)[:, None]
            candidate_residual = record["candidate_response"] - torch.einsum("nc,kcd->nkd", _design(cc), mu)[:, None]
            neighbor_response = candidate_residual[matching["indices"]]
            matched = (neighbor_response * matching["weights"][:, :, None, None, None]).sum(1)
            all_u.append(residual - matched)
            all_match.append(matching["match_distance"])
            all_availability.append(matching["availability"])
        u = torch.cat(all_u)
        match = torch.cat(all_match)
        availability = torch.cat(all_availability)
        valid_normal = normal & (availability > 0)
        if int(valid_normal.sum()) < 64:
            raise ValueError("Insufficient normal pixels with spatially valid references")
        d0 = match[availability > 0].median().clamp_min(0.001)
        buffers = {"content_center": center, "content_std": std, "mu": mu, "d0": d0}
        order = torch.randperm(len(labels), generator=self.generator)[:config.max_logit_pixels]
        # Both logit models and all prefixes use exactly this same sampled set.
        content_beta, content_receipt = _regularized_logit(_design(content[order]), labels[order], config)
        buffers["content_beta"] = content_beta
        prefix_metadata = {}
        for prefix in range(1, 5):
            v = trajectory_vector(u, prefix, config)
            normal_trajectory = v[valid_normal].reshape(-1, v.shape[-1])
            variance = normal_trajectory.var(0, unbiased=False)
            shrunk = ((1 - config.variance_shrinkage) * variance +
                      config.variance_shrinkage * variance.mean()).clamp_min(config.variance_floor)
            weight = 1.0 / (v.shape[-1] * shrunk)
            anomaly = cross_replica_u_stat(v, weight)
            noise_variation = 0.5 * ((v[:, 0] - v[:, 1]).square() * weight).sum(-1)
            scale = anomaly[valid_normal].abs().median().clamp_min(0.0001)
            v0 = noise_variation[valid_normal].median().clamp_min(0.0001)
            signed = anomaly.sign() * torch.log1p(anomaly.abs() / scale)
            joint_beta, joint_receipt = _regularized_logit(_joint_design(content[order], signed[order]), labels[order], config)
            buffers[f"W_{prefix}"] = weight
            buffers[f"anomaly_scale_{prefix}"] = scale
            buffers[f"v0_{prefix}"] = v0
            buffers[f"joint_beta_{prefix}"] = joint_beta
            evidence = (_joint_design(content, signed) @ joint_beta - _design(content) @ content_beta).clamp(-5, 5)
            q = availability * torch.exp(-match / d0) / (1 + noise_variation / v0)
            prefix_metadata[str(prefix)] = {
                "dimension": v.shape[-1], "W_normalization": "inverse shrunk variance divided by dimension",
                "W_sha256": tensor_tree_hash(weight), "control_enabled": prefix >= 2,
                "normal_trajectory_rows": len(normal_trajectory), "joint_fit": joint_receipt,
                "negative_A_fraction": float((anomaly < 0).float().mean()),
                "A_quantiles": torch.quantile(anomaly, torch.tensor([0., .25, .5, .75, 1.])).tolist(),
                "q_mean": float(q.mean()), "q_zero_fraction": float((q == 0).float().mean()),
                "ell_std": float(evidence.std(unbiased=False)), "ell_mean": float(evidence.mean()),
                "d0": float(d0), "v0": float(v0), "anomaly_scale": float(scale)}
        for key, value in buffers.items():
            _finite(value, key)
        artifact = {"format_version": MEASUREMENT_VERSION, "config": asdict(config),
                    "buffers": buffers, "metadata": dict(metadata), "prefix_metadata": prefix_metadata,
                    "fit_receipt": {"images": len(self.records), "sample_ids_sha256": tensor_tree_hash(sorted(self.ids)),
                                    "normal_pixels": int(normal.sum()), "tampered_pixels": int((~normal).sum()),
                                    "normal_statistics": "only known GT=0 training query pixels",
                                    "sampling": f"up to {config.pixels_per_class_per_image} uniform pixels per available class per image; equal pixel weight; identical joint/content sample",
                                    "content_fit": content_receipt,
                                    "semantics": "estimated content-subtracted evidence; no conditional-density guarantee"}}
        artifact["artifact_hash"] = calibration_hash(artifact)
        return artifact


class FixedTrajectoryEvidence(nn.Module):
    """Strictly loaded frozen calibration and pixel-preserving evidence evaluator."""
    def __init__(self, artifact: Mapping[str, Any], device=None):
        super().__init__()
        if artifact.get("format_version") != MEASUREMENT_VERSION:
            raise ValueError("Calibration measurement version mismatch")
        if artifact.get("artifact_hash") != calibration_hash(artifact):
            raise ValueError("Calibration artifact integrity check failed")
        self.config = EvidenceConfig(**artifact["config"])
        self.artifact_hash = artifact["artifact_hash"]
        self.metadata = dict(artifact["metadata"])
        if (self.metadata.get("reference_trained") is not True or
                self.metadata.get("train_root", "").rstrip("/") != "/data0/hl/FinalTrainData"):
            raise ValueError("FULL calibration must come from trained-reference measurements of training data")
        receipt = artifact.get("fit_receipt", {})
        if (receipt.get("images", 0) <= 0 or receipt.get("normal_pixels", 0) < 64 or
                receipt.get("tampered_pixels", 0) < 64 or set(artifact.get("prefix_metadata", {})) != {"1", "2", "3", "4"}):
            raise ValueError("Incomplete fitting receipt or missing calibrated trajectory prefixes")
        required_shapes = {"content_center": (6,), "content_std": (6,), "mu": (4, 7, 9),
                           "d0": (), "content_beta": (7,)}
        for prefix in range(1, 5):
            required_shapes.update({f"W_{prefix}": (9 * (2 * prefix - 1),),
                                    f"anomaly_scale_{prefix}": (), f"v0_{prefix}": (),
                                    f"joint_beta_{prefix}": (14,)})
        if set(artifact["buffers"]) != set(required_shapes):
            raise ValueError("Calibration buffers are incomplete or unexpected")
        for key, shape in required_shapes.items():
            value = artifact["buffers"][key]
            if not isinstance(value, Tensor) or tuple(value.shape) != shape:
                raise ValueError(f"Invalid calibration tensor {key}")
            _finite(value, key)
            if (key.startswith(("W_", "v0_", "anomaly_scale_")) or key in ("d0", "content_std")) and not bool((value > 0).all()):
                raise ValueError(f"Calibration scale/precision {key} must be positive")
            self.register_buffer(key, value.detach().clone().float().to(device=device))
        self._frozen_hash = tensor_tree_hash(self.state_dict())
        # Derived geometry only: no fitted values, RNG, or persistent buffers.
        self._geometry_cache = None
        self.train(False)

    def train(self, mode: bool = True):
        return super().train(False)

    def assert_frozen(self) -> str:
        current = tensor_tree_hash(self.state_dict())
        if current != self._frozen_hash or self.training or any(p.requires_grad for p in self.parameters()):
            raise RuntimeError("Frozen trajectory calibration changed during main training")
        return current

    def _apply(self, fn, recurse=True):
        # A module device/dtype move must not retain tensors on its old device.
        self._geometry_cache = None
        return super()._apply(fn, recurse=recurse)

    @torch.no_grad()
    def _geometry(self, height: int, width: int, device) -> dict:
        device = torch.device(device)
        key = (height, width, device, self.config.candidate_limit, self.config.invalid_border,
               self.config.exclusion_divisor, self.config.query_chunk)
        if self._geometry_cache is not None and self._geometry_cache[0] == key:
            return self._geometry_cache[1]
        # Retain only one geometry. At 512 and 1024 candidates the boolean mask
        # occupies 256 MiB, replacing repeated coordinate arithmetic per image.
        self._geometry_cache = None
        with _fp32(device):
            candidates = _grid_candidates(height, width, self.config, device)
            positions = _positions(height, width, device)
            edge = _edge_valid(height, width, self.config.invalid_border, device)
            candidate_positions = positions[candidates]
            excluded = torch.empty((height * width, len(candidates)), dtype=torch.bool, device=device)
            radius2 = (max(height, width) / self.config.exclusion_divisor) ** 2
            for start in range(0, height * width, self.config.query_chunk):
                stop = min(start + self.config.query_chunk, height * width)
                spatial2 = (positions[start:stop, None] - candidate_positions[None]).square().sum(-1)
                excluded[start:stop] = spatial2 < radius2
        geometry = {"candidates": candidates, "positions": positions, "edge": edge,
                    "spatial_exclusion": excluded}
        self._geometry_cache = (key, geometry)
        return geometry

    @staticmethod
    def measure(epsilon_ref: Tensor, epsilon_noise: Tensor) -> Tensor:
        return measure(epsilon_ref, epsilon_noise)

    @torch.no_grad()
    def prepare(self, image_obs: Tensor) -> dict:
        batch, _, height, width = image_obs.shape
        if height != self.metadata["input_resolution"] or width != self.metadata["input_resolution"]:
            raise ValueError("Resolution differs from the frozen calibration artifact")
        with _fp32(image_obs.device):
            raw = content_descriptor(image_obs)
            content = (raw - self.content_center[None, :, None, None]) / self.content_std[None, :, None, None]
            geometry = self._geometry(height, width, image_obs.device)
            candidates, positions, edge = (geometry[key] for key in ("candidates", "positions", "edge"))
            matches = []
            for b in range(batch):
                flat = content[b].flatten(1).T
                matches.append(_knn(flat, flat[candidates], positions, positions[candidates],
                                    height, width, self.config, edge, geometry["spatial_exclusion"]))
            context = {key: torch.stack([entry[key] for entry in matches]) for key in matches[0]}
            context.update({"content": content, "candidates": candidates,
                            "height": height, "width": width, "batch": batch,
                            "valid": edge.reshape(1, 1, height, width).expand(batch, -1, -1, -1),
                            "artifact_hash": self.artifact_hash})
            return context

    @torch.no_grad()
    def _relative_response(self, context: Mapping, d_pair: Tensor, scale: int) -> Tensor:
        batch, height, width = context["batch"], context["height"], context["width"]
        if d_pair.shape != (2, batch, 9, height, width):
            raise ValueError("A streamed scale requires responses [2,B,9,H,W]")
        with _fp32(d_pair.device):
            c = context["content"]
            design = torch.cat((torch.ones_like(c[:, :1]), c), 1)
            mean = torch.einsum("bchw,cd->bdhw", design, self.mu[scale])
            residual = d_pair.float() - mean[None]
            result = residual.clone()
            for replica in range(2):
                for b in range(batch):
                    flat = residual[replica, b].flatten(1).T
                    candidate_values = flat[context["candidates"]]
                    target = result[replica, b].flatten(1).T
                    # Queries are independent. Keep the same per-query neighbor
                    # reduction, trading temporary memory for fewer launches.
                    indices = context["indices"][b]
                    weights = context["weights"][b]
                    target -= (candidate_values[indices] * weights[..., None]).sum(1)
            return result

    def begin(self, context: Mapping) -> dict:
        if context["artifact_hash"] != self.artifact_hash:
            raise ValueError("Evidence context belongs to another calibration")
        return {"context": context, "u": []}

    @torch.no_grad()
    def append(self, state: dict, d_pair: Tensor) -> dict[str, Any]:
        """Append one scale in lambda order and return its matched prefix evidence."""
        prefix = len(state["u"]) + 1
        if prefix > 4:
            raise ValueError("Cannot append future scales beyond the registered K=4")
        state["u"].append(self._relative_response(state["context"], d_pair, prefix - 1))
        return self._evaluate_u(state["context"], state["u"])

    @torch.no_grad()
    def prefix(self, context: Mapping, responses: Tensor | Sequence[Tensor]) -> dict[str, Any]:
        """responses Tensor [2,m,B,9,H,W] or list of m tensors [2,B,9,H,W]."""
        state = self.begin(context)
        if isinstance(responses, Tensor):
            if responses.ndim != 6 or responses.shape[0] != 2:
                raise ValueError("Expected responses [2,m,B,9,H,W]")
            responses = responses.unbind(1)
        if not 1 <= len(responses) <= 4:
            raise ValueError("Expected 1..4 visible image scales")
        for scale, pair in enumerate(responses):
            state["u"].append(self._relative_response(context, pair, scale))
        return self._evaluate_u(context, state["u"])

    def _evaluate_u(self, context: Mapping, u: Sequence[Tensor]) -> dict[str, Any]:
        prefix = len(u)
        with _fp32(u[0].device):
            weight = getattr(self, f"W_{prefix}")
            anomaly = torch.zeros_like(u[0][0, :, :1])
            noise_variation = torch.zeros_like(anomaly)

            def add_block(block, offset):
                diagonal = weight[offset:offset+9][None, :, None, None]
                anomaly.add_((block[0] * diagonal * block[1]).sum(1, keepdim=True))
                noise_variation.add_(0.5 * ((block[0] - block[1]).square() * diagonal).sum(1, keepdim=True))

            for k in range(prefix):
                add_block(u[k], k * 9)
            for k in range(prefix - 1):
                delta = self.config.eta * (u[k+1] - u[k]) / (self.config.lambdas[k+1] - self.config.lambdas[k])
                add_block(delta, (prefix + k) * 9)
            scale = getattr(self, f"anomaly_scale_{prefix}")
            signed = anomaly.sign() * torch.log1p(anomaly.abs() / scale)
            content = context["content"]
            beta = getattr(self, f"joint_beta_{prefix}")
            prior = self.content_beta
            # Evaluates exactly [1,c,z,z*c] dot beta - [1,c] dot prior,
            # without allocating a full Bx14xHxW design tensor.
            ell = (beta[0] - prior[0] +
                   ((beta[1:7] - prior[1:7])[None, :, None, None] * content).sum(1, keepdim=True) +
                   signed * (beta[7] + (beta[8:14][None, :, None, None] * content).sum(1, keepdim=True))).clamp(-5, 5)
            shape = (context["batch"], 1, context["height"], context["width"])
            match = context["match_distance"].reshape(shape)
            availability = context["availability"].reshape(shape)
            q = availability * torch.exp(-match / self.d0) / (1 + noise_variation / getattr(self, f"v0_{prefix}"))
            return {"A": anomaly, "q": q, "ell": ell, "V_noise": noise_variation,
                    "match_distance": match, "availability": availability,
                    "prefix": prefix, "control_enabled": prefix >= 2,
                    "control_evidence": q * ell if prefix >= 2 else torch.zeros_like(ell)}


def save_calibration(artifact: Mapping, path: str | Path) -> str:
    """Atomically save and independently reload all calibration parameters."""
    if artifact.get("artifact_hash") != calibration_hash(artifact):
        raise ValueError("Cannot save an invalid calibration artifact")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    torch.save(dict(artifact), temporary)
    reloaded = torch.load(temporary, map_location="cpu", weights_only=True)
    FixedTrajectoryEvidence(reloaded).assert_frozen()
    os.replace(temporary, path)
    return artifact["artifact_hash"]


def load_calibration(path: str | Path, device=None, expected_hash: str | None = None) -> FixedTrajectoryEvidence:
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if expected_hash is not None and artifact.get("artifact_hash") != expected_hash:
        raise ValueError("Calibration identity differs from the registered artifact")
    return FixedTrajectoryEvidence(artifact, device=device)
