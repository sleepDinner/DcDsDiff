"""Read-only source datasets, incremental integrity audit, and frozen fit roles.

All source image/mask paths are read-only. Only project/cache/tect_diff is written.
Fit roles are training subsets, not validation sets; every retained row participates
in localization training. Exact RGB overlaps are excluded from all training roles.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

AUDIT_VERSION = "TECT-DIFF-DATA-V1-RGB8-LABEL128"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
ALL8_NAMES = ("Casiav1", "Columbia", "NIST16", "IMD2020", "DSO-1", "wild", "coverage", "Korus")


class CoordinateMismatch(ValueError):
    """A paired training image cannot supply valid original-coordinate supervision."""


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _writable_directory(project, relative):
    project = Path(project).resolve(strict=True)
    destination = (project / relative).resolve()
    if not destination.is_relative_to(project) or destination == project:
        raise ValueError("Writable cache escapes project root")
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / ".tect_diff_created.json"
    if marker.exists():
        if json.loads(marker.read_text())["task"] != "TECT-Diff":
            raise ValueError("Cache belongs to another task")
    else:
        atomic_json(marker, {"task": "TECT-Diff", "path": str(destination), "purpose": "incremental data audit"})
    return destination


def index_files(root, suffix=""):
    """Relative-path stems preserve nested folders and reject ambiguous duplicates."""
    root = Path(root).resolve(strict=True)
    result = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        relative = path.relative_to(root)
        stem = relative.stem
        if suffix:
            if not stem.endswith(suffix):
                raise ValueError(f"Unexpected mask suffix: {path}")
            stem = stem[:-len(suffix)]
        key = (relative.parent / stem).as_posix()
        if key in result:
            raise ValueError(f"Ambiguous duplicate relative stem {key}: {path}")
        result[key] = str(path)
    return result


def source_metadata(stem, dataset):
    """Name-level provenance only: names cannot rule out transformed leakage."""
    name = Path(stem).name
    if dataset == "FinalTrainData":
        if name.startswith("Fan_Real_IMG_"):
            return "FantasticReality", ["FantasticReality:IMG_" + name.rsplit("_", 1)[1]], True, "explicit_Fan_Real_plus_zero_mask"
        if name.startswith("Fan_Fake_"):
            ids = re.findall(r"IMG_\d+", name)
            if len(ids) != 2:
                raise ValueError(f"Unknown FantasticReality source naming: {name}")
            return "FantasticReality", sorted(set("FantasticReality:" + x for x in ids)), False, "explicit_Fan_Fake_mask"
        if name.startswith("Au_"):
            match = re.fullmatch(r"Au_([a-zA-Z]+)_(\d+)", name)
            if match is None:
                raise ValueError(f"Unknown CASIA2 authentic source naming: {name}")
            return "CASIA2", ["CASIA2:" + match[1] + match[2]], True, "explicit_Au_plus_zero_mask"
        if name.startswith("Tp_"):
            ids = re.findall(r"(?:^|_)([a-z]{3}\d+)(?=_|$)", name)
            if not ids:
                raise ValueError(f"Unknown CASIA2 source naming: {name}")
            return "CASIA2", sorted(set("CASIA2:" + x for x in ids)), False, "explicit_Tp_mask"
        raise ValueError(f"Unconfirmed training data provenance: {name}")
    if dataset == "Casiav1":
        ids = re.findall(r"(?:^|_)([a-z]{3}\d+)(?=_|$)", name)
        return "CASIA1", sorted(set("CASIA1:" + x for x in ids)), False, "registered_All8_mask"
    return dataset, [], False, "registered_All8_mask_source_relation_unavailable"


def paired_records(spec, split):
    root = Path(spec["root"]).resolve(strict=True)
    images = index_files(root / spec.get("images", "images"))
    masks = index_files(root / spec.get("masks", "masks"), spec.get("mask_suffix", ""))
    if set(images) != set(masks):
        raise ValueError(f"{spec['name']}: image/mask pairing unresolved; missing={sorted(set(images)-set(masks))[:8]}, extra={sorted(set(masks)-set(images))[:8]}")
    if "count" in spec and len(images) != int(spec["count"]):
        raise ValueError(f"{spec['name']}: expected {spec['count']}, found {len(images)}")
    result = []
    for key in sorted(images):
        source, ids, authentic, basis = source_metadata(key, spec["name"])
        result.append({"id": spec["name"] + ":" + key, "stem": key, "dataset": spec["name"], "split": split,
                       "image_path": images[key], "mask_path": masks[key], "source_dataset": source,
                       "source_ids": ids, "authentic_declared": authentic, "label_basis": basis,
                       "mask_encoding": "unit_or_uint8_threshold128", "ignore_values": list(spec.get("ignore_values", [])),
                       "mask_decoder": "registered_All8_PIL_L_ignore_alpha" if split == "test" else "training_channel_consensus_PIL_L"})
    return result


def read_mask(record):
    """Explicit 0/1 or 8-bit positive=white semantics; no score-driven inversion."""
    with Image.open(record["mask_path"]) as handle:
        mask = np.asarray(handle.convert("L") if record.get("mask_decoder") == "registered_All8_PIL_L_ignore_alpha" else handle)
    if mask.ndim == 3:
        channels = mask[..., :3]
        # Four registered Columbia PNGs have 1-4 stray dark-red background
        # pixels (max 18/255). Accept RGB only if every channel gives exactly
        # the same fixed binary label; this is independent of model scores.
        channel_labels = channels > (0 if channels.max() <= 1 else 127)
        if not np.all(channel_labels == channel_labels[..., :1]):
            raise ValueError(f"Unconfirmed color-coded mask: {record['mask_path']}")
        mask = np.asarray(Image.fromarray(channels).convert("L"))
    if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer) and mask.dtype != np.bool_:
        raise ValueError(f"Unconfirmed mask encoding: {record['mask_path']}")
    if mask.size == 0 or mask.min() < 0 or mask.max() > 255:
        raise ValueError(f"Mask outside documented uint8 range: {record['mask_path']}")
    valid = ~np.isin(mask, record.get("ignore_values", []))
    if not valid.any():
        raise ValueError(f"ALL_IGNORED_SAMPLE: {record['id']}")
    maximum = int(mask[valid].max())
    binary = ((mask > 0) if maximum <= 1 else (mask >= 128)) & valid
    return binary, valid, mask


def _mask_diagnostics(record):
    """Record encoding independently of the fixed mask decoder and model output."""
    with Image.open(record["mask_path"]) as handle:
        result = {"mask_mode": handle.mode, "mask_non_gray_pixels": 0,
                  "mask_rgb_threshold_disagreement_pixels": 0, "mask_alpha_nonopaque_pixels": 0,
                  "mask_alpha_ignored": "A" in handle.getbands()}
        if handle.mode in ("RGB", "RGBA"):
            values = np.asarray(handle)
            rgb = values[..., :3]
            result["mask_non_gray_pixels"] = int(np.count_nonzero(rgb.max(-1) != rgb.min(-1)))
            labels = rgb > (0 if rgb.max() <= 1 else 127)
            result["mask_rgb_threshold_disagreement_pixels"] = int(np.count_nonzero(labels.max(-1) != labels.min(-1)))
            if handle.mode == "RGBA":
                result["mask_alpha_nonopaque_pixels"] = int(np.count_nonzero(values[..., 3] != 255))
    return result


def _enrich_cached(item):
    record, metadata = item
    return {"id": record["id"], "metadata": {**metadata, **_mask_diagnostics(record)}}


def _signature(record):
    stats = []
    for key in ("image_path", "mask_path"):
        path = Path(record[key])
        stat = path.stat()
        stats.append([str(path.resolve()), stat.st_size, stat.st_mtime_ns])
    contract = [AUDIT_VERSION, stats, record["ignore_values"]]
    if record["split"] == "test":
        contract.append(record["mask_decoder"])
    return canonical_hash(contract)


def _audit_one(record):
    try:
        before = _signature(record)
        with Image.open(record["image_path"]) as handle:
            rgb = np.asarray(handle.convert("RGB"))
            image_mode = handle.mode
        gt, valid, raw = read_mask(record)
        if rgb.shape[:2] != gt.shape:
            raise CoordinateMismatch(f"Unresolved image/mask coordinate mismatch: {rgb.shape[:2]} vs {gt.shape}")
        if record["authentic_declared"] and np.any(raw[valid] != 0):
            raise ValueError("Authentic source has nonzero annotation")
        digest = hashlib.sha256(str(rgb.shape).encode())
        digest.update(memoryview(rgb))
        metadata = {"image_hw": list(rgb.shape[:2]), "mask_hw": list(gt.shape), "image_mode": image_mode,
                    "rgb_pixels_sha256": digest.hexdigest(), "image_sha256": sha256_file(record["image_path"]),
                    "mask_sha256": sha256_file(record["mask_path"]), "mask_min": int(raw.min()), "mask_max": int(raw.max()),
                    "soft_mask_pixels": int(np.count_nonzero((raw != 0) & (raw != 1) & (raw != 255))),
                    "empty_mask": not bool(gt.any()), "positive_pixels": int(gt.sum()),
                    "ignored_pixels": int((~valid).sum()), "valid_pixels": int(valid.sum()),
                    "binary_unit_mask": bool(raw.max() == 1), "signature": before, **_mask_diagnostics(record)}
        if before != _signature(record):
            raise ValueError("Source changed during read-only audit")
        return {"id": record["id"], "metadata": metadata}
    except Exception as error:
        return {"id": record["id"], "error": f"{type(error).__name__}: {error}",
                "error_code": "image_mask_coordinate_mismatch" if isinstance(error, CoordinateMismatch) else "unresolved_data_semantics"}


def _audit_records(records, cache, workers, progress):
    old = {}
    cache_path = cache / "decoded_audit.jsonl"
    if cache_path.exists():
        with cache_path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                    if "metadata" in row:
                        old[row["id"]] = row["metadata"]
                except json.JSONDecodeError:
                    continue  # Only a torn final append is discarded; sources are re-audited.
    pending, metadata, enrichment = [], {}, []
    for record in records:
        cached = old.get(record["id"])
        if cached and cached.get("signature") == _signature(record):
            metadata[record["id"]] = cached
            if "mask_mode" not in cached:
                enrichment.append((record, cached))
        else:
            pending.append(record)
    if progress:
        progress({"stage": "DATA_AUDIT", "total": len(records), "cache_hits": len(metadata), "pending": len(pending)})
    errors = []
    with cache_path.open("a", encoding="utf-8") as stream, ThreadPoolExecutor(max_workers=workers) as executor:
        # Old expensive RGB hashes remain valid. New encoding receipt fields
        # require only mask headers (and RGB/RGBA mask arrays), not image rehashes.
        for result in executor.map(_enrich_cached, enrichment):
            metadata[result["id"]] = result["metadata"]
            stream.write(json.dumps(result, separators=(",", ":")) + "\n")
        stream.flush()
        for index, result in enumerate(executor.map(_audit_one, pending), 1):
            if "error" in result:
                errors.append(result)
            else:
                metadata[result["id"]] = result["metadata"]
                stream.write(json.dumps(result, separators=(",", ":")) + "\n")
            if index % 100 == 0 or index == len(pending):
                stream.flush()
                os.fsync(stream.fileno())
                state = {"stage": "DATA_AUDIT", "completed": len(metadata), "total": len(records), "errors": len(errors)}
                atomic_json(cache / "audit_progress.json", state)
                if progress:
                    progress(state)
    if errors:
        atomic_json(cache / "audit_errors.json", errors)
        by_id = {row["id"]: row for row in records}
        blocking = [error for error in errors if by_id[error["id"]]["split"] != "train" or error["error_code"] != "image_mask_coordinate_mismatch"]
        if blocking:
            raise ValueError(f"Data audit blocked by {len(blocking)} semantic/test errors; see {cache / 'audit_errors.json'}")
    return ([{**record, **metadata[record["id"]]} for record in records if record["id"] in metadata], errors)


def _role(source_id, seed):
    return int(hashlib.sha256(f"{seed}:source-role:{source_id}".encode()).hexdigest()[:16], 16) % 2


def _clean_crop(record, min_size=96, guard=16):
    """Choose an entirely annotation-zero square plus a protected surrounding band."""
    gt, valid, raw = read_mask(record)
    bad = (raw != 0) | ~valid
    integral = np.pad(bad.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    height, width = gt.shape
    for size in (512, 352, 256, 192, 128, min_size):
        extent = size + 2 * guard
        if extent > min(height, width):
            continue
        for top in np.unique(np.linspace(0, height - extent, 12).astype(int)):
            for left in np.unique(np.linspace(0, width - extent, 12).astype(int)):
                bottom, right = top + extent, left + extent
                count = integral[bottom, right] - integral[top, right] - integral[bottom, left] + integral[top, left]
                if count == 0:
                    return [int(left + guard), int(top + guard), int(left + guard + size), int(top + guard + size)]
    return None


def _fit_roles(train, seed, options):
    reference_candidates = [row for row in train if row["authentic_declared"] and row["empty_mask"] and
                            all(_role(source, seed) == 0 for source in row["source_ids"])]
    minimum = int(options.get("reference_min_images", 128))
    mode = "authentic"
    if len(reference_candidates) < minimum:
        mode = "mask_clean_proxy"
        reference_candidates = []
        for row in train:
            if row["source_ids"] and all(_role(source, seed) == 0 for source in row["source_ids"]):
                crop = _clean_crop(row, int(options.get("proxy_min_crop", 96)), int(options.get("proxy_guard", 16)))
                if crop is not None:
                    reference_candidates.append({**row, "reference_crop": crop})
        if len(reference_candidates) < minimum:
            raise ValueError("REFERENCE_SOURCE_BLOCKED: insufficient authentic images or guarded annotation-zero crops")
    # Deduplicate exact RGB in reference training only. Main training remains intact.
    seen, reference = set(), []
    for row in reference_candidates:
        if row["rgb_pixels_sha256"] not in seen:
            reference.append(row)
            seen.add(row["rgb_pixels_sha256"])
    reference_source_ids = {source for row in reference for source in row["source_ids"]}
    candidates = [row for row in train if row["source_ids"] and
                  all(_role(source, seed) == 1 for source in row["source_ids"]) and row["rgb_pixels_sha256"] not in seen]
    groups = defaultdict(list)
    for row in candidates:
        groups[(row["source_dataset"], bool(row["positive_pixels"]))].append(row)
    rng = random.Random(seed)
    for rows in groups.values():
        rng.shuffle(rows)
    # Round-robin explicit source x image-label strata; no test statistics involved.
    calibration = []
    keys = sorted(groups)
    maximum = min(2048, int(options.get("calibration_max_images", 2048)))
    while len(calibration) < maximum and any(groups.values()):
        for key in keys:
            if groups[key] and len(calibration) < maximum:
                calibration.append(groups[key].pop())
    if not calibration or not any(row["positive_pixels"] for row in calibration) or not any(row["valid_pixels"] > row["positive_pixels"] for row in calibration):
        raise ValueError("CALIBRATION_SOURCE_BLOCKED: source-disjoint training roles lack labeled pixels of both classes")
    if reference_source_ids & {source for row in calibration for source in row["source_ids"]}:
        raise AssertionError("Source leakage between fit roles")
    return reference, calibration, mode


def prepare_data(project, config, progress=print):
    """Audit and persist full train/test manifests; call once before DDP training.

    Config keys: seed; data.train_root, benchmark_spec, workers,
    calibration_max_images, reference_min_images, proxy_min_crop, proxy_guard.
    The returned record lists are directly accepted by the datasets below.
    """
    project = Path(project).resolve(strict=True)
    options = config.get("data", {})
    spec_path = Path(options.get("benchmark_spec", "config/benchmark_all8.json"))
    if not spec_path.is_absolute():
        spec_path = project / spec_path
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if tuple(row["name"] for row in spec["tests"]) != ALL8_NAMES:
        raise ValueError("Unexpected All8 test dataset configuration")
    train_spec = {"name": "FinalTrainData", "root": options.get("train_root", "/data0/hl/FinalTrainData"), "mask_suffix": ""}
    train = paired_records(train_spec, "train")
    tests = {item["name"]: paired_records(item, "test") for item in spec["tests"]}
    cache = _writable_directory(project, "cache/tect_diff/data-audit")
    initial_train_count = len(train)
    rows, invalid = _audit_records([*train, *[row for group in tests.values() for row in group]], cache, max(1, min(16, int(options.get("workers", 4)))), progress)
    train = [row for row in rows if row["split"] == "train"]
    tests = {name: [row for row in rows if row["dataset"] == name] for name in ALL8_NAMES}
    test_hashes = defaultdict(list)
    test_sources = defaultdict(list)
    for group in tests.values():
        for row in group:
            test_hashes[row["rgb_pixels_sha256"]].append(row["id"])
            for source in row["source_ids"]:
                test_sources[source].append(row["id"])
    excluded = [{**error, "reason": "invalid_training_pair_coordinate_mismatch"} for error in invalid]
    retained = []
    for row in train:
        matches = test_hashes.get(row["rgb_pixels_sha256"], [])
        source_matches = sorted({sample for source in row["source_ids"] for sample in test_sources.get(source, [])})
        if matches or source_matches:
            excluded.append({"id": row["id"], "reason": "train_test_leakage", "exact_rgb_test_ids": matches, "known_source_test_ids": source_matches})
        else:
            retained.append(row)
    reference, calibration, mode = _fit_roles(retained, int(config.get("seed", 42)), options)
    manifests = {"train": retained, "reference": reference, "calibration": calibration, **{"test_" + name: group for name, group in tests.items()}}
    hashes = {key: canonical_hash(value) for key, value in manifests.items()}
    manifest_id = canonical_hash([AUDIT_VERSION, hashes, mode, options, int(config.get("seed", 42))])[:20]
    destination = _writable_directory(project, "cache/tect_diff/manifests/" + manifest_id)
    paths = {}
    for key, value in manifests.items():
        path = destination / (key + ".json")
        if path.exists() and canonical_hash(json.loads(path.read_text())) != hashes[key]:
            raise ValueError("Refusing to change frozen manifest")
        if not path.exists():
            atomic_json(path, value)
        paths[key] = str(path)
    summary = {"audit_version": AUDIT_VERSION, "train_before_exclusions": initial_train_count, "train_count": len(retained),
               "train_authentic_count": sum(row["authentic_declared"] for row in retained),
               "test_counts": {name: len(group) for name, group in tests.items()}, "excluded_train_count": len(excluded),
               "invalid_training_pair_count": len(invalid), "train_test_leakage_exclusion_count": len(excluded) - len(invalid),
               "reference_count": len(reference), "calibration_count": len(calibration), "reference_source_mode": mode,
               "mask_semantics": "0=unedited; unit masks >0; uint8 masks >=128=tampered; training RGB requires channel-label consensus then PIL-L; tests retain registered PIL-L including alpha ignore; no inversion; no implicit ignore value",
               "test_mask_decoder": "Existing registered All8 PIL.convert(L), alpha ignored; threshold >=128 (unit masks >0); color diagnostics recorded per sample; no score-based preprocessing choice",
               "dataset_statistics": {name: {"count": len(group), "empty_masks": sum(row["empty_mask"] for row in group),
                   "soft_masks": sum(row["soft_mask_pixels"] > 0 for row in group), "unit_masks": sum(row["binary_unit_mask"] for row in group),
                   "mask_modes": dict(Counter(row["mask_mode"] for row in group)),
                   "non_gray_mask_count": sum(row["mask_non_gray_pixels"] > 0 for row in group),
                   "alpha_ignored_mask_count": sum(row["mask_alpha_ignored"] for row in group),
                   "ignored_pixels": sum(row["ignored_pixels"] for row in group),
                   "image_height_range": [min(row["image_hw"][0] for row in group), max(row["image_hw"][0] for row in group)],
                   "image_width_range": [min(row["image_hw"][1] for row in group), max(row["image_hw"][1] for row in group)]}
                   for name, group in {"FinalTrainData": train, **tests}.items()},
               "source_counts": dict(Counter(row["source_dataset"] for row in retained)),
               "reference_source_counts": dict(Counter(row["source_dataset"] for row in reference)),
               "calibration_strata": dict(Counter(row["source_dataset"] + (":positive" if row["positive_pixels"] else ":empty") for row in calibration)),
               "role_rule": "SHA256(seed:source-role:source_id) low bit; reference=0; calibration requires all source IDs=1; reference RGB deduplicated; source x label round-robin calibration; all retained images still main training",
               "pairing_rule": "exact relative-path stem; registered per-dataset suffix removed only from mask; all keys unique and paired",
               "leakage_rule": "exclude train records with exact decoded RGB or known namespaced source IDs present in tests; quarantine image-mask dimension mismatches as invalid training pairs; test anomalies block; no source files changed",
               "leakage_limit": "Exact hashes and available name-level source IDs cannot exclude transformed or unrecognized same-source leakage; CASIA1 and CASIA2 IDs are separate namespaces",
               "reference_limit": "Explicit Au/Fan_Real provenance plus actual zero masks; no missing-mask authenticity inference" if mode == "authentic" else "Guarded annotation-zero crops; does not exclude whole-image editing pipeline effects",
               "anomaly_count": len(invalid), "invalid_training_pairs": invalid,
               "mask_encoding_diagnostics": [{key: row[key] for key in ("id", "mask_decoder", "mask_mode", "mask_non_gray_pixels", "mask_rgb_threshold_disagreement_pixels", "mask_alpha_nonopaque_pixels", "mask_alpha_ignored")}
                   for row in rows if row["mask_non_gray_pixels"] or row["mask_alpha_ignored"]],
               "manifest_id": manifest_id, "manifest_hashes": hashes, "manifest_paths": paths}
    atomic_json(destination / "exclusions.json", excluded)
    atomic_json(destination / "summary.json", summary)
    return {"train": retained, "reference": reference, "calibration": calibration, "tests": tests,
            "reference_source_mode": mode, "manifest_hashes": hashes, "manifest_paths": paths,
            "summary": summary, "summary_path": str(destination / "summary.json")}


def load_bundle(summary_path):
    summary = json.loads(Path(summary_path).read_text())
    loaded = {}
    for key, path in summary["manifest_paths"].items():
        value = json.loads(Path(path).read_text())
        if canonical_hash(value) != summary["manifest_hashes"][key]:
            raise ValueError(f"Frozen manifest hash mismatch: {key}")
        loaded[key] = value
    return {"train": loaded["train"], "reference": loaded["reference"], "calibration": loaded["calibration"],
            "tests": {name: loaded["test_" + name] for name in ALL8_NAMES}, "summary": summary,
            "summary_path": str(summary_path), "manifest_hashes": summary["manifest_hashes"],
            "manifest_paths": summary["manifest_paths"], "reference_source_mode": summary["reference_source_mode"]}


class ManifestDataset:
    """Only resize and optional joint flip. Evaluation GT retains original pixels."""
    def __init__(self, records, size=512, training=False, original_gt=False, image_range="zero_one", include_trace=True):
        self.records, self.size, self.training = list(records), int(size), bool(training)
        self.original_gt, self.image_range = bool(original_gt), image_range
        self.include_trace = bool(include_trace)
        if image_range not in ("zero_one", "minus_one_one"):
            raise ValueError("Unknown image normalization")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        import torch
        row = self.records[index]
        with Image.open(row["image_path"]) as handle:
            original = handle.convert("RGB")
            rgb = original.resize((self.size, self.size), Image.Resampling.BILINEAR)
            if self.include_trace:
                from tools.generate_git10k_aux import make_high_frequency_view
                # The inherited HFVG definition runs at native size, before resize.
                trace_rgb = make_high_frequency_view(np.asarray(original), cutoff_ratio=0.5, boost=10.0)
                trace_rgb = Image.fromarray(trace_rgb).resize((self.size, self.size), Image.Resampling.BILINEAR)
                trace = np.array(trace_rgb, dtype=np.float32).transpose(2, 0, 1) / 127.5 - 1
        gt, valid, _ = read_mask(row)
        mask_small = np.array(Image.fromarray(gt.astype(np.uint8)).resize((self.size, self.size), Image.Resampling.NEAREST))
        valid_small = np.array(Image.fromarray(valid.astype(np.uint8)).resize((self.size, self.size), Image.Resampling.NEAREST))
        image = np.array(rgb, dtype=np.float32).transpose(2, 0, 1) / 255.0
        flipped = self.training and random.random() < 0.5
        if flipped:
            image, mask_small, valid_small = image[..., ::-1], mask_small[:, ::-1], valid_small[:, ::-1]
            if self.include_trace:
                trace = trace[..., ::-1]
        if self.image_range == "minus_one_one":
            image = image * 2 - 1
        result = {"image": torch.from_numpy(image.copy()), "mask": torch.from_numpy(mask_small[None].copy()).float(),
                  "valid": torch.from_numpy(valid_small[None].copy()).bool(), "id": row["id"],
                  "image_hash": row["rgb_pixels_sha256"], "flipped": flipped}
        if self.include_trace:
            result["trace"] = torch.from_numpy(trace.copy())
        if self.original_gt:
            result.update({"gt": gt, "gt_valid": valid, "original_hw": tuple(gt.shape)})
        return result


class ReferenceDataset:
    """Only confirmed whole authentic images or pre-registered guarded clean crops."""
    def __init__(self, records, size=512, training=True, image_range="minus_one_one"):
        self.records, self.size, self.training = list(records), int(size), bool(training)
        self.image_range = image_range
        if image_range not in ("zero_one", "minus_one_one"):
            raise ValueError("Unknown image normalization")
        if any(not (row.get("reference_crop") or (row["authentic_declared"] and row["empty_mask"])) for row in self.records):
            raise ValueError("Reference record is neither authentic nor an audited clean crop")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        import torch
        row = self.records[index]
        with Image.open(row["image_path"]) as handle:
            image = handle.convert("RGB")
            if row.get("reference_crop"):
                image = image.crop(tuple(row["reference_crop"]))
            image = image.resize((self.size, self.size), Image.Resampling.BILINEAR)
        array = np.array(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        if self.training and random.random() < 0.5:
            array = array[..., ::-1]
        if self.image_range == "minus_one_one":
            array = array * 2 - 1
        return {"image": torch.from_numpy(array.copy()), "id": row["id"]}


def evaluation_collate(batch):
    """Stack only resized inputs, keeping variable-sized original GT as lists."""
    import torch
    result = {"image": torch.stack([row["image"] for row in batch]), "id": [row["id"] for row in batch],
            "gt": [row["gt"] for row in batch], "gt_valid": [row["gt_valid"] for row in batch],
            "original_hw": [row["original_hw"] for row in batch], "image_hash": [row["image_hash"] for row in batch]}
    if "trace" in batch[0]:
        result["trace"] = torch.stack([row["trace"] for row in batch])
    return result


def stable_noise_seed(sample_id, purpose, seed=0):
    """Independent of epoch, rank, worker, batch shape, and evaluation ordering."""
    return int(hashlib.sha256(f"TECT-DIFF:{seed}:{sample_id}:{purpose}".encode()).hexdigest()[:16], 16) % (2**63 - 1)
