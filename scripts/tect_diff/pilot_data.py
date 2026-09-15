"""Immutable, bounded CASIA2 development manifests; source datasets stay read-only.

The inherited reference/calibration remain bound to their original manifests.
Only explicitly named Au-directory images receive project-created zero masks.
The quick test subsets are development feedback, not independent validation.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.tect_diff.data import (
    _audit_records, _signature, _writable_directory, atomic_json, canonical_hash,
    index_files, paired_records, source_metadata, sha256_file,
)

PILOT_DATA_VERSION = "TECT-CASIA2-PILOT-DATA-V2-QUARANTINE"
QUARANTINE_POLICY = "EXPLICIT_HASH_BOUND_TRAINING_PAIRS_V1"
TEST_NAMES = ("Casiav1", "Columbia")
DEFAULT_SIZES = {"train_authentic": 1024, "train_tampered": 1024,
                 "quick_test_per_dataset": 128, "authentic_probe": 64,
                 "preflight_train": 64}


def _rank(row, seed, purpose):
    # Including the source group before the image ID makes ordering independent
    # of traversal, workers, and the ordering of the inherited JSON arrays.
    sources = sorted(row.get("source_ids", []))
    return hashlib.sha256(json.dumps([seed, purpose, sources, row["id"]],
                                   separators=(",", ":")).encode()).hexdigest()


def _take_unique(rows, count, seed, purpose):
    selected, seen = [], set()
    for row in sorted(rows, key=lambda row: _rank(row, seed, purpose)):
        digest = row["rgb_pixels_sha256"]
        if digest in seen:
            continue
        selected.append(row)
        seen.add(digest)
        if len(selected) == count:
            return selected
    raise ValueError(f"Insufficient unique eligible images for {purpose}: requested {count}, found {len(selected)}")


def select_pilot_records(train, tests, *, seed, sizes):
    """Select only after excluding overlap with BOTH COMPLETE test sets."""
    options = {key:sizes.get(key, value) for key, value in DEFAULT_SIZES.items()}
    if any(type(options[key]) is not int or options[key] <= 0 for key in DEFAULT_SIZES):
        raise ValueError("Pilot sample sizes must be positive integers")
    if options["preflight_train"] % 2:
        raise ValueError("preflight_train must be even for equal authentic/tampered coverage")
    if set(tests) != set(TEST_NAMES):
        raise ValueError("Pilot requires exactly Casiav1 and Columbia")
    rgb_labels = {}
    for row in train:
        digest, authentic = row["rgb_pixels_sha256"], row["authentic_declared"]
        if digest in rgb_labels and rgb_labels[digest] != authentic:
            raise ValueError(f"Conflicting authentic/tampered labels for identical RGB: {row['id']}")
        rgb_labels[digest] = authentic
    test_hashes, test_sources = defaultdict(list), defaultdict(list)
    for name in TEST_NAMES:
        for row in tests[name]:
            test_hashes[row["rgb_pixels_sha256"]].append(row["id"])
            for source in row.get("source_ids", []):
                test_sources[source].append(row["id"])
    retained, excluded = [], []
    for row in sorted(train, key=lambda row: row["id"]):
        matches = sorted(test_hashes.get(row["rgb_pixels_sha256"], []))
        source_matches = sorted({sample for source in row.get("source_ids", [])
                                 for sample in test_sources.get(source, [])})
        if matches or source_matches:
            excluded.append({"id": row["id"], "reason": "full_test_overlap",
                             "exact_rgb_test_ids": matches, "known_source_test_ids": source_matches})
        else:
            retained.append(row)
    authentic = [row for row in retained if row["authentic_declared"] and row["empty_mask"]]
    probe = _take_unique(authentic, options["authentic_probe"], seed, "authentic-probe")
    probe_sources = {source for row in probe for source in row["source_ids"]}
    probe_hashes = {row["rgb_pixels_sha256"] for row in probe}
    eligible = []
    for row in retained:
        if probe_sources.intersection(row["source_ids"]) or row["rgb_pixels_sha256"] in probe_hashes:
            excluded.append({"id": row["id"], "reason": "authentic_probe_source_or_rgb"})
        else:
            eligible.append(row)
    selected = []
    for authentic_flag, key in ((True, "train_authentic"), (False, "train_tampered")):
        candidates = [row for row in eligible if row["authentic_declared"] == authentic_flag]
        selected.extend(_take_unique(candidates, options[key], seed, key))
    selected.sort(key=lambda row: _rank(row, seed, "train-order"))
    preflight = []
    for authentic_flag in (True, False):
        preflight.extend(_take_unique([r for r in selected if r["authentic_declared"] == authentic_flag],
                                      options["preflight_train"] // 2, seed, "preflight"))
    quick = {name: _take_unique(tests[name], options["quick_test_per_dataset"], seed, "quick-"+name)
             for name in TEST_NAMES}
    return {"train": selected, "tests": quick, "tests_full": tests,
            "authentic_probe": probe, "preflight_train": preflight, "exclusions": excluded,
            "eligible_counts": dict(Counter("authentic" if r["authentic_declared"] else "tampered" for r in eligible)),
            "sample_sizes": options}


def _zero_mask(project, image_path):
    with Image.open(image_path) as handle:
        width, height = handle.size
    directory = _writable_directory(project, "cache/tect_diff/pilot-data-v1/explicit-au-zero-masks")
    path = directory / f"zero-{height}x{width}.png"
    if not path.exists():
        temporary = path.with_name(path.name+f".tmp-{os.getpid()}")
        Image.fromarray(np.zeros((height, width), dtype=np.uint8)).save(temporary, format="PNG")
        os.replace(temporary, path)
    with Image.open(path) as handle:
        mask = np.asarray(handle)
        if handle.mode != "L" or mask.shape != (height, width) or np.any(mask):
            raise ValueError(f"Project Au zero mask is invalid: {path}")
    return str(path)


def _casia_records(project, root):
    root = Path(root).resolve(strict=True)
    authentic, tampered, masks = index_files(root/"Au"), index_files(root/"Tp"), index_files(root/"Gt", "_gt")
    if set(tampered) != set(masks):
        raise ValueError("CASIA2 Tp/Gt pairing is unresolved; missing masks never imply authenticity")
    records = []
    for directory, images in (("Au", authentic), ("Tp", tampered)):
        listing = root / ("au_list.txt" if directory == "Au" else "tp_list.txt")
        if listing.exists():
            names = [line.strip() for line in listing.read_text().splitlines() if line.strip()]
            if len(names) != len(set(names)) or set(names) != {Path(path).relative_to(root/directory).as_posix() for path in images.values()}:
                raise ValueError(f"CASIA2 {directory} listing disagrees with source images")
        for stem, image_path in images.items():
            # Resolve and contain every source path; a link to another source
            # directory cannot inherit Au semantic authority from its link name.
            if not Path(image_path).resolve(strict=True).is_relative_to(root/directory):
                raise ValueError(f"CASIA2 image escapes its declared {directory} root")
            source, source_ids, is_authentic, _ = source_metadata(stem, "FinalTrainData")
            if is_authentic != (directory == "Au") or source != "CASIA2":
                raise ValueError(f"Unconfirmed CASIA2 {directory} image provenance: {stem}")
            mask_path = _zero_mask(project, image_path) if is_authentic else masks[stem]
            if not is_authentic and not Path(mask_path).resolve(strict=True).is_relative_to(root/"Gt"):
                raise ValueError("CASIA2 mask escapes declared Gt root")
            records.append({"id": "CASIA2:"+stem, "stem":stem, "dataset":"CASIA2", "split":"train",
                            "image_path":image_path, "mask_path":mask_path, "source_dataset":source,
                            "source_ids":source_ids, "authentic_declared":is_authentic,
                            "label_basis":"explicit_CASIA2_Au_directory_and_Au_name_project_zero_mask" if is_authentic else "explicit_CASIA2_Tp_matched_Gt",
                            "mask_encoding":"unit_or_uint8_threshold128", "ignore_values":[],
                            "mask_decoder":"training_channel_consensus_PIL_L"})
    return records


def _inherited_records(project, options, inherited):
    if inherited is None:
        path = Path(options["inherited_bundle"])
        if not path.is_absolute():
            path = project / path
        inherited = json.loads(path.read_text())
    if isinstance(inherited, (str, Path)):
        inherited = json.loads(Path(inherited).read_text())
    summary = inherited.get("summary", inherited)
    paths, hashes = summary["manifest_paths"], summary["manifest_hashes"]
    loaded = {}
    for key in ("reference", "calibration", *("test_"+name for name in TEST_NAMES)):
        value = json.loads(Path(paths[key]).read_text())
        if canonical_hash(value) != hashes[key]:
            raise ValueError(f"Inherited frozen manifest hash mismatch: {key}")
        if not value:
            raise ValueError(f"Inherited frozen manifest is empty: {key}")
        loaded[key] = value
    spec_path = Path(options.get("benchmark_spec", "config/benchmark_all8.json"))
    if not spec_path.is_absolute():
        spec_path = project/spec_path
    specs = {spec["name"]:spec for spec in json.loads(spec_path.read_text())["tests"]}
    for name in TEST_NAMES:
        fresh = paired_records(specs[name], "test")
        expected = {row["id"]:row for row in fresh}
        rows = loaded["test_"+name]
        if len(rows) != len(expected) or {row["id"] for row in rows} != set(expected):
            raise ValueError(f"Inherited {name} test membership changed")
        for row in rows:
            now = expected[row["id"]]
            if any(row[key] != now[key] for key in now) or row["signature"] != _signature(now):
                raise ValueError(f"Inherited {name} source or decoder changed: {row['id']}")
    return loaded, summary


def _freeze(path, value):
    if path.exists():
        if canonical_hash(json.loads(path.read_text())) != canonical_hash(value):
            raise ValueError(f"Refusing to change frozen pilot manifest: {path}")
    else:
        atomic_json(path, value)


def _fit_overlap(probe, inherited):
    """The probe is held out of main training, not inherited fitting exposure."""
    result = {}
    for role in ("reference", "calibration"):
        hashes, sources = defaultdict(list), defaultdict(list)
        for row in inherited[role]:
            if row.get("rgb_pixels_sha256"):
                hashes[row["rgb_pixels_sha256"]].append(row["id"])
            for source in row.get("source_ids", []):
                sources[source].append(row["id"])
        matches = []
        for row in probe:
            rgb_ids = sorted(hashes.get(row["rgb_pixels_sha256"], []))
            source_ids = sorted({sample for source in row["source_ids"] for sample in sources.get(source, [])})
            if rgb_ids or source_ids:
                matches.append({"probe_id":row["id"], "exact_rgb_fit_ids":rgb_ids, "known_source_fit_ids":source_ids})
        result[role] = {"overlap_probe_count":len(matches),
                        "exact_rgb_overlap_probe_count":sum(bool(row["exact_rgb_fit_ids"]) for row in matches),
                        "known_source_overlap_probe_count":sum(bool(row["known_source_fit_ids"]) for row in matches),
                        "matches":matches}
    return result


def _quarantine_records(records, exceptions):
    """Exclude only exact registered training pairs before interpreting masks."""
    if not isinstance(exceptions, list):
        raise ValueError("quarantine_pairs must be a list")
    by_id = {row["id"]:row for row in records}
    if len(by_id) != len(records):
        raise ValueError("Duplicate IDs in source records")
    required = {"id", "image_sha256", "mask_sha256", "reason"}
    seen, quarantined = set(), []
    for entry in exceptions:
        if not isinstance(entry, dict) or set(entry) != required:
            raise ValueError("Quarantine fields must be exactly id/image_sha256/mask_sha256/reason")
        sample_id = entry["id"]
        if not isinstance(sample_id, str) or sample_id not in by_id:
            raise ValueError(f"Unknown quarantine ID: {sample_id}")
        if sample_id in seen:
            raise ValueError(f"Duplicate quarantine ID: {sample_id}")
        if not isinstance(entry["reason"], str) or not entry["reason"].strip():
            raise ValueError("Quarantine reason must be a nonempty string")
        row = by_id[sample_id]
        if row.get("split") != "train" or row.get("dataset") != "CASIA2":
            raise ValueError("Only CASIA2 training pairs may be quarantined")
        for key in ("image_sha256", "mask_sha256"):
            if not isinstance(entry[key], str) or re.fullmatch(r"[a-f0-9]{64}", entry[key]) is None:
                raise ValueError(f"Quarantine {key} must be a lowercase SHA256 digest")
        before = _signature(row)
        actual = {kind+"_sha256":sha256_file(row[kind+"_path"]) for kind in ("image", "mask")}
        if any(actual[key] != entry[key] for key in actual):
            raise ValueError(f"Quarantine image/mask hash mismatch: {sample_id}")
        if before != _signature(row):
            raise ValueError(f"Quarantine pair changed during verification: {sample_id}")
        seen.add(sample_id)
        quarantined.append({"id":sample_id, "reason":"registered_training_pair_quarantine",
                            "quarantine_reason":entry["reason"], "policy":QUARANTINE_POLICY,
                            "image_path":row["image_path"], "mask_path":row["mask_path"], **actual})
    return [row for row in records if row["id"] not in seen], sorted(quarantined, key=lambda row:row["id"])


def prepare_pilot_bundle(project, config, inherited_bundle=None, progress=print):
    """Audit CASIA2 once, reuse verified tests, and freeze deterministic roles.

    config.data: train_root, inherited_bundle, benchmark_spec, workers, and
    optional quarantine_pairs [{id, image_sha256, mask_sha256, reason}].
    config.pilot: train_authentic, train_tampered, quick_test_per_dataset,
    authentic_probe, preflight_train. Defaults: 1024/1024/128/64/64.
    """
    project = Path(project).resolve(strict=True)
    options = config.get("data", {})
    inherited, parent = _inherited_records(project, options, inherited_bundle)
    tests = {name:inherited["test_"+name] for name in TEST_NAMES}
    records = _casia_records(project, options.get("train_root", "/data1/data/datasets/CASIA2.0"))
    train_before_exclusions = len(records)
    records, quarantined = _quarantine_records(records, options.get("quarantine_pairs", []))
    cache = _writable_directory(project, "cache/tect_diff/pilot-data-v1/audit")
    audited, invalid = _audit_records(records, cache, max(1, min(16, int(options.get("workers", 12)))), progress)
    empty_tampered = [row["id"] for row in audited if not row["authentic_declared"] and row["empty_mask"]]
    if empty_tampered:
        raise ValueError(f"CASIA2 Tp has an unresolved empty annotation: {empty_tampered[:8]}")
    selected = select_pilot_records(audited, tests, seed=int(config.get("seed", 42)), sizes=config.get("pilot", {}))
    manifests = {key:selected[key] for key in ("train", "authentic_probe", "preflight_train")}
    manifests.update({key:inherited[key] for key in ("reference", "calibration")})
    manifests.update({"test_"+name:selected["tests"][name] for name in TEST_NAMES})
    manifests.update({"test_full_"+name:tests[name] for name in TEST_NAMES})
    hashes = {key:canonical_hash(rows) for key, rows in manifests.items()}
    exclusions = [*quarantined, *invalid, *selected["exclusions"]]
    identity = {"version":PILOT_DATA_VERSION, "seed":int(config.get("seed", 42)),
                "hashes":hashes, "sample_sizes":selected["sample_sizes"], "exclusions":canonical_hash(exclusions)}
    manifest_id = canonical_hash(identity)[:20]
    destination = _writable_directory(project, "cache/tect_diff/pilot-data-v1/manifests/"+manifest_id)
    paths = {}
    for key, rows in manifests.items():
        path = destination/(key+".json")
        _freeze(path, rows)
        paths[key] = str(path)
    summary = {"audit_version":PILOT_DATA_VERSION, "manifest_id":manifest_id,
               "manifest_hashes":hashes, "manifest_paths":paths, "seed":identity["seed"],
               "sample_sizes":selected["sample_sizes"], "train_before_exclusions":train_before_exclusions,
               "train_after_quarantine":len(records), "quarantine_policy":QUARANTINE_POLICY,
               "quarantined_pair_count":len(quarantined), "quarantined_pairs":quarantined,
               "audited_train_count":len(audited), "invalid_training_pairs":invalid,
               "current_audit_status":"COMPLETED", "current_blocking_audit_error_count":0,
               "invalid_training_pair_count":len(invalid),
               "audit_cache_history_path":str(cache/"audit_errors.json"),
               "audit_cache_history_notice":"Shared audit_errors.json may retain earlier failed-attempt errors and is never a current-run status source. This frozen summary records current audit success, exact quarantines and remaining coordinate-invalid pairs; unknown semantic errors prevent creation of this summary.",
               "eligible_counts":selected["eligible_counts"], "train_count":len(selected["train"]),
               "train_authentic_count":sum(row["authentic_declared"] for row in selected["train"]),
               "test_counts":{name:len(selected["tests"][name]) for name in TEST_NAMES},
               "test_full_counts":{name:len(tests[name]) for name in TEST_NAMES},
               "authentic_probe_count":len(selected["authentic_probe"]), "preflight_train_count":len(selected["preflight_train"]),
               "reference_count":len(inherited["reference"]), "calibration_count":len(inherited["calibration"]),
               "reference_source_mode":parent["reference_source_mode"],
               "inherited_manifest_hashes":{key:parent["manifest_hashes"][key] for key in inherited},
               "inherited_manifest_paths":{key:parent["manifest_paths"][key] for key in inherited},
               "authentic_probe_inherited_fit_overlap":_fit_overlap(selected["authentic_probe"], inherited),
               "exclusion_counts":dict(Counter(row.get("reason", row.get("error_code", "invalid_pair")) for row in exclusions)),
               "mask_semantics":"positive white; unit masks >0, uint8 >=128; only explicit Au directory plus Au name receives a project-created original-size zero mask; no missing-mask inference",
               "selection_rule":"seed/source/ID SHA256 order; exact RGB unique per role; authentic probe selected first, then exclude its CASIA2 sources/RGB from all main training; balanced fixed train/preflight",
               "leakage_rule":"exclude exact decoded RGB or known namespaced source IDs against both complete test sets before any pilot selection",
               "leakage_limit":"Known CASIA1/CASIA2 source IDs remain distinct namespaces; transformed or unrecognized cross-version sources are not proven independent. Authentic probe is disjoint from main training, not necessarily from the inherited reference/calibration.",
               "selection_protocol":"test_selected", "phase":"development_pilot",
               "test_interpretation":"Quick and full two-set evaluations are development feedback; their use to revise the model prevents an independent generalization claim."}
    _freeze(destination/"exclusions.json", exclusions)
    _freeze(destination/"summary.json", summary)
    return load_pilot_bundle(destination/"summary.json")


def validate_pilot_expansion(bundle, parent_summary):
    """Verify a nested training expansion after preparation, before model work.

    The caller binds parent_summary to the registered, completed parent run.
    This checks frozen rows and roles; it neither selects data nor refits assets.
    Preflight rows may change with the larger training pool.
    """
    summary = bundle["summary"]
    fixed = ("reference", "calibration", "authentic_probe",
             *("test_"+name for name in TEST_NAMES),
             *("test_full_"+name for name in TEST_NAMES))
    roles = ("train", "preflight_train", *fixed)
    contract = ("audit_version", "seed", "quarantine_policy", "quarantined_pair_count",
                "quarantined_pairs", "invalid_training_pairs", "reference_source_mode",
                "inherited_manifest_hashes", "inherited_manifest_paths",
                "mask_semantics", "selection_rule", "leakage_rule")
    for key in contract:
        if summary[key] != parent_summary[key]:
            raise ValueError(f"Pilot expansion contract changed: {key}")
    loaded = []
    for metadata in (parent_summary, summary):
        if (metadata["audit_version"] != PILOT_DATA_VERSION
                or metadata["current_audit_status"] != "COMPLETED"
                or metadata["current_blocking_audit_error_count"] != 0):
            raise ValueError("Pilot expansion requires completed semantic audits")
        if set(metadata["manifest_paths"]) != set(roles) or set(metadata["manifest_hashes"]) != set(roles):
            raise ValueError("Pilot expansion manifest roles changed")
        manifests = {}
        for role in roles:
            path = Path(metadata["manifest_paths"][role])
            if not path.is_absolute():
                raise ValueError("Pilot expansion requires absolute frozen manifest paths")
            rows = json.loads(path.read_text())
            if not isinstance(rows, list) or not rows or canonical_hash(rows) != metadata["manifest_hashes"][role]:
                raise ValueError(f"Pilot expansion manifest hash mismatch: {role}")
            manifests[role] = rows
        loaded.append(manifests)
    parent, child = loaded
    actual = {key:bundle[key] for key in ("train", "preflight_train", "reference", "calibration", "authentic_probe")}
    actual.update({"test_"+name:bundle["tests"][name] for name in TEST_NAMES})
    actual.update({"test_full_"+name:bundle["tests_full"][name] for name in TEST_NAMES})
    if actual != child or bundle["manifest_hashes"] != summary["manifest_hashes"]:
        raise ValueError("Loaded pilot expansion differs from frozen manifests")
    for role in fixed:
        if summary["manifest_hashes"][role] != parent_summary["manifest_hashes"][role]:
            raise ValueError(f"Fixed pilot role changed: {role}")
    sizes = [metadata["sample_sizes"] for metadata in (parent_summary, summary)]
    changing = {"train_authentic", "train_tampered"}
    if ({key:value for key,value in sizes[0].items() if key not in changing}
            != {key:value for key,value in sizes[1].items() if key not in changing}
            or any(type(sizes[1][key]) is not int or sizes[1][key] <= sizes[0][key] for key in changing)
            or sizes[1]["train_authentic"] != sizes[1]["train_tampered"]):
        raise ValueError("Pilot expansion must only increase balanced training sizes")
    for manifests, metadata in zip(loaded, (parent_summary, summary)):
        rows = manifests["train"]
        if len({row["id"] for row in rows}) != len(rows):
            raise ValueError("Duplicate training IDs in pilot expansion")
        authentic = sum(row["authentic_declared"] is True for row in rows)
        if (len(rows) != metadata["train_count"] or authentic != metadata["train_authentic_count"]
                or authentic != metadata["sample_sizes"]["train_authentic"]
                or len(rows)-authentic != metadata["sample_sizes"]["train_tampered"]):
            raise ValueError("Pilot expansion training counts differ from manifest")
    child_by_id = {row["id"]:row for row in child["train"]}
    for row in parent["train"]:
        if child_by_id.get(row["id"]) != row:
            raise ValueError(f"Parent training row missing or changed: {row['id']}")
    held = [*child["authentic_probe"], *(row for name in TEST_NAMES for row in child["test_full_"+name])]
    held_rgb = {row["rgb_pixels_sha256"] for row in held}
    held_sources = {source for row in held for source in row["source_ids"]}
    quarantine_ids = {row["id"] for row in summary["quarantined_pairs"]}
    for row in child["train"]:
        if row["rgb_pixels_sha256"] in held_rgb or held_sources.intersection(row["source_ids"]):
            raise ValueError(f"Pilot expansion test/probe overlap: {row['id']}")
        if (row["id"] in quarantine_ids or type(row["authentic_declared"]) is not bool
                or row["empty_mask"] != row["authentic_declared"]):
            raise ValueError(f"Pilot expansion semantic/quarantine violation: {row['id']}")
    if len({row["rgb_pixels_sha256"] for row in child["train"]}) != len(child["train"]):
        raise ValueError("Duplicate training RGB in pilot expansion")
    return {"status":"PASSED", "parent_summary_hash":canonical_hash(parent_summary),
            "parent_manifest_id":parent_summary["manifest_id"], "manifest_id":summary["manifest_id"],
            "parent_train_hash":parent_summary["manifest_hashes"]["train"],
            "train_hash":summary["manifest_hashes"]["train"],
            "parent_train_count":len(parent["train"]), "train_count":len(child["train"]),
            "retained_parent_train_count":len(parent["train"]), "all_parent_rows_identical":True,
            "unchanged_manifest_hashes":{role:summary["manifest_hashes"][role] for role in fixed},
            "test_probe_source_and_rgb_disjoint":True,
            "quarantined_pair_count":summary["quarantined_pair_count"]}


def load_pilot_bundle(summary_path):
    summary = json.loads(Path(summary_path).read_text())
    if summary.get("audit_version") != PILOT_DATA_VERSION:
        raise ValueError("Unknown pilot manifest contract")
    loaded = {}
    for key, path in summary["manifest_paths"].items():
        rows = json.loads(Path(path).read_text())
        if canonical_hash(rows) != summary["manifest_hashes"][key]:
            raise ValueError(f"Frozen pilot manifest hash mismatch: {key}")
        loaded[key] = rows
    return {key:loaded[key] for key in ("train", "reference", "calibration", "authentic_probe", "preflight_train")} | {
        "tests":{name:loaded["test_"+name] for name in TEST_NAMES},
        "tests_full":{name:loaded["test_full_"+name] for name in TEST_NAMES},
        "reference_source_mode":summary["reference_source_mode"], "summary":summary,
        "summary_path":str(summary_path), "manifest_hashes":summary["manifest_hashes"],
        "manifest_paths":summary["manifest_paths"]}
