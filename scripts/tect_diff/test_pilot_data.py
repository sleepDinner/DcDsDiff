"""Data-contract checks for the bounded CASIA2 development protocol."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.tect_diff.data import _audit_one, canonical_hash, paired_records, read_mask, sha256_file
from scripts.tect_diff.pilot_data import prepare_pilot_bundle, load_pilot_bundle, select_pilot_records, _fit_overlap


def record(name, authentic, source, rgb=None):
    return {"id": name, "authentic_declared": authentic, "source_ids": [source],
            "rgb_pixels_sha256": rgb or name, "empty_mask": authentic}


class PilotSelectionTests(unittest.TestCase):
    def test_conflicting_authentic_labels_for_identical_rgb_are_rejected(self):
        rows = [record("au", True, "CASIA2:a", "same"), record("tp", False, "CASIA2:b", "same")]
        tests = {name:[record(name, False, name)] for name in ("Casiav1", "Columbia")}
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            select_pilot_records(rows, tests, seed=42, sizes={})

    def test_probe_exposure_to_inherited_fitting_is_disclosed(self):
        probe = [record("probe", True, "CASIA2:a", "same-rgb")]
        inherited = {"reference":[record("old-ref", True, "CASIA2:b", "same-rgb")],
                     "calibration":[record("old-cal", True, "CASIA2:a", "other-rgb")]}
        overlap = _fit_overlap(probe, inherited)
        self.assertEqual(overlap["reference"]["exact_rgb_overlap_probe_count"], 1)
        self.assertEqual(overlap["calibration"]["known_source_overlap_probe_count"], 1)

    def test_excludes_full_test_overlap_before_quick_subset_selection(self):
        rows = [record(f"au{x}", True, f"CASIA2:a{x}") for x in range(8)]
        rows += [record(f"tp{x}", False, f"CASIA2:t{x}") for x in range(8)]
        tests = {"Casiav1": [record("test-rgb", False, "CASIA1:q", "tp0"),
                               record("test-source", False, "CASIA2:t1")],
                 "Columbia": [record("test-columbia", False, "Columbia:a")]}
        sizes = {"train_authentic": 2, "train_tampered": 2, "authentic_probe": 2,
                 "quick_test_per_dataset": 1, "preflight_train": 2}
        result = select_pilot_records(rows, tests, seed=42, sizes=sizes)
        excluded = {x["id"] for x in result["exclusions"]}
        self.assertTrue({"tp0", "tp1"} <= excluded)
        self.assertEqual(len(result["tests_full"]["Casiav1"]), 2)

    def test_probe_and_train_are_source_and_rgb_disjoint_and_order_independent(self):
        rows = [record(f"au{x}", True, f"CASIA2:a{x}") for x in range(12)]
        rows += [record(f"tp{x}", False, f"CASIA2:a{x}") for x in range(12)]
        tests = {name: [record(name+str(x), False, name+str(x)) for x in range(3)]
                 for name in ("Casiav1", "Columbia")}
        sizes = {"train_authentic": 3, "train_tampered": 3, "authentic_probe": 2,
                 "quick_test_per_dataset": 2, "preflight_train": 4}
        first = select_pilot_records(rows, tests, seed=42, sizes=sizes)
        second = select_pilot_records(list(reversed(rows)), {k:list(reversed(v)) for k,v in tests.items()},
                                      seed=42, sizes=sizes)
        for key in ("train", "authentic_probe", "preflight_train"):
            self.assertEqual([x["id"] for x in first[key]], [x["id"] for x in second[key]])
        train_sources = {x for r in first["train"] for x in r["source_ids"]}
        probe_sources = {x for r in first["authentic_probe"] for x in r["source_ids"]}
        self.assertFalse(train_sources & probe_sources)
        self.assertEqual(sum(x["authentic_declared"] for x in first["preflight_train"]), 2)
        self.assertLessEqual({x["id"] for x in first["preflight_train"]}, {x["id"] for x in first["train"]})

    def test_insufficient_clean_source_pool_fails_instead_of_relaxing_constraints(self):
        sizes = {"train_authentic": 1, "train_tampered": 1, "authentic_probe": 1,
                 "quick_test_per_dataset": 1, "preflight_train": 2}
        rows = [record("au", True, "shared"), record("tp", False, "shared")]
        tests = {name:[record(name, False, name)] for name in ("Casiav1", "Columbia")}
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            select_pilot_records(rows, tests, seed=42, sizes=sizes)


class PilotBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        self.source = self.project / "source_data"
        for directory in ("Au", "Tp", "Gt"):
            (self.source / directory).mkdir(parents=True)
        for i in range(10):
            self.image(self.source / "Au" / f"Au_ani_{i:05d}.png", i+1)
            stem = f"Tp_D_CND_M_N_ani{i+100:05d}_ani{i+200:05d}_{i:05d}"
            self.image(self.source / "Tp" / (stem + ".png"), i+31)
            Image.fromarray(np.pad(np.ones((2, 3), dtype=np.uint8)*255, ((1, 1), (1, 2)))).save(self.source/"Gt"/(stem+"_gt.png"))
        tests = {}
        specs = []
        for name in ("Casiav1", "Columbia"):
            root = self.project / name
            (root / "images").mkdir(parents=True)
            (root / "masks").mkdir()
            for i in range(3):
                self.image(root / "images" / f"sample{i}.png", 70+i+(10 if name=="Columbia" else 0))
                Image.fromarray(np.ones((4, 6), dtype=np.uint8)*255).save(root / "masks" / f"sample{i}_gt.png")
            spec = {"name":name, "root":str(root), "mask_suffix":"_gt", "count":3}
            specs.append(spec)
            tests[name] = [{**row, **_audit_one(row)["metadata"]} for row in paired_records(spec, "test")]
        self.spec = self.project / "benchmark.json"
        self.spec.write_text(json.dumps({"tests":specs}))
        manifests = {"test_"+name:rows for name, rows in tests.items()}
        manifests.update({"reference":[{"id":"reference-original", "source_ids":["CASIA2:old"]}],
                          "calibration":[{"id":"calibration-original", "source_ids":["CASIA2:other"]}]})
        paths, hashes = {}, {}
        for key, rows in manifests.items():
            path = self.project / (key + ".json")
            path.write_text(json.dumps(rows)); paths[key] = str(path); hashes[key] = canonical_hash(rows)
        self.inherited = {"manifest_paths":paths, "manifest_hashes":hashes, "reference_source_mode":"authentic"}
        self.config = {"seed":42, "data":{"train_root":str(self.source), "benchmark_spec":str(self.spec), "workers":2},
                       "pilot":{"train_authentic":2, "train_tampered":2, "authentic_probe":2,
                                "quick_test_per_dataset":2, "preflight_train":2}}

    @staticmethod
    def image(path, value):
        Image.fromarray(np.full((4, 6, 3), value, dtype=np.uint8)).save(path)

    def test_explicit_au_zero_masks_and_inherited_artifacts_remain_bound(self):
        before = sorted(str(p.relative_to(self.source)) for p in self.source.rglob("*") if p.is_file())
        bundle = prepare_pilot_bundle(self.project, self.config, self.inherited, progress=None)
        self.assertEqual(len(bundle["train"]), 4)
        for row in bundle["authentic_probe"]:
            gt, valid, _ = read_mask(row)
            self.assertEqual(gt.shape, (4, 6)); self.assertFalse(gt.any()); self.assertTrue(valid.all())
            self.assertTrue(Path(row["mask_path"]).is_relative_to(self.project / "cache"))
        self.assertEqual(bundle["manifest_hashes"]["reference"], self.inherited["manifest_hashes"]["reference"])
        self.assertEqual(bundle["manifest_hashes"]["calibration"], self.inherited["manifest_hashes"]["calibration"])
        self.assertEqual(before, sorted(str(p.relative_to(self.source)) for p in self.source.rglob("*") if p.is_file()))
        self.assertEqual(load_pilot_bundle(bundle["summary_path"])["manifest_hashes"], bundle["manifest_hashes"])

    def test_missing_tampered_mask_is_never_inferred_authentic(self):
        next((self.source/"Gt").iterdir()).unlink()
        with self.assertRaisesRegex(ValueError, "pair"):
            prepare_pilot_bundle(self.project, self.config, self.inherited, progress=None)

    def test_empty_tampered_annotation_requires_explicit_resolution(self):
        Image.fromarray(np.zeros((4, 6), dtype=np.uint8)).save(next((self.source/"Gt").iterdir()))
        with self.assertRaisesRegex(ValueError, "empty annotation"):
            prepare_pilot_bundle(self.project, self.config, self.inherited, progress=None)

    def test_manifest_tampering_is_rejected(self):
        bundle = prepare_pilot_bundle(self.project, self.config, self.inherited, progress=None)
        Path(bundle["manifest_paths"]["train"]).write_text("[]")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_pilot_bundle(bundle["summary_path"])

    def test_inherited_test_source_change_is_rejected(self):
        row = json.loads(Path(self.inherited["manifest_paths"]["test_Casiav1"]).read_text())[0]
        self.image(Path(row["image_path"]), 199)
        with self.assertRaisesRegex(ValueError, "changed"):
            prepare_pilot_bundle(self.project, self.config, self.inherited, progress=None)

    def quarantine_fixture(self):
        mask = next((self.source/"Gt").iterdir())
        values = np.zeros((4, 6, 2), dtype=np.uint8)
        values[..., 1] = 255
        Image.fromarray(values).save(mask)
        stem = mask.stem[:-3]
        image = self.source/"Tp"/(stem+".png")
        return {"id":"CASIA2:"+stem, "image_sha256":sha256_file(image),
                "mask_sha256":sha256_file(mask), "reason":"Registered fixture LA-channel ambiguity"}

    def test_exact_registered_quarantine_precedes_mask_decode_and_preserves_sources(self):
        exception = self.quarantine_fixture()
        before = {str(p):sha256_file(p) for p in self.source.rglob("*") if p.is_file()}
        with self.assertRaisesRegex(ValueError, "Data audit blocked"):
            prepare_pilot_bundle(self.project, self.config, self.inherited, progress=None)
        historical_error_path = self.project/"cache/tect_diff/pilot-data-v1/audit/audit_errors.json"
        historical_errors = historical_error_path.read_bytes()
        self.config["data"]["quarantine_pairs"] = [exception]
        bundle = prepare_pilot_bundle(self.project, self.config, self.inherited, progress=None)
        summary = bundle["summary"]
        self.assertEqual(summary["train_before_exclusions"], 20)
        self.assertEqual(summary["train_after_quarantine"], 19)
        self.assertEqual(summary["quarantined_pair_count"], 1)
        self.assertEqual(summary["audited_train_count"], 19)
        self.assertEqual(summary["current_audit_status"], "COMPLETED")
        self.assertEqual(summary["current_blocking_audit_error_count"], 0)
        self.assertEqual(historical_error_path.read_bytes(), historical_errors)
        self.assertFalse(any(row["id"] == exception["id"] for row in bundle["train"]))
        excluded = json.loads((Path(bundle["summary_path"]).parent/"exclusions.json").read_text())
        quarantine = [row for row in excluded if row["reason"] == "registered_training_pair_quarantine"]
        self.assertEqual(quarantine[0]["image_sha256"], exception["image_sha256"])
        self.assertEqual(quarantine[0]["mask_sha256"], exception["mask_sha256"])
        self.assertEqual(quarantine[0]["quarantine_reason"], exception["reason"])
        self.assertEqual(before, {str(p):sha256_file(p) for p in self.source.rglob("*") if p.is_file()})

    def test_quarantine_rejects_unknown_duplicate_missing_reason_and_bad_hash(self):
        exception = self.quarantine_fixture()
        cases = [([{**exception, "id":"CASIA2:unknown"}], "Unknown"),
                 ([exception, exception], "Duplicate"),
                 ([{key:value for key,value in exception.items() if key != "reason"}], "fields"),
                 ([{**exception, "reason":"  "}], "reason"),
                 ([{**exception, "image_sha256":"0"*64}], "hash mismatch"),
                 ([{**exception, "mask_sha256":"0"*64}], "hash mismatch"),
                 ([{**exception, "image_sha256":"not-a-sha"}], "SHA256"),
                 ({"unexpected":"mapping"}, "list")]
        for entries, error in cases:
            with self.subTest(error=error, entries=entries):
                self.config["data"]["quarantine_pairs"] = entries
                with self.assertRaisesRegex(ValueError, error):
                    prepare_pilot_bundle(self.project, self.config, self.inherited, progress=None)

    def test_unregistered_ambiguous_mask_still_blocks(self):
        self.quarantine_fixture()
        with self.assertRaisesRegex(ValueError, "Data audit blocked"):
            prepare_pilot_bundle(self.project, self.config, self.inherited, progress=None)


if __name__ == "__main__":
    unittest.main()
