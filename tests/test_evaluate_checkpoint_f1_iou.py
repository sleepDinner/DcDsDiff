import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.evaluate_checkpoint_f1_iou import (  # noqa: E402
    average_dataset_results,
    binary_auc,
    binary_f1_iou,
    collect_mask_files,
    evaluate_prediction_folder,
    format_available_datasets,
    format_results_csv,
    format_results_table,
    infer_external_image_mask_roots,
    make_results_payload,
    parse_external_dataset_spec,
    prepare_external_dataset,
    resolve_dataset_sources,
    safe_dataset_dir_name,
    save_results_report,
    validate_dataset_roots,
    weighted_average_results,
)


class EvaluateCheckpointF1IoUTests(unittest.TestCase):
    def test_binary_f1_iou_perfect_match(self):
        pred = np.array([[0.9, 0.1], [0.8, 0.0]], dtype=np.float32)
        gt = np.array([[1, 0], [1, 0]], dtype=np.float32)

        f1, iou = binary_f1_iou(pred, gt, threshold=0.5)

        self.assertAlmostEqual(f1, 1.0)
        self.assertAlmostEqual(iou, 1.0)

    def test_binary_f1_iou_partial_overlap(self):
        pred = np.array([[0.9, 0.9], [0.1, 0.1]], dtype=np.float32)
        gt = np.array([[1, 0], [1, 0]], dtype=np.float32)

        f1, iou = binary_f1_iou(pred, gt, threshold=0.5)

        self.assertAlmostEqual(f1, 0.5)
        self.assertAlmostEqual(iou, 1.0 / 3.0)

    def test_binary_auc_uses_continuous_prediction_scores(self):
        gt = np.array([[1, 0], [1, 0]], dtype=np.float32)

        self.assertAlmostEqual(binary_auc(np.array([[0.9, 0.2], [0.8, 0.1]], dtype=np.float32), gt), 1.0)
        self.assertAlmostEqual(binary_auc(np.array([[0.1, 0.8], [0.2, 0.9]], dtype=np.float32), gt), 0.0)
        self.assertAlmostEqual(binary_auc(np.ones((2, 2), dtype=np.float32) * 0.5, gt), 0.5)

    def test_collect_mask_files_pairs_by_stem(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("L", (2, 2)).save(root / "sample.png")
            (root / "ignored.txt").write_text("not an image")

            masks = collect_mask_files(root)

        self.assertEqual(list(masks), ["sample"])

    def test_safe_dataset_dir_name_keeps_cli_friendly_names(self):
        self.assertEqual(safe_dataset_dir_name("DSO-1"), "DSO-1")
        self.assertEqual(safe_dataset_dir_name("CASIA v1"), "CASIA_v1")

    def test_parse_external_dataset_spec_inferrs_images_and_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_root = root / "images"
            mask_root = root / "masks"
            image_root.mkdir()
            mask_root.mkdir()
            Image.new("RGB", (2, 2)).save(image_root / "a.jpg")
            Image.new("L", (2, 2)).save(mask_root / "a.png")

            name, parsed_image_root, parsed_mask_root = parse_external_dataset_spec(f"Casiav1={root}")

        self.assertEqual(name, "Casiav1")
        self.assertEqual(parsed_image_root, image_root)
        self.assertEqual(parsed_mask_root, mask_root)

    def test_parse_external_dataset_spec_accepts_explicit_roots(self):
        name, image_root, mask_root = parse_external_dataset_spec("Korus=/data/images,/data/masks")

        self.assertEqual(name, "Korus")
        self.assertEqual(image_root, Path("/data/images"))
        self.assertEqual(mask_root, Path("/data/masks"))

    def test_infer_external_image_mask_roots_reports_child_folders(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "unknown").mkdir()

            with self.assertRaises(SystemExit) as ctx:
                infer_external_image_mask_roots(root)

        self.assertIn("Cannot infer image/mask folders", str(ctx.exception))
        self.assertIn("unknown", str(ctx.exception))

    def test_prepare_external_dataset_writes_four_channel_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_root = root / "images"
            mask_root = root / "masks"
            work_root = root / "prepared"
            image_root.mkdir()
            mask_root.mkdir()
            Image.new("RGB", (4, 4), color=(128, 64, 32)).save(image_root / "sample.jpg")
            Image.fromarray(np.array([[0, 0, 0, 0], [0, 255, 255, 0], [0, 255, 255, 0], [0, 0, 0, 0]], dtype=np.uint8)).save(mask_root / "sample.png")

            prepared_root, num_images = prepare_external_dataset(
                "External Test",
                image_root,
                mask_root,
                work_root,
                detail_radius=15.0,
                edge_kernel=3,
                cutoff_ratio=0.5,
                boost=10.0,
                overwrite=False,
                num_workers=1,
            )

            self.assertEqual(num_images, 1)
            self.assertTrue((prepared_root / "f" / "sample.png").exists())
            self.assertTrue((prepared_root / "m" / "sample.png").exists())
            self.assertTrue((prepared_root / "d" / "sample.png").exists())
            self.assertTrue((prepared_root / "t" / "sample.png").exists())

    def test_evaluate_prediction_folder_resizes_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gt_root = root / "gt"
            pred_root = root / "pred"
            gt_root.mkdir()
            pred_root.mkdir()
            Image.fromarray(np.array([[255, 0], [255, 0]], dtype=np.uint8)).save(gt_root / "a.png")
            Image.fromarray(np.array([[255, 0]], dtype=np.uint8)).save(pred_root / "a.png")

            results = evaluate_prediction_folder(gt_root, pred_root, threshold=0.5)

        self.assertEqual(results["num_images"], 1)
        self.assertAlmostEqual(results["F1"], 1.0)
        self.assertAlmostEqual(results["IoU"], 1.0)
        self.assertAlmostEqual(results["AUC"], 1.0)

    def test_average_dataset_results_uses_existing_subset_values(self):
        dataset_results = {
            "BN": {"F1": 0.1, "IoU": 0.2, "AUC": 0.3, "num_images": 1, "sources": ["BN"]},
            "PE": {"F1": 0.2, "IoU": 0.3, "AUC": 0.4, "num_images": 1, "sources": ["EI"]},
            "IA": {"F1": 0.3, "IoU": 0.4, "AUC": 0.5, "num_images": 1, "sources": ["IA"]},
            "PP": {"F1": 0.4, "IoU": 0.5, "AUC": 0.6, "num_images": 1, "sources": ["PP"]},
        }

        average = average_dataset_results(dataset_results, ["BN", "PE", "IA", "PP"])

        self.assertEqual(average["num_images"], 4)
        self.assertAlmostEqual(average["F1"], (0.1 + 0.2 + 0.3 + 0.4) / 4)
        self.assertAlmostEqual(average["IoU"], (0.2 + 0.3 + 0.4 + 0.5) / 4)
        self.assertAlmostEqual(average["AUC"], (0.3 + 0.4 + 0.5 + 0.6) / 4)

    def test_format_results_table_is_readable(self):
        dataset_results = {
            "BN": {"F1": 0.1, "IoU": 0.2, "AUC": 0.3, "num_images": 1, "sources": ["BN", "RBN"]},
            "PE": {"F1": 0.2, "IoU": 0.3, "AUC": 0.4, "num_images": 1, "sources": ["EI"]},
            "IA": {"F1": 0.3, "IoU": 0.4, "AUC": 0.5, "num_images": 1, "sources": ["IA"]},
            "PP": {"F1": 0.4, "IoU": 0.5, "AUC": 0.6, "num_images": 1, "sources": ["PP"]},
        }
        average = average_dataset_results(dataset_results, ["BN", "PE", "IA", "PP"])

        table = format_results_table(dataset_results, average, ["BN", "PE", "IA", "PP"], threshold=0.5)

        self.assertIn("Dataset", table)
        self.assertIn("Sources", table)
        self.assertIn("AUC", table)
        self.assertIn("BN", table)
        self.assertIn("BN+RBN", table)
        self.assertIn("Average", table)
        self.assertIn("0.2500", table)

    def test_save_results_report_writes_text_json_and_csv(self):
        dataset_keys = ["BN", "PE", "IA", "PP"]
        dataset_results = {
            "BN": {"F1": 0.1, "IoU": 0.2, "AUC": 0.3, "num_images": 1, "sources": ["BN", "RBN"]},
            "PE": {"F1": 0.2, "IoU": 0.3, "AUC": 0.4, "num_images": 1, "sources": ["EI"]},
            "IA": {"F1": 0.3, "IoU": 0.4, "AUC": 0.5, "num_images": 1, "sources": ["IA"]},
            "PP": {"F1": 0.4, "IoU": 0.5, "AUC": 0.6, "num_images": 1, "sources": ["PP"]},
        }
        source_results = {"BN": {"num_images": 1, "F1": 0.1, "IoU": 0.2, "AUC": 0.3}}
        average = average_dataset_results(dataset_results, dataset_keys)
        table = format_results_table(dataset_results, average, dataset_keys, threshold=0.5)
        csv_text = format_results_csv(dataset_results, average, dataset_keys)
        payload = make_results_payload(
            dataset_results,
            source_results,
            average,
            dataset_keys,
            threshold=0.5,
            checkpoint="/tmp/model-best.pt",
        )

        with tempfile.TemporaryDirectory() as tmp:
            paths = save_results_report(tmp, table, payload, csv_text)
            txt_path = Path(paths["txt"])
            json_path = Path(paths["json"])
            csv_path = Path(paths["csv"])

            self.assertTrue(txt_path.exists())
            self.assertTrue(json_path.exists())
            self.assertTrue(csv_path.exists())
            self.assertIn("Evaluation metrics", txt_path.read_text(encoding="utf-8"))
            self.assertIn('"average"', json_path.read_text(encoding="utf-8"))
            self.assertIn("Dataset,Sources,Images,F1,IoU,AUC", csv_path.read_text(encoding="utf-8"))

    def test_resolve_dataset_sources_maps_paper_names_to_released_prefixes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "Diff_dataset" / "Test" / "Diff"
            for key in ("Mix", "BN", "RBN", "EI", "Flux", "e", "t", "z", "IA", "PP"):
                for subdir in ("f", "m", "d", "t"):
                    (root / key / subdir).mkdir(parents=True)
                Image.new("L", (2, 2)).save(root / key / "f" / "a.png")
            cfg = OmegaConf.create(
                {
                    "test_dataset": {
                        "Mix": {
                            "name": "dataset.data_val.test_dataset",
                            "params": {
                                "image_root": str(root / "Mix" / "f") + "/",
                                "gt_root": str(root / "Mix" / "m") + "/",
                                "de_root": str(root / "Mix" / "d") + "/",
                                "trace_root": str(root / "Mix" / "t") + "/",
                                "testsize": 352,
                            },
                        }
                    }
                }
            )

            self.assertEqual(resolve_dataset_sources(cfg, "BN"), ["BN", "RBN"])
            self.assertEqual(resolve_dataset_sources(cfg, "PE"), ["EI", "Flux", "e", "t", "z"])
            self.assertEqual(resolve_dataset_sources(cfg, "IA"), ["IA"])
            self.assertEqual(resolve_dataset_sources(cfg, "PP"), ["PP"])

    def test_weighted_average_results_combines_source_prefixes_by_image_count(self):
        source_results = {
            "EI": {"F1": 0.2, "IoU": 0.4, "AUC": 0.6, "MAE": 0.1, "num_images": 1},
            "e": {"F1": 0.8, "IoU": 1.0, "AUC": 0.0, "MAE": 0.3, "num_images": 3},
        }

        result = weighted_average_results(source_results, ["EI", "e"])

        self.assertEqual(result["num_images"], 4)
        self.assertEqual(result["sources"], ["EI", "e"])
        self.assertAlmostEqual(result["F1"], (0.2 * 1 + 0.8 * 3) / 4)
        self.assertAlmostEqual(result["IoU"], (0.4 * 1 + 1.0 * 3) / 4)
        self.assertAlmostEqual(result["AUC"], (0.6 * 1 + 0.0 * 3) / 4)

    def test_validate_dataset_roots_reports_missing_subset_before_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "Diff_dataset" / "Test" / "Diff"
            for subdir in ("f", "m", "d", "t"):
                (root / "Mix" / subdir).mkdir(parents=True)
                (root / "BN" / subdir).mkdir(parents=True)
            cfg = OmegaConf.create(
                {
                    "test_dataset": {
                        "Mix": {
                            "name": "dataset.data_val.test_dataset",
                            "params": {
                                "image_root": str(root / "Mix" / "f") + "/",
                                "gt_root": str(root / "Mix" / "m") + "/",
                                "de_root": str(root / "Mix" / "d") + "/",
                                "trace_root": str(root / "Mix" / "t") + "/",
                                "testsize": 352,
                            },
                        }
                    },
                    "pred_root": None,
                    "results_folder": str(Path(tmp) / "eval"),
                }
            )

            with self.assertRaises(SystemExit) as ctx:
                validate_dataset_roots(cfg, ["BN", "PE"], skip_inference=False, multi_dataset=True)

        message = str(ctx.exception)
        self.assertIn("PE.image_root", message)
        self.assertIn("Available test datasets", message)
        self.assertIn("BN", message)

    def test_format_available_datasets_counts_subfolders(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "Diff_dataset" / "Test" / "Diff"
            for subdir in ("f", "m", "d", "t"):
                (root / "Mix" / subdir).mkdir(parents=True)
            Image.new("L", (2, 2)).save(root / "Mix" / "f" / "a.png")
            cfg = OmegaConf.create(
                {
                    "test_dataset": {
                        "Mix": {
                            "name": "dataset.data_val.test_dataset",
                            "params": {
                                "image_root": str(root / "Mix" / "f") + "/",
                                "gt_root": str(root / "Mix" / "m") + "/",
                                "de_root": str(root / "Mix" / "d") + "/",
                                "trace_root": str(root / "Mix" / "t") + "/",
                                "testsize": 352,
                            },
                        }
                    }
                }
            )

            output = format_available_datasets(cfg)

        self.assertIn("Dataset", output)
        self.assertIn("Mix", output)
        self.assertIn("1", output)


if __name__ == "__main__":
    unittest.main()
