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
    format_results_table,
    validate_dataset_roots,
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
            "BN": {"F1": 0.1, "IoU": 0.2, "AUC": 0.3, "num_images": 1},
            "PE": {"F1": 0.2, "IoU": 0.3, "AUC": 0.4, "num_images": 1},
            "IA": {"F1": 0.3, "IoU": 0.4, "AUC": 0.5, "num_images": 1},
            "PP": {"F1": 0.4, "IoU": 0.5, "AUC": 0.6, "num_images": 1},
        }

        average = average_dataset_results(dataset_results, ["BN", "PE", "IA", "PP"])

        self.assertEqual(average["num_images"], 4)
        self.assertAlmostEqual(average["F1"], (0.1 + 0.2 + 0.3 + 0.4) / 4)
        self.assertAlmostEqual(average["IoU"], (0.2 + 0.3 + 0.4 + 0.5) / 4)
        self.assertAlmostEqual(average["AUC"], (0.3 + 0.4 + 0.5 + 0.6) / 4)

    def test_format_results_table_is_readable(self):
        dataset_results = {
            "BN": {"F1": 0.1, "IoU": 0.2, "AUC": 0.3, "num_images": 1},
            "PE": {"F1": 0.2, "IoU": 0.3, "AUC": 0.4, "num_images": 1},
            "IA": {"F1": 0.3, "IoU": 0.4, "AUC": 0.5, "num_images": 1},
            "PP": {"F1": 0.4, "IoU": 0.5, "AUC": 0.6, "num_images": 1},
        }
        average = average_dataset_results(dataset_results, ["BN", "PE", "IA", "PP"])

        table = format_results_table(dataset_results, average, ["BN", "PE", "IA", "PP"], threshold=0.5)

        self.assertIn("Dataset", table)
        self.assertIn("AUC", table)
        self.assertIn("BN", table)
        self.assertIn("Average", table)
        self.assertIn("0.2500", table)

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
