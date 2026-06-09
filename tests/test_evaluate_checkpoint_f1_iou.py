import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.evaluate_checkpoint_f1_iou import (  # noqa: E402
    binary_f1_iou,
    collect_mask_files,
    evaluate_prediction_folder,
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


if __name__ == "__main__":
    unittest.main()
