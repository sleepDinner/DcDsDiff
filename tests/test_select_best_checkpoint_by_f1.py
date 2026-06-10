import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.select_best_checkpoint_by_f1 import (  # noqa: E402
    CheckpointCandidate,
    better_row,
    default_best_output_path,
    list_checkpoint_candidates,
    parse_checkpoint_epoch,
    row_from_results,
)


class SelectBestCheckpointByF1Tests(unittest.TestCase):
    def test_parse_checkpoint_epoch_only_accepts_epoch_checkpoints(self):
        self.assertEqual(parse_checkpoint_epoch(Path("model-12.pt")), 12)
        self.assertIsNone(parse_checkpoint_epoch(Path("model-best.pt")))
        self.assertIsNone(parse_checkpoint_epoch(Path("epoch12_best.pth")))

    def test_list_checkpoint_candidates_sorts_and_filters_epochs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("model-10.pt", "model-2.pt", "model-best.pt", "notes.txt", "model-4.pt"):
                (root / name).write_text("x", encoding="utf-8")

            candidates = list_checkpoint_candidates(root, start_epoch=2, end_epoch=10, every_n_epochs=2)

        self.assertEqual([candidate.epoch for candidate in candidates], [2, 4, 10])

    def test_row_from_results_uses_selection_metric(self):
        row = row_from_results(
            CheckpointCandidate(epoch=3, path=Path("/tmp/model-3.pt")),
            {"num_images": 2, "F1": 0.7, "IoU": 0.5, "AUC": 0.9, "MAE": 0.1},
            "F1",
        )

        self.assertEqual(row["epoch"], 3)
        self.assertAlmostEqual(row["selection_score"], 0.7)

    def test_better_row_prefers_higher_score_then_lower_epoch(self):
        incumbent = {"epoch": 4, "selection_score": 0.8}

        self.assertTrue(better_row({"epoch": 5, "selection_score": 0.9}, incumbent))
        self.assertFalse(better_row({"epoch": 5, "selection_score": 0.7}, incumbent))
        self.assertTrue(better_row({"epoch": 3, "selection_score": 0.8}, incumbent))
        self.assertFalse(better_row({"epoch": 5, "selection_score": 0.8}, incumbent))

    def test_default_best_output_path_uses_requested_epoch_name(self):
        self.assertEqual(default_best_output_path(Path("/runs/main"), 17), Path("/runs/main/epoch17_best.pth"))


if __name__ == "__main__":
    unittest.main()
