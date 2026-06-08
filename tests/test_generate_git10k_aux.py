import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.generate_git10k_aux import (
    collect_images,
    make_detail_map,
    make_high_frequency_view,
    mix_output_dirs,
    pair_image_mask_files,
)


class GenerateGit10KAuxTests(unittest.TestCase):
    def test_detail_map_emphasizes_mask_boundary(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[16:48, 16:48] = 255

        detail = make_detail_map(mask, radius=8)

        self.assertEqual(detail.shape, mask.shape)
        self.assertEqual(detail.dtype, np.uint8)
        self.assertGreater(detail[16, 32], detail[32, 32])
        self.assertEqual(int(detail[0, 0]), 0)

    def test_high_frequency_view_keeps_edge_response(self):
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[:, 32:] = 255

        trace = make_high_frequency_view(image, cutoff_ratio=0.5, boost=10.0)

        self.assertEqual(trace.shape, image.shape)
        self.assertEqual(trace.dtype, np.uint8)
        self.assertGreater(trace[:, 30:34].mean(), trace[:, 8:16].mean())

    def test_pair_image_mask_files_matches_by_stem(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_root = root / "Image"
            mask_root = root / "Mask"
            image_root.mkdir()
            mask_root.mkdir()
            Image.new("RGB", (4, 4)).save(image_root / "A.png")
            Image.new("RGB", (4, 4)).save(image_root / "B.jpg")
            Image.new("L", (4, 4)).save(mask_root / "A.png")
            Image.new("L", (4, 4)).save(mask_root / "C.png")

            pairs = pair_image_mask_files(image_root, mask_root)

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].stem, "A")

    def test_mix_output_dirs_points_to_project_mix_folder(self):
        dirs = mix_output_dirs(Path("/data/Diff_dataset"), "Mix")

        self.assertEqual(dirs["f"], Path("/data/Diff_dataset/Test/Diff/Mix/f"))
        self.assertEqual(dirs["m"], Path("/data/Diff_dataset/Test/Diff/Mix/m"))
        self.assertEqual(dirs["d"], Path("/data/Diff_dataset/Test/Diff/Mix/d"))
        self.assertEqual(dirs["t"], Path("/data/Diff_dataset/Test/Diff/Mix/t"))

    def test_collect_images_reports_missing_folder(self):
        missing = Path("/definitely/missing/GIT10K/Image")

        with self.assertRaises(SystemExit) as ctx:
            collect_images(missing)

        self.assertIn("Input folder does not exist", str(ctx.exception))
        self.assertIn(str(missing), str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
