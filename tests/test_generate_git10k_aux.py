import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.generate_git10k_aux import (
    ImageMaskPair,
    collect_images,
    make_detail_map,
    make_high_frequency_view,
    mix_output_dirs,
    pair_image_mask_files,
    prefix_from_stem,
    split_project_pairs,
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

    def test_prefix_from_stem_uses_official_letter_category(self):
        self.assertEqual(prefix_from_stem("PE001"), "PE")
        self.assertEqual(prefix_from_stem("BN_001"), "BN")
        self.assertEqual(prefix_from_stem("IA-12"), "IA")
        self.assertEqual(prefix_from_stem("PP sample"), "PP")

    def test_mix_output_dirs_points_to_project_mix_folder(self):
        dirs = mix_output_dirs(Path("/data/Diff_dataset"), "Mix")

        self.assertEqual(dirs["f"], Path("/data/Diff_dataset/Test/Diff/Mix/f"))
        self.assertEqual(dirs["m"], Path("/data/Diff_dataset/Test/Diff/Mix/m"))
        self.assertEqual(dirs["d"], Path("/data/Diff_dataset/Test/Diff/Mix/d"))
        self.assertEqual(dirs["t"], Path("/data/Diff_dataset/Test/Diff/Mix/t"))

    def test_global_split_mode_uses_overall_9_to_1_split(self):
        pairs = [
            ImageMaskPair(
                stem=f"sample_{index:02d}",
                image_path=Path(f"images/sample_{index:02d}.png"),
                mask_path=Path(f"masks/sample_{index:02d}.png"),
                prefix=f"sample_{index:02d}",
            )
            for index in range(10)
        ]

        assignment = split_project_pairs(pairs, train_ratio=0.9, split_mode="global", seed=7)

        self.assertEqual(sum(split == "train" for split in assignment.values()), 9)
        self.assertEqual(sum(split == "test" for split in assignment.values()), 1)
        self.assertEqual(set(assignment), {pair.stem for pair in pairs})

    def test_prefix_split_mode_keeps_existing_per_prefix_behavior(self):
        pairs = [
            ImageMaskPair(
                stem=f"A_{index}",
                image_path=Path(f"Image/A_{index}.png"),
                mask_path=Path(f"Mask/A_{index}.png"),
                prefix="A",
            )
            for index in range(10)
        ]

        assignment = split_project_pairs(pairs, train_ratio=0.9, split_mode="prefix", seed=7)

        self.assertEqual(sum(split == "train" for split in assignment.values()), 9)
        self.assertEqual(sum(split == "test" for split in assignment.values()), 1)

    def test_collect_images_reports_missing_folder(self):
        missing = Path("/definitely/missing/GIT10K/Image")

        with self.assertRaises(SystemExit) as ctx:
            collect_images(missing)

        self.assertIn("Input folder does not exist", str(ctx.exception))
        self.assertIn(str(missing), str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
