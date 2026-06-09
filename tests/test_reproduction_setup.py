from pathlib import Path
import unittest

from omegaconf import OmegaConf
import yaml


ROOT = Path(__file__).resolve().parents[1]


class ReproductionSetupTests(unittest.TestCase):
    def test_default_config_file_exists(self):
        init_utils = (ROOT / "utils" / "init_utils.py").read_text(encoding="utf-8")
        self.assertIn("default='./config/DcDsDiff_352x352.yaml'", init_utils)
        self.assertTrue((ROOT / "config" / "DcDsDiff_352x352.yaml").exists())

    def test_dataset_config_matches_loader_arguments(self):
        config = yaml.safe_load((ROOT / "config" / "dataset_352x352.yaml").read_text(encoding="utf-8"))
        train_params = config["train_dataset"]["params"]
        mix_params = config["test_dataset"]["Mix"]["params"]
        for params in (train_params, mix_params):
            self.assertIn("image_root", params)
            self.assertIn("gt_root", params)
            self.assertIn("de_root", params)
            self.assertIn("trace_root", params)
            self.assertNotIn("depth_root", params)
            self.assertNotIn("text_root", params)

    def test_dataset_config_includes_all_official_test_subsets(self):
        config = yaml.safe_load((ROOT / "config" / "dataset_352x352.yaml").read_text(encoding="utf-8"))

        for dataset_key in ("BN", "PE", "IA", "PP"):
            self.assertIn(dataset_key, config["test_dataset"])
            params = config["test_dataset"][dataset_key]["params"]
            self.assertEqual(params["image_root"], f"/data0/hl/Diff_dataset/Test/Diff/{dataset_key}/f/")
            self.assertEqual(params["gt_root"], f"/data0/hl/Diff_dataset/Test/Diff/{dataset_key}/m/")
            self.assertEqual(params["de_root"], f"/data0/hl/Diff_dataset/Test/Diff/{dataset_key}/d/")
            self.assertEqual(params["trace_root"], f"/data0/hl/Diff_dataset/Test/Diff/{dataset_key}/t/")

    def test_finaltraindata_config_only_changes_dataset_paths(self):
        config = OmegaConf.load(ROOT / "config" / "FinalTrainData_352x352.yaml")
        self.assertEqual(config["__base__"], ["config/DcDsDiff_352x352.yaml"])
        self.assertEqual(config["train_dataset"]["params"]["image_root"], "/data0/hl/FinalTrainData_Diff/train/Diff/f/")
        self.assertEqual(config["test_dataset"]["Mix"]["params"]["gt_root"], "/data0/hl/FinalTrainData_Diff/Test/Diff/Mix/m/")
        self.assertNotIn("optimizer", config)
        self.assertNotIn("model", config)

    def test_model_loads_pretrained_weights_into_trace_backbone(self):
        net_py = (ROOT / "model" / "net.py").read_text(encoding="utf-8")
        self.assertIn("self.backbone_t.load_state_dict", net_py)
        self.assertNotIn("self.backbone_n", net_py)

    def test_msie_is_constructed_with_decoder_channel_count(self):
        net_py = (ROOT / "model" / "net.py").read_text(encoding="utf-8")
        self.assertIn("self.msie = MSIE(embedding_dim // 8)", net_py)

    def test_collate_sequence_annotation_is_python38_compatible(self):
        collate_utils = (ROOT / "utils" / "collate_utils.py").read_text(encoding="utf-8")
        self.assertIn("Sequence as ABCSequence", collate_utils)
        self.assertIn("from typing import Callable, Type, Union, List, Set, Sequence", collate_utils)
        self.assertIn("isinstance(batch[0], ABCSequence)", collate_utils)

    def test_huggingface_cache_ref_exists(self):
        ref = ROOT / "pretrained_weights" / "models--Anonymity--pvt_pretrained" / "refs" / "main"
        self.assertTrue(ref.exists())
        self.assertEqual(ref.read_text().strip(), "a11fd1f27a892002dde9a2050e423ce6c3491bb4")

    def test_tracker_config_converts_omegaconf_to_plain_container(self):
        from utils.trainer import tracker_config_from_cfg

        cfg = OmegaConf.create({"num_workers": 1, "train": {"batch_size": 6}})
        tracker_config = tracker_config_from_cfg(cfg)

        self.assertEqual(tracker_config, {"num_workers": 1, "train": {"batch_size": 6}})
        self.assertIsInstance(tracker_config, dict)
        self.assertIsInstance(tracker_config["train"], dict)

    def test_rtx_4000_nccl_defaults_are_set_before_accelerate(self):
        init_env = (ROOT / "utils" / "init_env.py").read_text(encoding="utf-8")
        self.assertIn('os.environ.setdefault("NCCL_P2P_DISABLE", "1")', init_env)
        self.assertIn('os.environ.setdefault("NCCL_IB_DISABLE", "1")', init_env)

        eval_script = (ROOT / "tools" / "evaluate_checkpoint_f1_iou.py").read_text(encoding="utf-8")
        self.assertLess(
            eval_script.index('os.environ.setdefault("NCCL_P2P_DISABLE", "1")'),
            eval_script.index("import torch"),
        )
        self.assertLess(
            eval_script.index('os.environ.setdefault("NCCL_IB_DISABLE", "1")'),
            eval_script.index("import torch"),
        )


if __name__ == "__main__":
    unittest.main()
