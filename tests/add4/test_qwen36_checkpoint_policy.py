"""Exercise the Add Qwen3.6 checkpoint opt-out without loading model weights."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class CheckpointPolicyTest(unittest.TestCase):
    def trainer(self, save_freq):
        path = Path(__file__).resolve().parents[2] / "examples/sft/add4/train_add4_qwen36.py"
        spec = importlib.util.spec_from_file_location("qwen36_checkpoint_entry", path)
        entry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
        save = Mock()
        load = Mock(return_value=0)

        class Base:
            def _build_ckpt_handler(self):
                self.ckpt_handler = SimpleNamespace(save_checkpoint=save, load_checkpoint=load)

        sft = SimpleNamespace(SFTTrainer=Base)
        with patch.dict(sys.modules, {"add4_qwen_models": SimpleNamespace(validate_student_model=Mock())}):
            entry.install_moe_sft(sft)
        trainer = sft.SFTTrainer()
        trainer.config = SimpleNamespace(trainer=SimpleNamespace(save_freq=save_freq))
        trainer._build_ckpt_handler()
        return trainer, save, load

    def test_disabled_blocks_smoke_epoch_and_unconditional_final_save(self):
        trainer, save, load = self.trainer(-1)
        for step in (1, 125, 250, 375):
            trainer.ckpt_handler.save_checkpoint(step=step)
        save.assert_not_called()
        self.assertEqual(trainer.ckpt_handler.load_checkpoint(), 0)
        load.assert_called_once()

    def test_enabled_policies_keep_original_handler(self):
        for frequency in (125, "after_each_epoch"):
            with self.subTest(frequency=frequency):
                trainer, save, _ = self.trainer(frequency)
                trainer.ckpt_handler.save_checkpoint(step=375)
                save.assert_called_once_with(step=375)


if __name__ == "__main__":
    unittest.main()
