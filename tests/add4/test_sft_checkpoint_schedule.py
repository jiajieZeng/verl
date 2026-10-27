"""Run the real SFT loop with lightweight workers to check save boundaries."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


class TrainerConfig(dict):
    __getattr__ = dict.__getitem__


class CheckpointScheduleTest(unittest.TestCase):
    def saved_steps(self, frequency, epoch_end=False, resume_step=0, total_steps=375):
        path = Path(__file__).resolve().parents[2] / 'verl/trainer/sft_trainer.py'
        tree = ast.parse(path.read_text())
        trainer = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SFTTrainer')
        fit = next(n for n in trainer.body if isinstance(n, ast.FunctionDef) and n.name == 'fit')
        namespace = dict(
            log_with_rank=Mock(), logger=Mock(), aggressive_empty_cache=Mock(),
            log_gpu_memory_usage=Mock(), tqdm=lambda data, **kwargs: data,
            NonTensorData=lambda value: value,
            tu=SimpleNamespace(get_tensordict=Mock(return_value={}), assign_non_tensor=Mock()),
        )
        exec(compile(ast.Module(body=[fit], type_ignores=[]), str(path), 'exec'), namespace)
        saved = Mock()
        state = SimpleNamespace(
            config=SimpleNamespace(
                trainer=TrainerConfig(total_epochs=3, save_on_epoch_end=epoch_end),
                model=SimpleNamespace(use_remove_padding=True),
                data=SimpleNamespace(use_dynamic_bsz=True, max_token_len_per_gpu=65536,
                                     micro_batch_size_per_gpu=1, pad_mode='no_padding'),
            ),
            engine=SimpleNamespace(is_mp_src_rank_with_outputs=lambda: False),
            resume_global_step=resume_step, steps_per_epoch=125,
            total_training_steps=total_steps, global_batch_size=32,
            model_config=SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0)),
            train_sampler=Mock(), train_dataloader=range(125), val_dataloader=None,
            rank=0, training_client=Mock(), start_profile_step=-1, end_profile_step=-1,
            _get_batch_seqlens=lambda **kwargs: [1], test_freq=-1, save_freq=frequency,
            ckpt_handler=SimpleNamespace(save_checkpoint=saved),
        )
        namespace['fit'](state)
        return [call.kwargs['step'] for call in saved.call_args_list]

    def test_frequent_saves_include_epoch_boundaries_without_duplicates(self):
        self.assertEqual(self.saved_steps(10, True), sorted(set(range(10, 376, 10)) | {125, 250, 375}))

    def test_twenty_step_local_schedule(self):
        self.assertEqual(self.saved_steps(20, True), sorted(set(range(20, 376, 20)) | {125, 250, 375}))

    def test_existing_default_schedules_are_preserved(self):
        self.assertEqual(self.saved_steps(125), [125, 250, 375])
        self.assertEqual(self.saved_steps(-1), [375])
        self.assertEqual(self.saved_steps(10), list(range(10, 375, 10)) + [375])

    def test_resumed_run_uses_global_boundaries_and_saves_early_final_step(self):
        self.assertEqual(self.saved_steps(10, True, resume_step=120, total_steps=132), [125, 130, 132])


if __name__ == '__main__':
    unittest.main()
