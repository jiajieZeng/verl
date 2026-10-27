import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

SOURCE = Path(__file__).resolve().parents[2] / 'examples/sft/add4/add4_checkpoint_retention.py'
spec = importlib.util.spec_from_file_location('add4_retention_tested', SOURCE)
retention = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retention)


class RetentionTest(unittest.TestCase):
    def checkpoint(self, root, step):
        path = root / f'global_step_{step}'
        path.mkdir()
        for prefix in ('model', 'optim', 'extra_state'):
            for rank in range(4):
                (path / f'{prefix}_world_size_4_rank_{rank}.pt').write_bytes(b'state')
        (path / 'data_0.pt').write_bytes(b'dataloader')
        return path

    def test_latest_one_preserves_epochs_and_previous_on_failed_save(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for step in (40, 60, 80, 100, 120, 125, 140):
                self.checkpoint(root, step)
                report = retention.complete_and_prune(root, step, 125, 1, 4)
                expected = sorted({step} | ({125} if step >= 125 else set()))
                self.assertEqual(report['retained'], expected)
            incomplete = self.checkpoint(root, 160)
            (incomplete / 'optim_world_size_4_rank_3.pt').unlink()
            with self.assertRaises(ValueError):
                retention.complete_and_prune(root, 160, 125, 1, 4)
            self.assertTrue((root / 'global_step_140').is_dir())
            self.assertTrue((root / 'global_step_125').is_dir())

    def test_schedule_keeps_epochs_and_latest_two_after_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            schedule = sorted(set(range(20, 376, 20)) | {125, 250, 375})
            seen = []
            for step in schedule:
                self.checkpoint(root, step)
                seen.append(step)
                report = retention.complete_and_prune(root, step, 125, 2, 4)
                expected = sorted(set(seen[-2:]) | {x for x in seen if x % 125 == 0})
                self.assertEqual(report['retained'], expected)
                self.assertEqual(sorted(int(p.name.rsplit('_', 1)[1]) for p in root.iterdir()), expected)
            self.assertEqual(report['retained'], [125, 250, 360, 375])

    def test_incomplete_save_preserves_previous_checkpoints_and_unmarked_dirs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for step in (20, 40):
                self.checkpoint(root, step)
                retention.complete_and_prune(root, step, 125, 2, 4)
            incomplete = self.checkpoint(root, 60)
            (incomplete / 'optim_world_size_4_rank_3.pt').unlink()
            with self.assertRaises(ValueError):
                retention.complete_and_prune(root, 60, 125, 2, 4)
            self.assertTrue((root / 'global_step_20').exists())
            self.checkpoint(root, 80)
            retention.complete_and_prune(root, 80, 125, 2, 4)
            self.assertTrue(incomplete.exists())
            self.assertFalse((root / 'global_step_20').exists())

    def test_symlink_outside_run_is_never_pruned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'run'
            root.mkdir()
            outside = Path(directory) / 'outside'
            outside.mkdir()
            (root / 'global_step_1').symlink_to(outside, target_is_directory=True)
            self.checkpoint(root, 20)
            retention.complete_and_prune(root, 20, 125, 2, 4)
            self.assertTrue(outside.exists())
            self.assertTrue((root / 'global_step_1').is_symlink())

    def test_failed_save_never_runs_retention(self):
        handler = SimpleNamespace(max_ckpt_to_keep=2, default_local_dir='/unused',
                                  save_checkpoint=Mock(side_effect=RuntimeError('save failed')))
        retention.install_epoch_retention(handler, 125, 2)
        self.assertIsNone(handler.max_ckpt_to_keep)
        with patch.object(retention, 'complete_and_prune') as prune:
            with self.assertRaisesRegex(RuntimeError, 'save failed'):
                handler.save_checkpoint(60)
            prune.assert_not_called()

    def test_successful_handler_save_prunes_only_after_all_files_exist(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            handler = SimpleNamespace(max_ckpt_to_keep=2, default_local_dir=root,
                                      save_checkpoint=lambda step: self.checkpoint(root, step))
            retention.install_epoch_retention(handler, 125, 2)
            with patch('torch.distributed.get_rank', return_value=0), \
                    patch('torch.distributed.get_world_size', return_value=4), \
                    patch('torch.distributed.broadcast_object_list'):
                for step in (100, 120, 125, 140, 160):
                    handler.save_checkpoint(step)
            self.assertEqual(sorted(p.name for p in root.iterdir()),
                             ['global_step_125', 'global_step_140', 'global_step_160'])

    def test_optimizer_state_roundtrip_preserves_next_update(self):
        import torch
        spec = importlib.util.spec_from_file_location('streaming_for_resume_test',
                                                      SOURCE.parent / 'add4_streaming_adamw.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        parameter = torch.nn.Parameter(torch.tensor([1., 2., 3.]))
        original = module.StreamingAdamW([parameter], lr=.01)
        parameter.grad = torch.tensor([.2, -.1, .4])
        original.step()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'optim.pt'
            torch.save(original.state_dict(), checkpoint)
            restored_parameter = torch.nn.Parameter(parameter.detach().clone())
            restored = module.StreamingAdamW([restored_parameter], lr=.01)
            restored.load_state_dict(torch.load(checkpoint, weights_only=False))
        for optimizer, value in ((original, parameter), (restored, restored_parameter)):
            value.grad = torch.tensor([-.3, .7, .2])
            optimizer.step()
        torch.testing.assert_close(parameter, restored_parameter, rtol=0, atol=0)
        for key in ('step', 'exp_avg', 'exp_avg_sq'):
            torch.testing.assert_close(original.state[parameter][key],
                                       restored.state[restored_parameter][key], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
