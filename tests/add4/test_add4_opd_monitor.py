import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'examples/sft/add4'))
from add4_opd_monitor import OPDMonitor, MonitoredTracking, facts_valid


@pytest.mark.parametrize('text,expected', [('{"facts":[]}', True), ('{"facts":["a"]}', True),
    ('{"facts":"a"}', False), ('{"facts":[1]}', False), ('```json\n{}\n```', False)])
def test_add_format(text, expected):
    assert facts_valid(text) is expected


def test_failed_generation_is_counted_before_abort(tmp_path):
    from add4_opd_client import SGLangClient
    monitor = OPDMonitor(tmp_path, tokenizer=SimpleNamespace(decode=lambda *a, **k: '{"facts":[]}'))
    client = SGLangClient('http://localhost:1')
    client.monitor = monitor
    client.request = lambda *a, **k: {'meta_info': {'output_token_logprobs': [[-1, 2]], 'finish_reason': {'type': 'length'}}}
    with pytest.raises(ValueError, match='EOS'):
        client.sample([1], dict(max_new_tokens=2, context_length=8, temperature=.6, valid_vocab_size=4,
                               output_vocab_size=6), 3, 'mask', 1)
    snapshot = monitor.snapshot()
    assert snapshot['generation/attempts'] == 1
    assert snapshot['generation/truncated_rate'] == 1
    assert snapshot['generation/eos_rate'] == 0


def test_threaded_http_observations_and_failure_event(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    monitor = OPDMonitor(tmp_path)
    calls = []
    tracker = MonitoredTracking(SimpleNamespace(logger={},
        log=lambda data, step, **kw: calls.append((step, data)), finish=lambda **kw: None), monitor)
    monitor.begin(12)
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(lambda x: monitor.http('teacher', '/generate', .1, 100, x != 99), range(100)))
    tracker.log({'train/loss': .5, 'train/grad_norm': math.inf}, 12)
    tracker.log({'val/loss': .4}, 12)
    tracker.failure(ValueError('no secret exception text should be serialized'))
    assert [x[0] for x in calls] == [0, 1, 2]
    assert all(x[1]['opd/step'] == 12 for x in calls)
    assert calls[0][1]['http/teacher_score_errors'] == 1
    assert calls[0][1]['http/teacher_score_bytes'] == 10000
    assert calls[0][1]['train/skipped_updates_total'] == 1
    records = [json.loads(x) for x in (tmp_path/'metrics.jsonl').read_text().splitlines()]
    assert records[0]['metrics']['train/grad_norm'] is None
    assert records[-1]['metrics']['errors/fatal_total'] == 1
    assert 'secret' not in (tmp_path/'metrics.jsonl').read_text()


def test_tail_floor_metrics_use_only_response_tokens(tmp_path):
    from tensordict import TensorDict
    import verl.utils.tensordict_utils as tu
    from add4_opd_trainer import opd_or_sft_loss
    def nested(x):
        return torch.nested.as_nested_tensor([torch.tensor(x)], layout=torch.jagged)
    data = TensorDict({'input_ids': nested([0, 1, 2]), 'loss_mask': nested([0, 1, 1]),
                       'opd_teacher_ids': nested([[0], [0], [0]]),
                       'opd_teacher_logp': nested([[100.], [0.], [-1.]])}, batch_size=[1])
    tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=2)
    _, metrics = opd_or_sft_loss(None, {'opd_losses': nested([1., 1., 0.]),
                                      'opd_student_mass': nested([.8, .6, 0.])}, data)
    assert metrics['opd/teacher_tail_floor_rate'] == .5
    assert metrics['opd/student_tail_mass'] == pytest.approx(.3)
    assert metrics['opd/teacher_tail_mass'] == pytest.approx((2**-24 + 1-math.exp(-1))/2)


def test_all_398_validation_rows_survive_eight_rank_configuration():
    from add4_opd_monitor_runtime import build_loaders
    for rank in range(8):
        trainer = SimpleNamespace(val_dataset=list(range(398)), config=SimpleNamespace(
            data=SimpleNamespace(num_workers=0)), collate_fn=lambda x: x)
        def build():
            assert trainer.val_dataset is None
        build_loaders(trainer, build)
        assert [x[0] for x in trainer.val_dataloader] == list(range(398))


def test_monitor_profiles_compose(monkeypatch):
    from hydra import compose, initialize_config_dir
    root = Path(__file__).resolve().parents[2]
    for key, value in dict(VERL_ROOT=str(root), QWEN35_MODEL_DIR='/model', QWEN36_MODEL_DIR='/model',
                           ADD_TOKENIZER_DIR='/tokenizer', ADD_TRAIN_FILES='/train', ADD_RUN_ID='test',
                           ADD_RUN_DIR='/run').items():
        monkeypatch.setenv(key, value)
    for model in ('qwen35', 'qwen36'):
        with initialize_config_dir(config_dir=str(root/'examples/sft/add4'), version_base=None):
            config = compose(config_name=f'{model}.opd.monitor')
            original = compose(config_name=f'{model}.sft.example')
        for key in ('model', 'engine', 'optim', 'checkpoint'):
            assert config[key] == original[key]
        assert config.trainer.logger == ['console', 'wandb']
        assert config.trainer.validate_before_training
        assert config.trainer.test_freq == 50
        assert config.data.val_files.endswith('validation-188.parquet')


def test_resources_do_not_report_stale_values_as_current(tmp_path):
    import time
    from add4_opd_resources import gpu_metrics
    values = gpu_metrics('0, 90, 500, 1000, 60\n1, N/A, 400, 1000, 55', 'teacher_host')
    assert values['teacher_host/gpu0/utilization_pct'] == 90
    assert 'teacher_host/gpu1/utilization_pct' not in values
    path = tmp_path/'resources.json'
    path.write_text(json.dumps({'sample_time': time.time()-200, 'metrics': values}))
    monitor = OPDMonitor(tmp_path/'monitor')
    monitor.resources(path)
    assert monitor.snapshot()['resources/snapshot_fresh'] == 0
    assert 'teacher_host/gpu0/utilization_pct' not in monitor.snapshot()
    with pytest.raises(ValueError, match='Start'):
        monitor.resources(path, required=True)


def test_evaluation_counts_and_checkpoint_are_checked():
    from log_opd_evaluation import evaluation_metrics
    request = dict(checkpoint_step=50, checkpoint='/run/global_step_50')
    result = dict(request, datasets={'pro': dict(tasks=2, attempts=8, successes=3, solved_tasks=1)})
    metrics = evaluation_metrics(request, result)
    assert metrics['eval/all/avg_at_4'] == 3/8
    assert metrics['eval/all/pass_at_4'] == .5
    with pytest.raises(ValueError, match='step'):
        evaluation_metrics(request, dict(result, checkpoint_step=100))
    result['datasets']['pro']['attempts'] = 7
    with pytest.raises(ValueError, match='four completed'):
        evaluation_metrics(request, result)


@pytest.mark.parametrize('fail', [False, True])
def test_monitoring_fit_lifecycle_restores_hooks_and_keeps_checkpoint_queue(tmp_path, monkeypatch, fail):
    import torch.distributed as dist
    from omegaconf import OmegaConf
    import verl.utils.tensordict_utils as tu
    from add4_opd_monitor_runtime import run_monitored
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    dist.init_process_group('gloo', init_method=f'file://{tmp_path}/gloo', rank=0, world_size=1)
    calls, finishes = [], []
    class Tracking:
        logger = {}
        def __init__(self, **kwargs): pass
        def log(self, data, step): calls.append((step, data))
        def finish(self, exit_code=0): finishes.append(exit_code)
    sft = SimpleNamespace(Tracking=Tracking)
    def original_train(**kwargs):
        return tu.get_tensordict({}, {'metrics': {'loss': 1., 'grad_norm': 2.}})
    saved = []
    def save(step): saved.append(step)
    trainer = SimpleNamespace(
        config=OmegaConf.create({'opd': {'run_id': 'smoke'}, 'trainer': {
            'default_local_dir': str(tmp_path/'checkpoints'), 'logger': ['console'],
            'project_name': 'test', 'experiment_name': 'test'}, 'optim': {'clip_grad': 1.}}),
        engine=SimpleNamespace(is_mp_src_rank_with_outputs=lambda: True, get_data_parallel_rank=lambda: 0),
        training_client=SimpleNamespace(train_batch=original_train),
        ckpt_handler=SimpleNamespace(save_checkpoint=save), model_config=SimpleNamespace(tokenizer=None),
        val_dataset=None, resume_global_step=0)
    def fit():
        tracking = sft.Tracking()
        trainer._opd_monitor.begin(1)
        result = trainer.training_client.train_batch()
        metrics = tu.get(result, 'metrics')
        tracking.log({'train/loss': metrics['loss'], 'train/grad_norm': metrics['grad_norm']}, 1)
        trainer.ckpt_handler.save_checkpoint(step=1)
        if fail:
            raise ValueError('intentional test failure')
    try:
        if fail:
            with pytest.raises(ValueError): run_monitored(trainer, sft, fit)
        else:
            run_monitored(trainer, sft, fit)
        assert sft.Tracking is Tracking
        assert trainer.training_client.train_batch is original_train
        assert trainer.ckpt_handler.save_checkpoint is save
        assert saved == [1]
        assert finishes == [int(fail)]
        queue = list(tmp_path.rglob('eval-queue/step-00000001.json'))
        assert len(queue) == 1
        assert json.loads(queue[0].read_text())['checkpoint'] == str(tmp_path/'checkpoints/global_step_1')
        assert all(data['opd/step'] == 1 for _, data in calls)
    finally:
        dist.destroy_process_group()


def _startup_failure_worker(rank, root):
    import datetime
    import torch.distributed as dist
    from omegaconf import OmegaConf
    from add4_opd_monitor_runtime import run_monitored
    root = Path(root)
    dist.init_process_group('gloo', init_method=f'file://{root}/dist', rank=rank, world_size=2,
                            timeout=datetime.timedelta(seconds=20))
    class Tracking:
        logger = {}
        def __init__(self, **kwargs): pass
        def log(self, *args, **kwargs): pass
        def finish(self, **kwargs): pass
    trainer = SimpleNamespace(config=OmegaConf.create({
        'opd': {'run_id': 'startup-test'}, 'trainer': {
            'default_local_dir': str(root/'checkpoints'), 'logger': ['console'],
            'project_name': 'test', 'experiment_name': 'test', 'require_resource_snapshot': True}}),
        engine=SimpleNamespace(is_mp_src_rank_with_outputs=lambda: True, get_data_parallel_rank=lambda: rank),
        training_client=SimpleNamespace(train_batch=lambda **kw: None),
        ckpt_handler=SimpleNamespace(save_checkpoint=lambda **kw: None),
        model_config=SimpleNamespace(tokenizer=None), val_dataset=None, resume_global_step=0)
    try:
        try:
            run_monitored(trainer, SimpleNamespace(Tracking=Tracking), lambda: None)
        except RuntimeError as error:
            (root/f'rank-{rank}.txt').write_text(str(error))
        else:
            raise AssertionError('Missing resource snapshot must stop all ranks')
    finally:
        dist.destroy_process_group()


def test_monitor_startup_failure_reaches_every_rank(tmp_path):
    import torch.multiprocessing as mp
    mp.spawn(_startup_failure_worker, args=(str(tmp_path),), nprocs=2, join=True)
    for rank in range(2):
        assert 'startup failed on rank 0' in (tmp_path/f'rank-{rank}.txt').read_text()
