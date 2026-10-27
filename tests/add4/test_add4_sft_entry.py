import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('add4_sft_entry', ROOT / 'examples/sft/add4/train_add4.py')
entry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(entry)


def test_fused_head_remains_in_root_and_other_targets_unchanged():
    head, embedding, layer = object(), object(), object()
    model = SimpleNamespace(lm_head=head, config=SimpleNamespace(model_type='qwen3_5'))
    calls = []
    def selector(model_arg, classes):
        calls.append((model_arg, classes))
        return [embedding, head, layer]
    classes = ['Qwen3_5DecoderLayer']
    assert entry.root_owned_lm_head(selector)(model, classes) == [embedding, layer]
    assert calls == [(model, classes)]


def test_selector_errors_propagate():
    model = SimpleNamespace(lm_head=object(), config=SimpleNamespace(model_type='qwen3_5'))
    def broken(*args):
        raise RuntimeError('wrap failure')
    with pytest.raises(RuntimeError, match='wrap failure'):
        entry.root_owned_lm_head(broken)(model, [])


def test_wrong_model_rejected():
    model = SimpleNamespace(config=SimpleNamespace(model_type='other'))
    with pytest.raises(ValueError, match='requires Qwen3.5'):
        entry.root_owned_lm_head(lambda *args: [])(model, [])


def loader_fixture(train_count=4000, val_count=220, dp_size=1, train_batch=32):
    from torch.utils.data import DataLoader, SequentialSampler
    obj = SimpleNamespace(
        config=SimpleNamespace(data=SimpleNamespace(train_batch_size=train_batch, validation_batch_size=4, num_workers=0)),
        engine=SimpleNamespace(get_data_parallel_size=lambda: dp_size),
        train_dataset=list(range(train_count)), val_dataset=list(range(val_count)),
        collate_fn=lambda rows: rows,
    )
    def build(self):
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=32, drop_last=True, collate_fn=self.collate_fn)
        self.val_sampler = SequentialSampler(self.val_dataset)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=32, sampler=self.val_sampler,
                                        drop_last=True, collate_fn=self.collate_fn)
    return obj, entry.independent_validation_batch(build)


def test_all_train_examples_each_of_three_epochs_and_all_validation_examples():
    obj, build = loader_fixture()
    build(obj)
    for _ in range(3):
        batches = list(obj.train_dataloader)
        assert len(batches) == 125
        assert [row for batch in batches for row in batch] == list(range(4000))
    assert len(obj.val_dataloader) == 55
    assert obj.val_dataloader.batch_size == 4 and not obj.val_dataloader.drop_last
    assert [row for batch in obj.val_dataloader for row in batch] == list(range(220))


def test_validation_partial_batch_is_retained():
    obj, build = loader_fixture(val_count=221)
    build(obj)
    assert [row for batch in obj.val_dataloader for row in batch] == list(range(221))


def test_batch128_retains_all4000_with_final32_each_epoch():
    obj, build = loader_fixture(train_batch=128, val_count=398)
    build(obj)
    for _ in range(3):
        batches = list(obj.train_dataloader)
        assert len(batches) == 32
        assert [len(b) for b in batches] == [128] * 31 + [32]
        assert [row for batch in batches for row in batch] == list(range(4000))
    assert [row for batch in obj.val_dataloader for row in batch] == list(range(398))


def test_distributed_sampler_tails_are_rejected():
    obj, build = loader_fixture(train_count=4001, dp_size=4)
    with pytest.raises(ValueError, match='refusing sample loss'):
        build(obj)
    obj, build = loader_fixture(val_count=221, dp_size=4)
    with pytest.raises(ValueError, match='refusing incomplete validation'):
        build(obj)


def test_baseline_validates_all_batches_before_fit(monkeypatch, capsys):
    import sys
    from types import ModuleType
    import torch
    verl, utils, tu = [ModuleType(n) for n in ['verl', 'verl.utils', 'verl.utils.tensordict_utils']]
    verl.utils = utils
    utils.tensordict_utils = tu
    tu.get_tensordict = lambda tensor_dict, non_tensor_dict: tensor_dict
    tu.get = lambda output, key: output[key]
    for mod in [verl, utils, tu]:
        monkeypatch.setitem(sys.modules, mod.__name__, mod)
    calls = []
    monkeypatch.setattr(torch.distributed, 'barrier', lambda: calls.append('barrier'))
    def infer(batch):
        calls.append(batch)
        return {'metrics': {'loss': float(batch)}}
    obj = SimpleNamespace(
        config=SimpleNamespace(trainer={'validate_before_training': True},
            model=SimpleNamespace(use_remove_padding=True),
            data=SimpleNamespace(use_dynamic_bsz=True, max_token_len_per_gpu=65536,
                                 micro_batch_size_per_gpu=1, pad_mode='no_padding')),
        global_batch_size=128, model_config=SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0)),
        engine=SimpleNamespace(is_mp_src_rank_with_outputs=lambda: True,
                               get_data_parallel_rank=lambda: 0, get_data_parallel_group=lambda: None),
        val_dataloader=[1, 3], training_client=SimpleNamespace(infer_batch=infer),
        device_name='cpu', resume_global_step=0,
    )
    entry.validation_before_training(lambda self: calls.append('fit'))(obj)
    assert calls == [1, 3, 'barrier', 'fit']
    assert 'step:0 - val/loss:2.0' in capsys.readouterr().out
    def fail(batch):
        raise RuntimeError('validation failed')
    obj.training_client.infer_batch = fail
    calls.clear()
    with pytest.raises(RuntimeError, match='validation failed'):
        entry.validation_before_training(lambda self: calls.append('fit'))(obj)
    assert calls == []
