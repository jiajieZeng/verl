import importlib.util
from pathlib import Path

import pytest
import torch

source = Path(__file__).resolve().parents[2] / 'examples/sft/add4/add4_streaming_adamw.py'
spec = importlib.util.spec_from_file_location('add4_streaming_adamw', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
StreamingAdamW = module.StreamingAdamW


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable'))])
@pytest.mark.parametrize('amsgrad', [False, True])
def test_updates_match_adamw_with_groups_missing_grads_and_schedule(device, amsgrad):
    torch.manual_seed(735)
    actual = [torch.nn.Parameter(torch.randn(9, 7, device=device)) for _ in range(3)]
    expected = [torch.nn.Parameter(p.detach().clone()) for p in actual]
    groups = lambda ps: [{'params': ps[:2], 'lr': .01}, {'params': ps[2:], 'lr': .003, 'weight_decay': .2}]
    candidate = StreamingAdamW(groups(actual), amsgrad=amsgrad, weight_decay=.07)
    reference = torch.optim.AdamW(groups(expected), amsgrad=amsgrad, weight_decay=.07, foreach=False, fused=False)
    group_objects = candidate.param_groups
    schedulers = [torch.optim.lr_scheduler.StepLR(o, step_size=1, gamma=.9) for o in (candidate, reference)]
    for step in range(4):
        for i, (p, q) in enumerate(zip(actual, expected)):
            p.grad = torch.randn_like(p) if (step + i) % 3 else None
            q.grad = p.grad.clone() if p.grad is not None else None
        candidate.step()
        reference.step()
        assert candidate.param_groups is group_objects
        for p, q in zip(actual, expected):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
            for key, value in candidate.state[p].items():
                torch.testing.assert_close(value.cpu(), reference.state[q][key].cpu(), rtol=0, atol=0)
                assert value.device.type == 'cpu'
        for scheduler in schedulers:
            scheduler.step()


def test_rejects_precision_change_and_restores_groups():
    p = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
    p.grad = torch.ones_like(p)
    optimizer = StreamingAdamW([p])
    groups = optimizer.param_groups
    with pytest.raises(ValueError, match='FP32'):
        optimizer.step()
    assert optimizer.param_groups is groups


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_resume_keeps_moments_on_cpu_before_next_forward(device):
    import copy
    parameter = torch.nn.Parameter(torch.tensor([1., 2., 3.], device=device))
    original = StreamingAdamW([parameter], lr=.01, amsgrad=True)
    parameter.grad = torch.tensor([.2, -.1, .4], device=device)
    original.step()
    restored_parameter = torch.nn.Parameter(parameter.detach().clone())
    restored = StreamingAdamW([restored_parameter], lr=.01, amsgrad=True)
    restored.load_state_dict(copy.deepcopy(original.state_dict()))
    for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
        assert restored.state[restored_parameter][key].device.type == 'cpu'
    for optimizer, value in ((original, parameter), (restored, restored_parameter)):
        value.grad = torch.tensor([-.3, .7, .2], device=device)
        optimizer.step()
    torch.testing.assert_close(parameter, restored_parameter, rtol=0, atol=0)
