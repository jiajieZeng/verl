"""FP32 confident teacher scores must survive parsing and custom backward."""
import math
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'examples/sft/add4'))
from add4_opd_core import parse_teacher_topk
from add4_opd_topk import chunked_topk_loss, topk_terms


def parse_scores(scores, ids, valid):
    # One response position, one padded row; overfetch includes one more ID.
    row = [[score, token, None] for score, token in zip(scores, ids)]
    extra = next(token for token in range(valid + 1) if token not in ids)
    row.append([-1000., extra, None])
    result = {'meta_info': {
        'input_token_logprobs': [[scores[0], ids[0], None]],
        'input_token_ids_logprobs': [[[-1000., valid, None]]],
        'input_top_logprobs': [row],
    }}
    return parse_teacher_topk(result, [1], [ids[0]], valid, valid + 1, len(ids))


@pytest.mark.parametrize('gap', [20, 25, 30, 40])
@pytest.mark.parametrize('k', [1, 128])
def test_confident_real_vocabulary_forward_backward(gap, k):
    valid = 248077
    teacher_logits = torch.zeros(1, valid)
    teacher_logits[0, 0] = gap
    teacher, ids = teacher_logits.log_softmax(-1).topk(k, dim=-1)
    if gap >= 30:
        assert teacher[0, 0] == 0  # Reproduce the reported FP32 saturation.
    # Actual dense teacher rows provide the extra overfetch entry here.
    extra_score = float(teacher_logits.log_softmax(-1)[0, 1])
    row = [[float(score), int(token), None] for score, token in zip(teacher[0], ids[0])]
    extra = next(token for token in range(valid) if token not in ids[0].tolist())
    row.append([extra_score, extra, None])
    parsed = parse_teacher_topk({'meta_info': {
        'input_token_logprobs': [[float(teacher[0, 0]), 0, None]],
        'input_token_ids_logprobs': [[[-1000., valid, None]]],
        'input_top_logprobs': [row],
    }}, [1], [0], valid, valid + 1, k)
    ids = torch.tensor(parsed['ids'])
    teacher = torch.tensor(parsed['logprobs'])
    h = torch.ones(1, 1, requires_grad=True)
    # Concentrate the student's tail on one excluded token. This isolates the
    # teacher rounding case from CPU FP32 softmax summation error over 248K
    # identical small logits (the oracle deliberately uses FP64).
    w = torch.full((valid, 1), -40.)
    w[0] = math.log(.8)
    w[extra] = math.log(.2)
    w.requires_grad_()
    loss = chunked_topk_loss(h, w, ids, teacher, torch.ones(1), valid, 1)[0].sum()
    dh, dw = torch.autograd.grad(loss, (h, w))
    assert torch.isfinite(loss) and torch.isfinite(dh).all() and torch.isfinite(dw).all()

    # Independent dense FP64 oracle: explicit complement and Python mass sum.
    hd = h.detach().double().requires_grad_()
    wd = w.detach().double().requires_grad_()
    lp = (hd @ wd.T).log_softmax(-1)
    keep = torch.ones(valid, dtype=torch.bool)
    keep[ids[0]] = False
    p_tail = lp.exp()[:, keep].sum(-1)
    q_tail = max(1 - math.fsum(math.exp(x) for x in teacher[0].tolist()), 2 ** -24)
    top_lp = lp.gather(-1, ids)
    reference = (top_lp.exp() * (top_lp - teacher.double())).sum()
    reference += (p_tail * (p_tail.log() - math.log(q_tail))).sum()
    expected = torch.autograd.grad(reference, (hd, wd))
    torch.testing.assert_close(loss.double(), reference, atol=2e-5, rtol=2e-5)
    for actual, ref in zip((dh, dw), expected):
        torch.testing.assert_close(actual.double(), ref, atol=2e-5, rtol=2e-4)
    before = w.detach().softmax(0)[keep].sum()
    after = (w.detach() - .1 * dw).softmax(0)[keep].sum()
    assert after < before  # Gradient descent suppresses excess student tail.


@pytest.mark.parametrize('mass_log', [0., 5e-6, 1e-5, 2e-5])
@pytest.mark.parametrize('k', [1, 2])
def test_parser_and_loss_agree_on_rounding_tolerance(mass_log, k):
    scores = [mass_log - math.log(k)] * k
    ids = torch.arange(k)[None]
    teacher = torch.tensor([scores], dtype=torch.double)
    h = torch.ones(1, 1, dtype=torch.double, requires_grad=True)
    w = torch.zeros(4, 1, dtype=torch.double, requires_grad=True)
    if mass_log > 1e-5:
        with pytest.raises(ValueError):
            parse_scores(scores, list(range(k)), 4)
        with pytest.raises(ValueError, match='probability mass'):
            chunked_topk_loss(h, w, ids, teacher, torch.ones(1), 4)
        with pytest.raises(ValueError, match='probability mass'):
            topk_terms(w.T, ids, teacher, 4)
    else:
        parse_scores(scores, list(range(k)), 4)
        result = chunked_topk_loss(h, w, ids, teacher, torch.ones(1), 4)[0].sum()
        grads = torch.autograd.grad(result, (h, w))
        assert torch.isfinite(result) and all(torch.isfinite(g).all() for g in grads)
