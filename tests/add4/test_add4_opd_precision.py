"""Independent objective and low-precision regressions for both custom heads."""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'examples/sft/add4'))
from add4_opd_full import chunked_full_loss
from add4_opd_topk import chunked_topk_loss, topk_terms


def test_tail_bucket_removes_incentive_to_move_mass_out_of_teacher_support():
    # Internal ratios agree but p(K) is already too small: partial reverse KL
    # nevertheless pushes it down further (its optimum is q(K)/e).
    ids = torch.tensor([[0, 1]])
    q = torch.tensor([[.6, .3, .1]], dtype=torch.double)
    logits = torch.tensor([[.4, .2, .4]], dtype=torch.double).log().requires_grad_()
    p = logits.softmax(-1)
    partial = (p[:, :2] * (p[:, :2].log() - q[:, :2].log())).sum()
    old_grad, = torch.autograd.grad(partial, logits, retain_graph=True)
    old_mass = (logits - .1 * old_grad).softmax(-1)[:, :2].sum()
    loss = topk_terms(logits, ids, q[:, :2].log(), 3)[0].sum()
    grad, = torch.autograd.grad(loss, logits)
    new_mass = (logits - .1 * grad).softmax(-1)[:, :2].sum()
    assert old_mass < .6 < new_mass


@pytest.mark.parametrize('k', [2, 4])
def test_coarsened_distribution_has_zero_kl_and_gradient_at_teacher(k):
    q = torch.tensor([[.4, .3, .2, .1]], dtype=torch.double)
    logits = q.log().requires_grad_()
    ids = torch.arange(k)[None]
    loss = topk_terms(logits, ids, q[:, :k].log(), 4)[0].sum()
    grad, = torch.autograd.grad(loss, logits)
    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-14, rtol=0)
    torch.testing.assert_close(grad, torch.zeros_like(grad), atol=1e-14, rtol=0)


def test_tiny_tail_and_saturated_student_remain_finite():
    # FP32 sum(exp(logq_K)) rounds to one, but the supplied logprob still
    # contains a recoverable 1e-9 tail. Student complement must not use 1-p_K.
    q = torch.tensor([[-1e-9]], dtype=torch.float32)
    logits = torch.tensor([[0., -100.]], requires_grad=True)
    loss = topk_terms(logits, torch.tensor([[0]]), q, 2)[0].sum()
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    rounded_loss = topk_terms(logits, torch.tensor([[0]]), torch.zeros_like(q), 2)[0].sum()
    rounded_grad, = torch.autograd.grad(rounded_loss, logits)
    assert torch.isfinite(rounded_loss) and torch.isfinite(rounded_grad).all()


@pytest.mark.parametrize('objective', ['topk', 'full'])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
@pytest.mark.parametrize('chunk', [1, 7, 257])
def test_low_precision_head_matches_fp32_dense_objective_and_gradient(tmp_path, objective, dtype, chunk):
    torch.manual_seed(120)
    n, valid, width = 257, 31, 16
    h = (torch.randn(n, width) * .5).to(dtype).requires_grad_()
    w = (torch.randn(valid + 3, width) * .5).to(dtype).requires_grad_()
    q = torch.randn(n, valid).log_softmax(-1)
    mask = torch.arange(n) % 5 != 0
    coeff = torch.randn(n)  # Cancellation across chunks exposes BF16 accumulation.
    entropy_coef = .005
    if objective == 'topk':
        teacher, ids = q.topk(5, dim=-1)
        result = chunked_topk_loss(h, w, ids, teacher, mask, valid, chunk, entropy_coef)
    else:
        np.save(tmp_path / 'sample-000000.npy', q.numpy())
        refs = torch.stack((torch.zeros(n, dtype=torch.long), torch.arange(n)), dim=-1)
        result = chunked_full_loss(h, w, refs, mask, tmp_path, valid, chunk, entropy_coef)
    grads = torch.autograd.grad((result[0] * coeff).sum(), (h, w))

    # Independent dense FP32 oracle, using exactly the quantized inputs above.
    hd, wd = h.detach().float().requires_grad_(), w.detach().float().requires_grad_()
    lp = (hd @ wd[:valid].T).log_softmax(-1)
    p = lp.exp()
    if objective == 'topk':
        top_p = p.gather(-1, ids)
        p_bucket = torch.cat((top_p, 1 - top_p.sum(-1, keepdim=True)), dim=-1)
        q_bucket = torch.cat((teacher.exp(), 1 - teacher.exp().sum(-1, keepdim=True)), dim=-1)
        kl = (p_bucket * (p_bucket.log() - q_bucket.log())).sum(-1)
    else:
        kl = (p * (lp - q)).sum(-1)
    expected = (kl + entropy_coef * (p * lp).sum(-1)) * mask
    dense_grads = torch.autograd.grad((expected * coeff).sum(), (hd, wd))
    assert result[0].dtype == torch.float32
    torch.testing.assert_close(result[0], expected, rtol=2e-5, atol=2e-6)
    # Compare to FP32 accumulation followed by a single dtype conversion. A
    # tolerance below one ordinary BF16 rounding unit rejects repeated casts.
    for actual, reference in zip(grads, dense_grads):
        torch.testing.assert_close(actual.float(), reference.to(dtype).float(), rtol=2e-3, atol=2e-5)
    assert grads[1][valid:].count_nonzero() == 0
    assert grads[0][~mask].count_nonzero() == 0


@pytest.mark.parametrize('objective', ['topk', 'full'])
def test_head_gradient_accumulates_small_chunks_before_bf16_cast(tmp_path, objective):
    # All logits are exactly zero in both precisions. With K=V the old and new
    # objectives also agree, isolating repeated BF16 rounding in dw.add_().
    n = 513
    h = torch.ones(n, 3, dtype=torch.bfloat16, requires_grad=True)
    w = torch.zeros(6, 3, dtype=torch.bfloat16, requires_grad=True)
    teacher = torch.tensor([[.5, .25, .125, .125]]).log()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        if objective == 'topk':
            result = chunked_topk_loss(h, w, torch.arange(4)[None].expand(n, -1),
                                      teacher.expand(n, -1), torch.ones(n), 4, 1)
        else:
            np.save(tmp_path / 'sample-000000.npy', teacher.numpy())
            result = chunked_full_loss(h, w, torch.zeros(n, 2, dtype=torch.long),
                                      torch.ones(n), tmp_path, 4, 1)
    _, actual = torch.autograd.grad(result[0].sum(), (h, w))
    reference_weight = w.detach().float().requires_grad_()
    lp = (h.detach().float() @ reference_weight[:4].T).log_softmax(-1)
    reference_loss = (lp.exp() * (lp - teacher)).sum()
    expected, = torch.autograd.grad(reference_loss, reference_weight)
    torch.testing.assert_close(actual, expected.to(torch.bfloat16), rtol=0, atol=0)
