"""Reverse KL on teacher top-k IDs plus one bucket for the remaining probability.

The head projects only response positions, in bounded chunks. Backward recomputes
each chunk, keeping neither a context-by-vocabulary tensor nor its autograd graph.
"""
import torch

from add4_opd_core import TEACHER_LOGPROB_TOLERANCE

TEACHER_TAIL_MASS_FLOOR = 2 ** -24


def _tail_logprobs(logp, teacher_ids, teacher_logp):
    # Form the student complement directly: 1 - sum(p_K) loses small tails.
    tail = logp.scatter(-1, teacher_ids, -torch.inf).logsumexp(-1)
    # Teacher scores arrive as FP32 logprobs. Subtract mass in FP64 so that
    # representable small tails do not disappear in an FP32 reduction.
    log_mass = teacher_logp.detach().double().logsumexp(-1)
    if not torch.isfinite(log_mass).all() or (log_mass > TEACHER_LOGPROB_TOLERANCE).any():
        raise ValueError('Invalid teacher top-k probability mass')
    # A confident FP32 teacher can round its selected mass to one (or slightly
    # above it). Floor only the unresolvable tail; preserve the supplied top-K.
    teacher_tail = (-log_mass.expm1()).clamp_min(TEACHER_TAIL_MASS_FLOOR).log().to(logp.dtype)
    return tail, teacher_tail


def topk_terms(logits, teacher_ids, teacher_logp, valid_size):
    """Global-V probabilities, coarsened to K+1 buckets; never a subset softmax."""
    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    logp = logits[..., :valid_size].to(dtype).log_softmax(-1)
    selected = logp.gather(-1, teacher_ids)
    probability = selected.exp()
    kl = (probability * (selected - teacher_logp.detach())).sum(-1)
    if teacher_ids.shape[-1] < valid_size:
        tail, teacher_tail = _tail_logprobs(logp, teacher_ids, teacher_logp)
        kl = kl + tail.exp() * (tail - teacher_tail)
    entropy = -(logp.exp() * logp).sum(-1)
    return kl, entropy, probability.sum(-1), teacher_logp.detach().exp().sum(-1)


class _ChunkedTopK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, ids, teacher, mask, valid_size, chunk_size, entropy_coef):
        flat = hidden.reshape(-1, hidden.shape[-1])
        if ids.shape != teacher.shape or ids.ndim != 2 or ids.shape[0] != flat.shape[0] or mask.shape != (flat.shape[0],):
            raise ValueError('Misaligned packed top-k supervision')
        selected = torch.nonzero(mask.bool(), as_tuple=True)[0]
        if not selected.numel() or chunk_size <= 0 or not 0 < valid_size <= weight.shape[0]:
            raise ValueError('Invalid or empty top-k response batch')
        chosen_ids, chosen_teacher = ids[selected], teacher[selected]
        if (chosen_ids < 0).any() or (chosen_ids >= valid_size).any() or not torch.isfinite(chosen_teacher).all():
            raise ValueError('Invalid teacher support or logprobs')
        if (chosen_teacher > TEACHER_LOGPROB_TOLERANCE).any() or (
            chosen_teacher.double().logsumexp(-1) > TEACHER_LOGPROB_TOLERANCE
        ).any():
            raise ValueError('Invalid teacher top-k probability mass')
        sorted_ids = chosen_ids.sort(-1).values
        if (sorted_ids[:, 1:] == sorted_ids[:, :-1]).any():
            raise ValueError('Duplicate teacher support IDs')
        dtype = torch.float64 if hidden.dtype == torch.float64 else torch.float32
        outputs = [torch.zeros(flat.shape[0], device=hidden.device, dtype=dtype) for _ in range(5)]
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            projection = weight[:valid_size].to(dtype)
            for part in selected.split(chunk_size):
                logits = flat[part].to(dtype) @ projection.T
                kl, entropy, student_mass, teacher_mass = topk_terms(logits, ids[part], teacher[part], valid_size)
                for target, value in zip(outputs, (kl - entropy_coef * entropy, kl, entropy, student_mass, teacher_mass)):
                    target[part] = value
        ctx.save_for_backward(hidden, weight, ids, teacher.detach(), selected)
        ctx.valid_size, ctx.chunk_size, ctx.entropy_coef = valid_size, chunk_size, entropy_coef
        ctx.mark_non_differentiable(*outputs[1:])
        return tuple(outputs)

    @staticmethod
    def backward(ctx, grad_loss, *ignored):
        hidden, weight, ids, teacher, selected = ctx.saved_tensors
        flat = hidden.reshape(-1, hidden.shape[-1])
        dtype = torch.float64 if hidden.dtype == torch.float64 else torch.float32
        dh = torch.zeros_like(flat, dtype=dtype)
        dw = torch.zeros_like(weight, dtype=dtype)
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            projection = weight[:ctx.valid_size].to(dtype)
            for part in selected.split(ctx.chunk_size):
                h = flat[part].to(dtype)
                logp = (h @ projection.T).log_softmax(-1)
                probs = logp.exp()
                top_logp = logp.gather(-1, ids[part])
                # dL/dp_j is shared by all IDs in the tail bucket. Then apply
                # the full softmax Jacobian, including both K and its complement.
                derivative = torch.zeros_like(logp)
                if ids.shape[-1] < ctx.valid_size:
                    tail, teacher_tail = _tail_logprobs(logp, ids[part], teacher[part])
                    derivative[:] = (tail - teacher_tail + 1)[:, None]
                derivative.scatter_(-1, ids[part], top_logp - teacher[part] + 1)
                dz = probs * (derivative - (probs * derivative).sum(-1, keepdim=True))
                if ctx.entropy_coef:
                    entropy = -(probs * logp).sum(-1, keepdim=True)
                    dz.add_(ctx.entropy_coef * probs * (logp + entropy))
                dz = dz * grad_loss[part, None].to(dtype)
                dh[part] = dz @ projection
                dw[:ctx.valid_size].add_(dz.T @ h)
        # Cast once at the autograd boundary, never once per response chunk.
        return dh.reshape_as(hidden).to(hidden.dtype), dw.to(weight.dtype), None, None, None, None, None, None


def chunked_topk_loss(hidden, weight, ids, teacher, mask, valid_size, chunk_size=128, entropy_coef=0.0):
    return _ChunkedTopK.apply(hidden, weight, ids, teacher, mask, valid_size, chunk_size, entropy_coef)
