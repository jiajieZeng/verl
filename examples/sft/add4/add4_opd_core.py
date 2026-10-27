"""Token alignment, valid-vocabulary normalization and OPD reference objectives."""
import math

# Allow small positive logprobs from FP32 scoring/valid-vocabulary correction.
TEACHER_LOGPROB_TOLERANCE = 1e-5


def validate_vocabulary(tokenizer, valid_size, output_size):
    ids = set(tokenizer.get_vocab().values())
    if ids != set(range(valid_size)) or output_size <= valid_size:
        raise ValueError('Expected a contiguous defined vocabulary and trailing padded output rows')


def prompt_from_sft_ids(ids, mask):
    selected = [i for i, value in enumerate(mask) if value]
    if not selected or selected[0] == 0 or selected != list(range(selected[0], selected[-1] + 1)):
        raise ValueError('Expected one nonempty completion span after the Add prefix')
    return list(ids[:selected[0]])


def renormalize_logprob(selected_logprob, unused_logprobs):
    """log p(token | token in V), from full-head logprobs at one position."""
    if not math.isfinite(selected_logprob) or selected_logprob > TEACHER_LOGPROB_TOLERANCE:
        raise ValueError('Invalid selected-token logprob')
    if any(math.isnan(x) or x > 0 for x in unused_logprobs):
        raise ValueError('Invalid padded-row logprob')
    mass = math.fsum(math.exp(x) for x in unused_logprobs)
    if not 0 <= mass < 1:
        raise ValueError('Invalid padded-row probability mass')
    result = selected_logprob - math.log1p(-mass)
    if result > TEACHER_LOGPROB_TOLERANCE:
        raise ValueError('Inconsistent selected and padded-row probabilities')
    return result


def parse_teacher_scores(result, prompt_ids, response_ids, valid_size, output_size):
    meta = result['meta_info']
    actual = meta['input_token_logprobs']
    padded = meta['input_token_ids_logprobs']
    count = len(response_ids)
    # SGLang may retain the one predecessor entry at logprob_start_len.
    if len(actual) not in (count, count + 1) or len(padded) != len(actual):
        raise ValueError('Teacher score count does not match the response span')
    if len(actual) == count + 1 and actual[0][1] != prompt_ids[-1]:
        raise ValueError('Unexpected predecessor token in teacher scores')
    expected_unused = set(range(valid_size, output_size))
    scores = []
    for token, row, unused in zip(response_ids, actual[-count:], padded[-count:], strict=True):
        if row[1] != token or not 0 <= token < valid_size:
            raise ValueError('Teacher/student token ID mismatch')
        if unused is None or len(unused) != len(expected_unused):
            raise ValueError('Teacher omitted padded vocabulary rows')
        if {item[1] for item in unused} != expected_unused:
            raise ValueError('Teacher returned incorrect padded vocabulary IDs')
        scores.append(renormalize_logprob(row[0], [item[0] for item in unused]))
    return scores


def parse_teacher_topk(result, prompt_ids, response_ids, valid_size, output_size, top_k):
    """Keep teacher top-k in valid V, retaining global-V probabilities (no subset softmax)."""
    parse_teacher_scores(result, prompt_ids, response_ids, valid_size, output_size)
    meta = result['meta_info']
    rows = meta['input_top_logprobs']
    if len(rows) != len(meta['input_token_logprobs']):
        raise ValueError('Teacher top-k positions do not match response scores')
    requested = min(output_size, top_k + output_size - valid_size)
    ids, scores = [], []
    for row, unused in zip(rows[-len(response_ids):], meta['input_token_ids_logprobs'][-len(response_ids):], strict=True):
        if row is None or len(row) != requested:
            raise ValueError('Teacher omitted requested top-k rows')
        if len({item[1] for item in row}) != requested:
            raise ValueError('Duplicate teacher top-k token IDs')
        if any(not isinstance(item[1], int) or not 0 <= item[1] < output_size
               or not math.isfinite(item[0]) or item[0] > TEACHER_LOGPROB_TOLERANCE for item in row):
            raise ValueError('Invalid teacher top-k entry')
        if any(a[0] < b[0] - 1e-6 for a, b in zip(row, row[1:])):
            raise ValueError('Teacher top-k entries are not probability-sorted')
        valid = [item for item in row if item[1] < valid_size][:top_k]
        if len(valid) != top_k:
            raise ValueError('Not enough defined tokens in teacher top-k')
        logp = [renormalize_logprob(item[0], [x[0] for x in unused]) for item in valid]
        if math.fsum(math.exp(x) for x in logp) > math.exp(TEACHER_LOGPROB_TOLERANCE):
            raise ValueError('Teacher top-k probability mass exceeds one')
        ids.append([item[1] for item in valid])
        scores.append(logp)
    return {'ids': ids, 'logprobs': scores}


def valid_token_logprobs(logits, labels, valid_size, temperature=1.0):
    """Reference implementation for tests and small CPU smoke models."""
    import torch
    if temperature <= 0 or torch.any(labels < 0) or torch.any(labels >= valid_size):
        raise ValueError('Invalid temperature or target token ID')
    return (logits[..., :valid_size].float() / temperature).log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)


def sampled_opd_loss(current_logp, old_logp, teacher_logp, mask, denominator=None, dp_size=1):
    """Importance-sampling policy gradient with detached token-local reverse-KL advantage.

    Student sampling and current/old logp use the same temperature and valid V.
    Teacher scores use its raw distribution restricted to the same V.
    """
    import torch
    if current_logp.shape != old_logp.shape or current_logp.shape != teacher_logp.shape or current_logp.shape != mask.shape:
        raise ValueError('OPD tensors must have identical shapes')
    selected = mask.to(torch.bool)
    current, old, teacher = current_logp[selected], old_logp[selected].detach(), teacher_logp[selected].detach()
    if current.numel() == 0 or not all(torch.isfinite(x).all() for x in (current, old, teacher)):
        raise ValueError('Empty or nonfinite OPD response scores')
    denominator = current.numel() if denominator is None else denominator
    if denominator <= 0:
        raise ValueError('Empty global response-token denominator')
    ratio = (current - old).exp()
    if not torch.isfinite(ratio).all():
        raise ValueError('Nonfinite policy importance ratio')
    advantage = teacher - old
    return -(ratio * advantage).sum() * dp_size / denominator


def restrict_fused_head(original, valid_size, output_size):
    """Slice only the loss view; keep stored parameters/checkpoints unchanged."""
    def forward(self, hidden_states, vocab_weights, input_ids, temperature=1.0, **kwargs):
        if vocab_weights.shape[0] != output_size:
            raise ValueError('Unexpected student output-head size')
        return original(self, hidden_states=hidden_states, vocab_weights=vocab_weights[:valid_size],
                        input_ids=input_ids, temperature=temperature, **kwargs)
    return forward
