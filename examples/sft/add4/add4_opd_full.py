"""Full-valid-vocabulary teacher transport and bounded-memory reverse KL."""
import base64
import hashlib
from pathlib import Path

import numpy as np
import torch


def parse_full_scores(result, prompt, response, valid_size, output_size, version):
    meta = result['meta_info']
    if meta.get('weight_version') != version:
        raise ValueError('Full-vocabulary teacher weight version changed')
    actual = meta['input_token_logprobs']
    count = len(response)
    if not count or len(actual) not in (count, count + 1):
        raise ValueError('Incorrect full-vocabulary response span')
    if len(actual) == count + 1 and actual[0][1] != prompt[-1]:
        raise ValueError('Incorrect predecessor token')
    if [row[1] for row in actual[-count:]] != response:
        raise ValueError('Teacher/student token IDs differ')
    shape = meta['input_top_logprobs_shape']
    nulls = meta['input_top_logprobs_null_prefix']
    if (not isinstance(nulls, int) or nulls not in (0, 1) or
            shape != [len(actual) - nulls, output_size] or shape[0] < count):
        raise ValueError('Full vocabulary was truncated or positions are misaligned')
    if any(row[0] is not None for row in actual[:nulls]):
        raise ValueError('Unexpected null-prefix marker')
    arrays = []
    for name, dtype, marker in [('val', '<f4', 'float32'), ('idx', '<i4', 'int32')]:
        key = f'input_top_logprobs_{name}_flat_b64'
        if meta.get(key + '_dtype') != marker:
            raise ValueError('Expected little-endian FP32/int32 full-vocabulary arrays')
        raw = base64.b64decode(meta[key], validate=True)
        if len(raw) != shape[0] * output_size * 4:
            raise ValueError('Incorrect full-vocabulary byte length')
        arrays.append(np.frombuffer(raw, dtype=dtype).reshape(shape)[-count:])
    values, ids = arrays
    if np.any(ids < 0) or np.any(ids >= output_size):
        raise ValueError('Undefined output-head column')
    if not np.all(np.sort(ids, axis=1) == np.arange(output_size)):
        raise ValueError('Full vocabulary has missing or duplicate IDs')
    if np.any(np.isnan(values)) or np.any(values > 1e-5):
        raise ValueError('Invalid teacher logprob')
    dense = np.empty((count, output_size), dtype=np.float32)
    np.put_along_axis(dense, ids, values, axis=1)
    if not np.all(np.isfinite(dense[:, :valid_size])):
        raise ValueError('Nonfinite teacher distribution on valid tokens')
    log_mass = np.logaddexp.reduce(dense.astype(np.float64), axis=1)
    if not np.all(np.abs(log_mass) < 5e-4):
        raise ValueError('Teacher full distribution is not normalized')
    selected = dense[np.arange(count), response]
    if not np.allclose(selected, [row[0] for row in actual[-count:]], rtol=0, atol=5e-4):
        raise ValueError('Full-vocabulary rows do not align with scored tokens')
    valid = dense[:, :valid_size].copy()
    valid -= np.logaddexp.reduce(valid.astype(np.float64), axis=1)[:, None]
    return valid


def full_request(prompt, response, output_size):
    return dict(input_ids=prompt + response,
                sampling_params={'max_new_tokens': 0, 'temperature': 1.0},
                return_logprob=True, return_text_in_logprobs=False,
                logprob_start_len=len(prompt) - 1,
                # This requests EVERY output column. No top-k truncation remains.
                top_logprobs_num=output_size, return_flat_raw_top_logprobs=True,
                return_flat_raw_top_logprobs_b64=True)


def score_full_to_file(client, prompt, response, config, destination):
    valid, output = config['valid_vocab_size'], config['output_vocab_size']
    chunk = config['teacher_score_chunk_size']
    if not prompt or not response or chunk <= 0 or not 0 < valid < output:
        raise ValueError('Invalid full-vocabulary request')
    if any(type(x) is not int or not 0 <= x < valid for x in prompt + response):
        raise ValueError('Undefined input token ID')
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.partial.npy')
    stored = np.lib.format.open_memmap(temporary, mode='w+', dtype='<f4', shape=(len(response), valid))
    digest = hashlib.sha256()
    try:
        for start in range(0, len(response), chunk):
            part = response[start:start + chunk]
            prefix = prompt + response[:start]
            result = client.request('/generate', full_request(prefix, part, output))
            scores = parse_full_scores(result, prefix, part, valid, output, config['teacher_weight_version'])
            stored[start:start + len(part)] = scores
            digest.update(scores.tobytes())
        stored.flush()
    finally:
        del stored
    temporary.replace(destination)
    return dict(path=str(destination), shape=[len(response), valid], dtype='float32',
                sha256=digest.hexdigest(), weight_version=config['teacher_weight_version'])


def read_teacher_rows(folder, refs, valid_size, device, dtype):
    """Only response chunks enter accelerator memory; prompt carries two integers/token."""
    refs = refs.detach().cpu().numpy()
    if refs.ndim != 2 or refs.shape[1] != 2 or np.any(refs < 0):
        raise ValueError('Invalid full-vocabulary row references')
    output = np.empty((len(refs), valid_size), dtype=np.float32)
    for sample in np.unique(refs[:, 0]):
        positions = np.flatnonzero(refs[:, 0] == sample)
        source = np.load(Path(folder) / f'sample-{sample:06d}.npy', mmap_mode='r', allow_pickle=False)
        if source.ndim != 2 or source.shape[1] != valid_size or source.dtype != np.float32:
            raise ValueError('Invalid full-vocabulary score file')
        if np.any(refs[positions, 1] >= source.shape[0]):
            raise ValueError('Teacher row reference is out of bounds')
        output[positions] = source[refs[positions, 1]]
        del source
    teacher = torch.from_numpy(output).to(device=device, dtype=dtype)
    if not torch.isfinite(teacher).all() or (teacher.logsumexp(-1).abs() > 5e-4).any():
        raise ValueError('Teacher full distribution is not finite and normalized')
    return teacher


class _ChunkedFull(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, refs, mask, folder, valid_size, chunk_size, entropy_coef):
        flat = hidden.reshape(-1, hidden.shape[-1])
        if refs.shape != (flat.shape[0], 2) or mask.shape != (flat.shape[0],):
            raise ValueError('Misaligned full-vocabulary supervision')
        selected = torch.nonzero(mask.bool(), as_tuple=True)[0]
        if not selected.numel() or chunk_size <= 0 or not 0 < valid_size <= weight.shape[0]:
            raise ValueError('Invalid full-vocabulary response batch')
        dtype = torch.float64 if hidden.dtype == torch.float64 else torch.float32
        outputs = [torch.zeros(flat.shape[0], device=hidden.device, dtype=dtype) for _ in range(5)]
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            projection = weight[:valid_size].to(dtype)
            for part in selected.split(chunk_size):
                q = read_teacher_rows(folder, refs[part], valid_size, hidden.device, dtype)
                logp = (flat[part].to(dtype) @ projection.T).log_softmax(-1)
                p = logp.exp()
                kl, entropy = (p * (logp - q)).sum(-1), -(p * logp).sum(-1)
                for target, value in zip(outputs, (kl - entropy_coef * entropy, kl, entropy, p.sum(-1), q.exp().sum(-1))):
                    target[part] = value
        ctx.save_for_backward(hidden, weight, refs, selected)
        ctx.folder, ctx.valid_size, ctx.chunk_size, ctx.entropy_coef = str(folder), valid_size, chunk_size, entropy_coef
        ctx.mark_non_differentiable(*outputs[1:])
        return tuple(outputs)

    @staticmethod
    def backward(ctx, grad_loss, *ignored):
        hidden, weight, refs, selected = ctx.saved_tensors
        flat = hidden.reshape(-1, hidden.shape[-1])
        dtype = torch.float64 if hidden.dtype == torch.float64 else torch.float32
        dh = torch.zeros_like(flat, dtype=dtype)
        dw = torch.zeros_like(weight, dtype=dtype)
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            projection = weight[:ctx.valid_size].to(dtype)
            for part in selected.split(ctx.chunk_size):
                h = flat[part].to(dtype)
                q = read_teacher_rows(ctx.folder, refs[part], ctx.valid_size, hidden.device, dtype)
                logp = (h @ projection.T).log_softmax(-1)
                p = logp.exp()
                difference = logp - q
                dz = p * (difference - (p * difference).sum(-1, keepdim=True))
                if ctx.entropy_coef:
                    entropy = -(p * logp).sum(-1, keepdim=True)
                    dz.add_(ctx.entropy_coef * p * (logp + entropy))
                dz = dz * grad_loss[part, None].to(dtype)
                dh[part] = dz @ projection
                dw[:ctx.valid_size].add_(dz.T @ h)
        # Keep chunk accumulation in FP32 (FP64 for numerical checks).
        return dh.reshape_as(hidden).to(hidden.dtype), dw.to(weight.dtype), None, None, None, None, None, None


def chunked_full_loss(hidden, weight, refs, mask, folder, valid_size, chunk_size=8, entropy_coef=0.0):
    return _ChunkedFull.apply(hidden, weight, refs, mask, str(folder), valid_size, chunk_size, entropy_coef)
