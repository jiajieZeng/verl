"""Lossless, opt-in response trimming for full-vocabulary OPD scoring."""


def trim_full_scoring_response(request, result):
    """Drop only unused output top-k data; preserve the existing full parser wire format.

    Call on the teacher host before cross-host transport. This does not avoid
    SGLang's upstream computation/serialization, and must not trim generation.
    Non-full, batched and streaming requests pass through unchanged.
    """
    if not isinstance(request, dict) or not isinstance(result, dict):
        return result
    sampling = request.get('sampling_params')
    ids = request.get('input_ids')
    meta = result.get('meta_info')
    if not isinstance(sampling, dict) or not isinstance(meta, dict):
        return result
    shape = meta.get('input_top_logprobs_shape')
    if not (
        isinstance(ids, list) and ids and all(type(x) is int for x in ids)
        and sampling.get('max_new_tokens') == 0
        and request.get('stream', False) is False
        and request.get('return_logprob') is True
        and request.get('return_flat_raw_top_logprobs') is True
        and request.get('return_flat_raw_top_logprobs_b64') is True
        and isinstance(shape, list) and len(shape) == 2
        and type(shape[1]) is int and shape[1] > 0
        and request.get('top_logprobs_num') == shape[1]
        and 'input_top_logprobs_val_flat_b64' in meta
        and 'input_top_logprobs_idx_flat_b64' in meta
        and 'output_top_logprobs' in meta
    ):
        return result
    return {**result, 'meta_info': {
        key: value for key, value in meta.items() if key != 'output_top_logprobs'
    }}
