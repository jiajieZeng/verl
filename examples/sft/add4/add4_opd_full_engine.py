"""Opt-in full-vocabulary hooks; teacher scores stay in CPU memory-mapped files."""
from functools import wraps

import torch

from add4_opd_engine import Add4OPDOutput, FIELDS
from add4_opd_full import chunked_full_loss


def wrap_full_forward(original, valid_size, output_size, chunk_size, entropy_coef):
    @wraps(original)
    def forward(self, input_ids=None, labels=None, temperature=1.0, shift_labels=None,
                cu_seqlens=None, cu_seqlens_cpu=None, opd_full_refs=None, opd_full_store=None,
                opd_mask=None, **kwargs):
        if opd_full_refs is None:
            return original(self, input_ids=input_ids, labels=labels, temperature=temperature,
                            shift_labels=shift_labels, cu_seqlens=cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu, **kwargs)
        if float(temperature) != 1.0:
            raise ValueError('Full-vocabulary loss requires raw distributions at temperature 1')
        if cu_seqlens is not None:
            kwargs['cu_seqlens'] = cu_seqlens
        if cu_seqlens_cpu is not None:
            kwargs['cu_seqlens_cpu'] = cu_seqlens_cpu
        outputs = self.model(input_ids, **kwargs)
        from torch.distributed.tensor import DTensor
        weight = self.lm_head.weight
        if isinstance(weight, DTensor):
            weight = weight.full_tensor()
        if weight.shape[0] != output_size:
            raise ValueError('Unexpected student output-head shape')
        values = chunked_full_loss(outputs[0], weight, opd_full_refs, opd_mask,
                                   opd_full_store, valid_size, chunk_size, entropy_coef)
        return Add4OPDOutput(**dict(zip(FIELDS, values, strict=True)))
    return forward


def wrap_full_inputs(original):
    @wraps(original)
    def prepare(self, micro_batch):
        import verl.utils.tensordict_utils as tu
        inputs, args = original(self, micro_batch)
        if 'opd_full_rows' in micro_batch:
            if args['pad_size']:
                raise ValueError('Full-vocabulary OPD requires pad_to_length=false and SP=1')
            inputs['opd_full_refs'] = torch.roll(micro_batch['opd_full_rows'].values(), -1, dims=0)
            inputs['opd_mask'] = torch.roll(micro_batch['loss_mask'].values(), -1, dims=0)
            inputs['opd_full_store'] = tu.get_non_tensor_data(micro_batch, 'opd_full_store', default=None)
        return inputs, args
    return prepare


def wrap_full_outputs(original):
    @wraps(original)
    def prepare(self, output, output_args, micro_batch, logits_processor_func):
        if 'opd_full_rows' not in micro_batch:
            return original(self, output, output_args, micro_batch, logits_processor_func)
        offsets = micro_batch['input_ids'].offsets()
        return {name: torch.nested.nested_tensor_from_jagged(getattr(output, name), offsets) for name in FIELDS}
    return prepare


def install_full_engine(config):
    from verl.models.transformers import qwen3_5
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
    qwen3_5.forward_with_torch_backend = wrap_full_forward(qwen3_5.forward_with_torch_backend,
        config.valid_vocab_size, config.output_vocab_size, config.head_chunk_size, config.entropy_coef)
    FSDPEngineWithLMHead.prepare_model_inputs = wrap_full_inputs(FSDPEngineWithLMHead.prepare_model_inputs)
    FSDPEngineWithLMHead.prepare_model_outputs = wrap_full_outputs(FSDPEngineWithLMHead.prepare_model_outputs)
