"""Process-local, opt-in VERL hooks for the Add top-k objective."""
from functools import wraps
from dataclasses import dataclass

import torch
from transformers.utils import ModelOutput

from add4_opd_topk import chunked_topk_loss


FIELDS = ('opd_losses', 'opd_reverse_kl', 'opd_entropy', 'opd_student_mass', 'opd_teacher_mass')


@dataclass
class Add4OPDOutput(ModelOutput):
    # ModelOutput registers this as a pytree, so FSDP can find output tensors.
    opd_losses: torch.Tensor | None = None
    opd_reverse_kl: torch.Tensor | None = None
    opd_entropy: torch.Tensor | None = None
    opd_student_mass: torch.Tensor | None = None
    opd_teacher_mass: torch.Tensor | None = None


def wrap_model_forward(original, valid_size, output_size, chunk_size, entropy_coef):
    @wraps(original)
    def forward(self, input_ids=None, labels=None, temperature=1.0, shift_labels=None,
                cu_seqlens=None, cu_seqlens_cpu=None, opd_ids=None, opd_teacher=None, opd_mask=None, **kwargs):
        if opd_ids is None:
            return original(self, input_ids=input_ids, labels=labels, temperature=temperature,
                            shift_labels=shift_labels, cu_seqlens=cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu, **kwargs)
        if float(temperature) != 1.0:
            raise ValueError('Top-k distillation uses raw student/teacher distributions at temperature 1')
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
        values = chunked_topk_loss(outputs[0], weight, opd_ids, opd_teacher, opd_mask,
                                  valid_size, chunk_size, entropy_coef)
        return Add4OPDOutput(**dict(zip(FIELDS, values, strict=True)))
    return forward


def wrap_model_inputs(original):
    @wraps(original)
    def prepare(self, micro_batch):
        inputs, args = original(self, micro_batch)
        if 'opd_teacher_ids' in micro_batch:
            if args['pad_size']:
                raise ValueError('Top-k OPD requires pad_to_length=false and SP=1')
            # Supervision is stored at target-token positions; hidden[t] predicts target[t+1].
            for target, source in [('opd_ids', 'opd_teacher_ids'), ('opd_teacher', 'opd_teacher_logp'),
                                   ('opd_mask', 'loss_mask')]:
                inputs[target] = torch.roll(micro_batch[source].values(), -1, dims=0)
        return inputs, args
    return prepare


def wrap_model_outputs(original):
    @wraps(original)
    def prepare(self, output, output_args, micro_batch, logits_processor_func):
        if 'opd_teacher_ids' not in micro_batch:
            return original(self, output, output_args, micro_batch, logits_processor_func)
        offsets = micro_batch['input_ids'].offsets()
        return {name: torch.nested.nested_tensor_from_jagged(getattr(output, name), offsets) for name in FIELDS}
    return prepare


def wrap_training_metrics(original, dp_size):
    """VERL's SFT postprocessor leaves custom microbatch metrics as nested lists."""
    @wraps(original)
    def postprocess(*args, **kwargs):
        import verl.utils.tensordict_utils as tu
        result = original(*args, **kwargs)
        metrics = tu.get(result, 'metrics')
        def total(value):
            return sum(total(x) for x in value) if isinstance(value, (tuple, list)) else float(value)
        for name in tuple(metrics):
            if name.startswith('opd/'):
                metrics[name] = total(metrics[name]) / dp_size()
        tu.assign_non_tensor(result, metrics=metrics)
        return result
    return postprocess


def install_topk_engine(config):
    from verl.models.transformers import qwen3_5
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
    qwen3_5.forward_with_torch_backend = wrap_model_forward(qwen3_5.forward_with_torch_backend,
        config.valid_vocab_size, config.output_vocab_size, config.head_chunk_size, config.entropy_coef)
    FSDPEngineWithLMHead.prepare_model_inputs = wrap_model_inputs(FSDPEngineWithLMHead.prepare_model_inputs)
    FSDPEngineWithLMHead.prepare_model_outputs = wrap_model_outputs(FSDPEngineWithLMHead.prepare_model_outputs)
