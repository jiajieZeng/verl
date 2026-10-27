"""Model contracts for new Add training entries; legacy dense SFT remains untouched."""
from functools import wraps

SUPPORTED_STUDENT_TYPES = ('qwen3_5', 'qwen3_5_moe')


def validate_student_model(config, expected_type=None):
    kind = getattr(config, 'model_type', None)
    if kind not in SUPPORTED_STUDENT_TYPES or (expected_type is not None and kind != expected_type):
        raise ValueError('Expected the configured Qwen3.5 dense or Qwen3.6 MoE student architecture')
    if kind == 'qwen3_5_moe':
        text = config.text_config
        if not 0 < text.num_experts_per_tok <= text.num_experts or text.moe_intermediate_size <= 0:
            raise ValueError('Invalid MoE expert configuration')
        if text.output_router_logits:
            raise ValueError('This Add objective does not include router auxiliary loss; set output_router_logits=false')
    return kind


def root_owned_qwen_head(selector, expected_type=None):
    """Keep fused head reads inside root FSDP hooks; preserve decoder/expert targets."""
    @wraps(selector)
    def select(model, classes):
        validate_student_model(model.config, expected_type)
        return [module for module in selector(model, classes) if module is not model.lm_head]
    return select
