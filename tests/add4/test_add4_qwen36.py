import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'examples/sft/add4'))
from add4_qwen_models import root_owned_qwen_head, validate_student_model
from add4_opd_export import StudentShardWriter


def tiny_moe():
    from transformers import Qwen3_5MoeConfig, Qwen3_5MoeForConditionalGeneration
    config = Qwen3_5MoeConfig(
        text_config=dict(vocab_size=32, hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                         num_key_value_heads=1, head_dim=8, layer_types=['full_attention']*2, num_experts=4,
                         num_experts_per_tok=2, moe_intermediate_size=8, shared_expert_intermediate_size=8,
                         rope_parameters={'rope_type': 'default', 'rope_theta': 10000.,
                                          'partial_rotary_factor': .5, 'mrope_section': [1, 1, 0]},
                         output_router_logits=False),
        vision_config=dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=16,
                           num_position_embeddings=16, patch_size=2, spatial_merge_size=2, temporal_patch_size=1),
        image_token_id=29, video_token_id=30, vision_start_token_id=27, vision_end_token_id=28)
    config._attn_implementation = 'eager'
    return Qwen3_5MoeForConditionalGeneration(config)


def test_moe_wrap_targets_keep_experts_and_decoder_layers():
    from verl.utils.fsdp_utils import _select_fsdp2_wrap_targets
    model = tiny_moe()
    original = _select_fsdp2_wrap_targets(model, model._no_split_modules)
    actual = root_owned_qwen_head(_select_fsdp2_wrap_targets, 'qwen3_5_moe')(model, model._no_split_modules)
    assert model.lm_head in original and model.lm_head not in actual
    assert set(actual) == set(original) - {model.lm_head}
    assert all(layer in actual for layer in model.model.language_model.layers)
    with pytest.raises(ValueError): validate_student_model(model.config, 'qwen3_5')
    model.config.text_config.output_router_logits = True
    with pytest.raises(ValueError, match='auxiliary'): validate_student_model(model.config)


def test_original_sft_keeps_rejecting_moe_and_new_entry_import_is_inert():
    import train_add4
    original = train_add4.root_owned_lm_head
    import train_add4_qwen36
    assert train_add4.root_owned_lm_head is original
    with pytest.raises(ValueError, match='dense'):
        original(lambda *args: [])(tiny_moe(), [])
    base = type('Base', (), {'_build_config': lambda self: None})
    sft = SimpleNamespace(SFTTrainer=base)
    train_add4_qwen36.install_moe_sft(sft)
    trainer = sft.SFTTrainer()
    trainer.model_config = SimpleNamespace(hf_config=tiny_moe().config)
    trainer.config = SimpleNamespace(engine=SimpleNamespace(strategy='fsdp2'),
                                    model=SimpleNamespace(lora_rank=0, lora_adapter_path=None))
    trainer._build_config()


@pytest.mark.parametrize('objective', ['sft', 'opd'])
def test_actual_tiny_qwen_moe_backward_optimizer_and_restore(tmp_path, monkeypatch, objective):
    from verl.models.transformers.qwen3_5 import forward_with_torch_backend
    import verl.utils.experimental.torch_functional as fused
    from add4_opd_engine import wrap_model_forward
    # The image has GPU FlashAttention installed; explicitly exercise its torch reference on CPU.
    monkeypatch.setattr(fused, '_FLASH_ATTN_CROSS_ENTROPY_AVAILABLE', False)
    torch.manual_seed(35)
    model = tiny_moe()
    x = torch.tensor([[1, 2, 3, 4, 5]])
    mask = torch.tensor([0, 1, 1, 1, 0])
    if objective == 'sft':
        output = forward_with_torch_backend(model, input_ids=x, use_cache=False)
        loss = -output.log_probs.reshape(-1)[mask.bool()].mean()
    else:
        ids = torch.tensor([[0, 1, 2]]*5)
        q = torch.full((5, 3), -2.)
        forward = wrap_model_forward(forward_with_torch_backend, 30, 32, 2, .005)
        output = forward(model, input_ids=x, opd_ids=ids, opd_teacher=q, opd_mask=mask, use_cache=False)
        loss = output.opd_losses.sum() / mask.sum()
    assert torch.isfinite(loss)
    loss.backward()
    mlp = model.model.language_model.layers[0].mlp
    for parameter in (mlp.gate.weight, mlp.experts.gate_up_proj, mlp.experts.down_proj, mlp.shared_expert.gate_proj.weight):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    optimizer.step()
    path = tmp_path / 'checkpoint.pt'
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, path)
    restored = tiny_moe()
    state = torch.load(path, weights_only=True)
    restored.load_state_dict(state['model'])
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=.001)
    restored_optimizer.load_state_dict(state['optimizer'])
    torch.testing.assert_close(restored(x, use_cache=False).logits, model(x, use_cache=False).logits)
    assert restored_optimizer.state


def test_native_unfused_moe_export_reload_preserves_logits(tmp_path):
    from verl.workers.engine.fsdp.utils import unfuse_moe_params
    from safetensors.torch import load_file
    from transformers import Qwen3_5MoeForConditionalGeneration
    torch.manual_seed(9)
    model = tiny_moe().to(torch.bfloat16)
    writer = StudentShardWriter(tmp_path, 16384)
    for name, tensor in unfuse_moe_params(model.state_dict().items(), 'qwen3_5_moe'):
        writer.add(name, tensor)
    writer.flush()
    assert any('experts.0.gate_proj.weight' in key for key in writer.mapping)
    assert len(set(writer.mapping.values())) < len(writer.mapping) / 4
    # Same disk layout is accepted by the SGLang MoE loader's per-expert mapping.
    exported = {}
    for file in set(writer.mapping.values()): exported.update(load_file(str(tmp_path / file)))
    packed = model.model.language_model.layers[0].mlp.experts.gate_up_proj
    prefix = 'model.language_model.layers.0.mlp.experts.'
    torch.testing.assert_close(exported[prefix + '0.gate_proj.weight'], packed[0, :8])
    torch.testing.assert_close(exported[prefix + '0.up_proj.weight'], packed[0, 8:])
    model.config.save_pretrained(tmp_path)
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps({
        'metadata': {'total_size': writer.total_bytes}, 'weight_map': writer.mapping}))
    restored, loading = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        tmp_path, dtype=torch.bfloat16, attn_implementation='eager', output_loading_info=True)
    assert not loading['missing_keys'] and not loading['unexpected_keys']
    x = torch.tensor([[1, 2, 3, 4]])
    torch.testing.assert_close(model(x, use_cache=False).logits, restored(x, use_cache=False).logits)


def test_shard_writer_copies_views_preserves_integer_dtype_and_rejects_duplicates(tmp_path):
    from safetensors.torch import load_file
    writer = StudentShardWriter(tmp_path, 64)
    backing = torch.ones(20, 4, dtype=torch.bfloat16)
    writer.add('view', backing[0])
    backing.zero_()
    writer.add('integer', torch.tensor([3], dtype=torch.int64))
    with pytest.raises(ValueError, match='Duplicate'): writer.add('view', backing[1])
    writer.flush()
    result = load_file(str(tmp_path / writer.mapping['view']))
    assert result['view'].tolist() == [1] * 4
    assert result['integer'].dtype == torch.int64


def test_qwen36_hydra_template_requires_explicit_paths(monkeypatch):
    from hydra import initialize_config_dir, compose
    from omegaconf import OmegaConf
    from omegaconf.errors import InterpolationResolutionError
    root = Path(__file__).resolve().parents[2]
    values = dict(QWEN36_MODEL_DIR='/models/qwen36', ADD_TOKENIZER_DIR='/tokenizer',
                  ADD_TRAIN_FILES='/train.parquet', VERL_ROOT=str(root),
                  ADD_RUN_ID='qwen36-smoke', ADD_RUN_DIR='/data/zengjiajie/opd/qwen36-smoke')
    for name in values: monkeypatch.delenv(name, raising=False)
    # The repository root is needed to compose Hydra's base config; model/data remain unset.
    monkeypatch.setenv('VERL_ROOT', str(root))
    with initialize_config_dir(config_dir=str(root/'examples/sft/add4'), version_base=None):
        config = compose(config_name='qwen36.sft.example')
    with pytest.raises(InterpolationResolutionError): _ = config.model.path
    for name, value in values.items(): monkeypatch.setenv(name, value)
    assert config.model.path == values['QWEN36_MODEL_DIR']
    assert config.engine.strategy == 'fsdp2' and config.engine.offload_policy
    assert config.data.max_length == 262144 and config.trainer.total_training_steps == 1
    OmegaConf.to_container(config, resolve=True)
    opd = OmegaConf.load(root/'examples/sft/add4/opd.qwen36.example.yaml')
    assert opd.student_model_type == 'qwen3_5_moe' and opd.teacher_top_k == 128
