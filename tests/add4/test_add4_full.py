import base64
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'examples/sft/add4'))
from add4_opd_full import parse_full_scores, score_full_to_file, chunked_full_loss


def response_for(prompt, response, size=6, predecessor=True):
    tokens = ([prompt[-1]] if predecessor else []) + response
    probs = np.arange(1, size + 1, dtype=np.float64)
    logp = np.log(probs / probs.sum()).astype('<f4')
    ids = np.tile(np.arange(size - 1, -1, -1, dtype='<i4'), (len(response), 1))
    values = np.tile(logp[::-1], (len(response), 1))
    return {'meta_info': {
        'weight_version': 'frozen',
        'input_token_logprobs': ([[None, prompt[-1], None]] if predecessor else []) + [[float(logp[x]), x, None] for x in response],
        'input_top_logprobs_shape': [len(response), size],
        'input_top_logprobs_null_prefix': int(predecessor),
        'input_top_logprobs_val_flat_b64': base64.b64encode(values.tobytes()).decode(),
        'input_top_logprobs_idx_flat_b64': base64.b64encode(ids.tobytes()).decode(),
        'input_top_logprobs_val_flat_b64_dtype': 'float32',
        'input_top_logprobs_idx_flat_b64_dtype': 'int32'}}


@pytest.mark.parametrize('predecessor', [False, True])
def test_full_distribution_excludes_padding_and_normalizes_every_valid_token(predecessor):
    result = parse_full_scores(response_for([0, 1], [2, 3], predecessor=predecessor), [0, 1], [2, 3], 4, 6, 'frozen')
    np.testing.assert_allclose(np.exp(result), [[.1, .2, .3, .4]] * 2, rtol=1e-6)


@pytest.mark.parametrize('failure', ['missing', 'duplicate', 'dtype', 'bytes', 'version', 'token', 'null', 'nan', 'mass'])
def test_full_wire_rejects_incomplete_or_misaligned_scores(failure):
    result = response_for([0, 1], [2, 3]); meta = result['meta_info']
    if failure == 'missing': meta['input_top_logprobs_shape'][1] = 4
    if failure == 'dtype': meta['input_top_logprobs_val_flat_b64_dtype'] = 'float16'
    if failure == 'bytes': meta['input_top_logprobs_val_flat_b64'] = 'AA=='
    if failure == 'version': meta['weight_version'] = 'another'
    if failure == 'token': meta['input_token_logprobs'][-1][1] = 0
    if failure == 'null': meta['input_top_logprobs_null_prefix'] = 2
    if failure in ('duplicate', 'nan', 'mass'):
        key = 'input_top_logprobs_' + ('idx' if failure == 'duplicate' else 'val') + '_flat_b64'
        arr = np.frombuffer(base64.b64decode(meta[key]), dtype='<i4' if failure == 'duplicate' else '<f4').copy()
        if failure == 'duplicate': arr[0] = arr[1]
        elif failure == 'nan': arr[0] = np.nan
        else: arr -= 1
        meta[key] = base64.b64encode(arr.tobytes()).decode()
    with pytest.raises(ValueError): parse_full_scores(result, [0, 1], [2, 3], 4, 6, 'frozen')


def test_full_http_chunks_preserve_prefix_and_publish_atomically(tmp_path):
    from add4_opd_client import SGLangClient
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread
    seen = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            seen.append(payload)
            split = payload['logprob_start_len'] + 1
            result = response_for(payload['input_ids'][:split], payload['input_ids'][split:])
            self.send_response(200); self.end_headers(); self.wfile.write(json.dumps(result).encode())
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.serve_forever); thread.start()
    config = dict(valid_vocab_size=4, output_vocab_size=6, teacher_score_chunk_size=2, teacher_weight_version='frozen')
    path = tmp_path / 'scores.npy'
    try:
        result = score_full_to_file(SGLangClient(f'http://127.0.0.1:{server.server_port}'), [0, 1], [2, 3, 2], config, path)
    finally:
        server.shutdown(); server.server_close(); thread.join()
    assert result['shape'] == [3, 4] and path.is_file() and not path.with_suffix('.partial.npy').exists()
    assert [r['input_ids'] for r in seen] == [[0, 1, 2, 3], [0, 1, 2, 3, 2]]
    assert [r['logprob_start_len'] for r in seen] == [1, 3]
    assert all(r['top_logprobs_num'] == 6 and r['return_flat_raw_top_logprobs_b64'] and 'messages' not in r for r in seen)
    original = path.read_bytes()
    bad = SimpleNamespace(request=lambda *a: response_for([0, 1], [2], size=5))
    with pytest.raises(ValueError): score_full_to_file(bad, [0, 1], [2], config, path)
    assert path.read_bytes() == original


@pytest.mark.parametrize('chunk', [1, 3])
def test_full_loss_and_gradients_equal_dense_reference(tmp_path, chunk):
    torch.manual_seed(128)
    h = torch.randn(5, 3, dtype=torch.double, requires_grad=True)
    w = torch.randn(8, 3, dtype=torch.double, requires_grad=True)
    q = torch.randn(3, 6).log_softmax(-1)
    np.save(tmp_path / 'sample-000000.npy', q.numpy())
    refs = torch.tensor([[-1, -1], [0, 2], [0, 0], [-1, -1], [0, 1]])
    mask = torch.tensor([0, 1, 1, 0, 1]); coeff = torch.tensor([0., .3, .7, 0., .2], dtype=torch.double)
    output = chunked_full_loss(h, w, refs, mask, tmp_path, 6, chunk, .005)
    loss = (output[0] * coeff).sum(); dh, dw = torch.autograd.grad(loss, (h, w))
    h2, w2 = h.detach().requires_grad_(), w.detach().requires_grad_()
    lp = (h2[mask.bool()] @ w2[:6].T).log_softmax(-1); teacher = q[[2, 0, 1]].double()
    kl = (lp.exp() * (lp - teacher)).sum(-1); entropy = -(lp.exp() * lp).sum(-1)
    expected = ((kl - .005 * entropy) * coeff[mask.bool()]).sum()
    gh, gw = torch.autograd.grad(expected, (h2, w2))
    torch.testing.assert_close(loss, expected); torch.testing.assert_close(dh, gh); torch.testing.assert_close(dw, gw)
    assert dw[6:].count_nonzero() == 0 and dh[~mask.bool()].count_nonzero() == 0
    assert (output[1][mask.bool()] >= -1e-6).all()
    torch.testing.assert_close(output[3][mask.bool()], torch.ones(3, dtype=torch.double))
    assert torch.autograd.gradcheck(lambda a,b: chunked_full_loss(a,b,refs,mask,tmp_path,6,chunk,.005)[0], (h,w))


@pytest.mark.parametrize('kind', ['dense', 'moe'])
def test_full_loss_actual_qwen_backward(tmp_path, kind):
    from test_add4_qwen36 import tiny_moe
    from add4_opd_full_engine import wrap_full_forward
    model = tiny_moe()
    if kind == 'dense':
        from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
        model = Qwen3_5ForConditionalGeneration(Qwen3_5Config(
            text_config=model.config.text_config.to_dict(), vision_config=model.config.vision_config.to_dict()))
        model.config._attn_implementation = 'eager'
    q = torch.randn(3, 30).log_softmax(-1)
    np.save(tmp_path / 'sample-000000.npy', q.numpy())
    forward = wrap_full_forward(lambda *a,**k: None, 30, 32, 2, .005)
    output = forward(model, input_ids=torch.tensor([[1,2,3,4,5]]),
                     opd_full_refs=torch.tensor([[-1,-1],[0,0],[0,1],[0,2],[-1,-1]]),
                     opd_mask=torch.tensor([0,1,1,1,0]), opd_full_store=tmp_path, use_cache=False)
    loss = output.opd_losses.sum(); assert torch.isfinite(loss); loss.backward()
    assert model.lm_head.weight.grad.abs().sum() > 0
    if kind == 'moe':
        mlp = model.model.language_model.layers[0].mlp
        assert mlp.gate.weight.grad.abs().sum() > 0 and mlp.experts.gate_up_proj.grad.abs().sum() > 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001); optimizer.step()
    saved = tmp_path / 'model.pt'; torch.save(model.state_dict(), saved)
    model.load_state_dict(torch.load(saved, weights_only=True))


def test_full_batch_row_refs_survive_packing_and_loss_reduction(tmp_path):
    import torch.distributed as dist
    from tensordict import TensorDict
    import verl.utils.tensordict_utils as tu
    from add4_opd_full_trainer import FullOPDBatchAdapter
    from add4_opd_full_engine import wrap_full_inputs, wrap_full_outputs
    from add4_opd_engine import Add4OPDOutput, FIELDS
    from add4_opd_trainer import opd_or_sft_loss
    dist.init_process_group('gloo', init_method=f'file://{tmp_path}/dist', rank=0, world_size=1)
    try:
        adapter = FullOPDBatchAdapter.__new__(FullOPDBatchAdapter)
        adapter.rank, adapter.world, adapter.step = 0, 1, 0
        adapter.trainer, adapter.eos, adapter.attempt, adapter.processor = SimpleNamespace(), 3, 'attempt', 'mask'
        adapter.config = dict(full_score_dir=str(tmp_path/'scores'), audit_dir=str(tmp_path/'audit'), run_id='run',
            max_new_tokens=2, max_concurrency=1, temperature=.6, seed=42, valid_vocab_size=4, output_vocab_size=6,
            teacher_score_chunk_size=1, teacher_weight_version='frozen', entropy_coef=.005)
        adapter.verify_services = lambda: None
        adapter.sync_weights = lambda: f'run:{adapter.step}'
        adapter.student = SimpleNamespace(sample=lambda *a, **k: [2,3])
        def request(path, payload):
            split = payload['logprob_start_len'] + 1
            return response_for(payload['input_ids'][:split], payload['input_ids'][split:])
        adapter.teacher = SimpleNamespace(request=request)
        data = TensorDict({k:torch.nested.as_nested_tensor([torch.tensor(v),torch.tensor(v)],layout=torch.jagged)
            for k,v in {'input_ids':[0,1,2,3], 'loss_mask':[0,0,1,1], 'position_ids':[0,1,2,3]}.items()},batch_size=[2])
        data = adapter.prepare(data, lambda t,d: d['input_ids'].offsets().diff().tolist())
        assert data['opd_full_rows'].values().shape == (8,2)
        # Use the same jagged-aware selection as VERL dynamic microbatching.
        micro = tu.index_select_tensor_dict(data, torch.tensor([1,0]))
        inputs,_ = wrap_full_inputs(lambda *a: ({},{'pad_size':0}))(None,micro)
        assert inputs['opd_full_refs'][inputs['opd_mask'].bool()].tolist() == [[1,0],[1,1],[0,0],[0,1]]
        values = torch.arange(8.,requires_grad=True)
        output = Add4OPDOutput(**{name:values for name in FIELDS})
        prepared = wrap_full_outputs(None)(None,output,{},micro,None)
        tu.assign_non_tensor(micro,dp_size=1,batch_num_tokens=4)
        loss,_ = opd_or_sft_loss(None,prepared,micro);loss.backward()
        assert values.grad.tolist() == [0.,.25,.25,0.,0.,.25,.25,0.]
        assert len(list((tmp_path/'scores').rglob('sample-*.npy'))) == 2
    finally:
        dist.destroy_process_group()


def test_full_preflight_requires_complete_teacher_distribution():
    from add4_opd_preflight import check_teacher
    tok = SimpleNamespace(all_special_tokens=[], encode=lambda *a, **k: [0,1,2])
    config = dict(student_model_type='qwen3_5', teacher_served_model_name='teacher',
        teacher_weight_version='frozen', sglang_version='0.5.21', context_length=262144,
        loss_type='full_vocab_reverse_kl', valid_vocab_size=4, output_vocab_size=6)
    def request(path, payload=None):
        if path == '/model_info': return dict(served_model_name='teacher', model_type='qwen4_exp', weight_version='frozen')
        if path == '/server_info': return dict(version='0.5.21', dtype='bfloat16', quantization=None,
            mamba_ssm_dtype='float32', context_length=262144)
        if path == '/tokenize': return dict(tokens=[[0,1,2]] * len(payload['prompt']))
        assert path == '/generate' and payload['top_logprobs_num'] == 6
        return response_for([0,1], [2])
    report = check_teacher(SimpleNamespace(request=request), tok, config)
    assert report['scoring_verified'] and report['scoring_vocab_size'] == 4


def test_actual_vocabulary_width_full_loss_and_padding_gradient(tmp_path):
    valid, total = 248077, 248320
    parsed = parse_full_scores(response_for([0,1], [2], size=total), [0,1], [2], valid, total, 'frozen')
    np.save(tmp_path / 'sample-000000.npy', parsed)
    h = torch.tensor([[.5, -.2]], requires_grad=True)
    w = torch.linspace(-.1,.1,total*2).reshape(total,2).requires_grad_()
    result = chunked_full_loss(h,w,torch.tensor([[0,0]]),torch.tensor([1]),tmp_path,valid,1,.005)
    reference_lp = (h @ w[:valid].T).log_softmax(-1)
    reference = (reference_lp.exp() * (1.005 * reference_lp - torch.from_numpy(parsed))).sum()
    torch.testing.assert_close(result[0].sum(),reference)
    grads = torch.autograd.grad(result[0].sum(), (h,w))
    expected = torch.autograd.grad(reference, (h,w))
    for actual, ref in zip(grads,expected): torch.testing.assert_close(actual,ref,atol=1e-6,rtol=1e-4)
    assert grads[1][valid:].count_nonzero() == 0


@pytest.mark.parametrize('model', ['qwen35','qwen36'])
def test_full_profiles_compose_with_canonical_sft_and_validate(monkeypatch, model):
    from hydra import initialize_config_dir, compose
    from omegaconf import OmegaConf
    from add4_opd_trainer import validate_config
    root = Path(__file__).resolve().parents[2]
    for name,value in dict(VERL_ROOT=str(root), QWEN35_MODEL_DIR='/model35', QWEN36_MODEL_DIR='/model36',
        ADD_TOKENIZER_DIR='/tokenizer', ADD_TRAIN_FILES='/data/train.parquet', ADD_RUN_ID='test',
        ADD_RUN_DIR='/data/zengjiajie/projects/verl/add4-opd/test').items(): monkeypatch.setenv(name,value)
    with initialize_config_dir(config_dir=str(root/'examples/sft/add4'),version_base=None):
        config = compose(config_name=model+'.sft.example')
    OmegaConf.set_struct(config, False)
    config.opd = OmegaConf.load(root / ('examples/sft/add4/opd.full.'+model+'.example.yaml'))
    config.opd.run_id='test'
    config.opd.student_url='http://127.0.0.1:18082'
    config.opd.student_logit_processor_file='/data/mask.json'
    validate_config(config)
    resolved=OmegaConf.to_container(config,resolve=True)
    assert resolved['data']['custom_cls']['path'] == str(root/'examples/sft/add4/add4_sft_dataset.py')
    config.opd.teacher_top_k=128
    with pytest.raises(ValueError, match='teacher_top_k=null'): validate_config(config)
    config.opd.teacher_top_k=None
    config.trainer.nnodes=2
    with pytest.raises(ValueError, match='single-node'): validate_config(config)
    config.trainer.nnodes=1
    config.opd.tail_bucket=True
    with pytest.raises(ValueError, match='does not use a tail bucket'): validate_config(config)
    topk_profile = 'opd.example.yaml' if model == 'qwen35' else 'opd.qwen36.example.yaml'
    config.opd = OmegaConf.load(root / 'examples/sft/add4' / topk_profile)
    config.opd.run_id='test'
    config.opd.student_url='http://127.0.0.1:18082'
    config.opd.student_logit_processor_file='/data/mask.json'
    validate_config(config)
    config.opd.tail_bucket=False
    with pytest.raises(ValueError, match='requires tail_bucket=true'): validate_config(config)
    config.opd.tail_bucket=True
    config.opd.topk_renormalize=True
    with pytest.raises(ValueError, match='without subset softmax'): validate_config(config)
