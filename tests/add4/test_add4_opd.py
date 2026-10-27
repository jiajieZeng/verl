import copy
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib
import json
import math
from pathlib import Path
import sys
import threading
import time

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[2] / 'examples/sft/add4'
sys.path.insert(0, str(SCRIPTS))
from add4_opd_core import (parse_teacher_scores, prompt_from_sft_ids, renormalize_logprob,
                           restrict_fused_head, sampled_opd_loss, valid_token_logprobs, validate_vocabulary)
from add4_opd_client import SGLangClient


def score_response(response=(2, 3)):
    return {'meta_info': {'input_token_logprobs': [[math.log(.2), x, None] for x in response],
        'input_token_ids_logprobs': [[[math.log(.1), 4, None], [math.log(.1), 5, None]] for _ in response]}}


def test_actual_ids_are_preserved_and_fixed_labels_removed():
    assert prompt_from_sft_ids([10, 11, 12, 13, 14], [0, 0, 1, 1, 0]) == [10, 11]
    with pytest.raises(ValueError):
        prompt_from_sft_ids([1, 2, 3, 4], [0, 1, 0, 1])


def test_valid_vocabulary_requires_contiguous_id_mapping():
    class Tokenizer:
        def get_vocab(self): return {'a': 0, 'b': 2}
    with pytest.raises(ValueError):
        validate_vocabulary(Tokenizer(), 2, 4)


def test_teacher_renormalization_matches_direct_mask():
    logits = torch.tensor([1., 2., 3., 4., 5., 6.], dtype=torch.float64)
    full = logits.log_softmax(0)
    actual = renormalize_logprob(full[2].item(), full[4:].tolist())
    assert actual == pytest.approx(logits[:4].log_softmax(0)[2].item())
    assert renormalize_logprob(-2, [-math.inf, -math.inf]) == -2


def test_teacher_token_alignment_and_missing_rows():
    result = score_response()
    assert parse_teacher_scores(result, [0, 1], [2, 3], 4, 6) == pytest.approx([math.log(.25)] * 2)
    for field in ('input_token_logprobs', 'input_token_ids_logprobs'):
        malformed = copy.deepcopy(result)
        malformed['meta_info'][field].pop()
        with pytest.raises(ValueError):
            parse_teacher_scores(malformed, [0, 1], [2, 3], 4, 6)
    with pytest.raises(ValueError, match='ID mismatch'):
        parse_teacher_scores(result, [0, 1], [3, 2], 4, 6)


def test_teacher_predecessor_does_not_shift_response():
    result = score_response()
    result['meta_info']['input_token_logprobs'].insert(0, [None, 1, None])
    result['meta_info']['input_token_ids_logprobs'].insert(0, None)
    assert parse_teacher_scores(result, [0, 1], [2, 3], 4, 6) == pytest.approx([math.log(.25)] * 2)


def test_padded_head_rows_have_no_gradient_and_no_normalization_effect():
    logits = torch.tensor([[0., 1., 2., 100., 200.]], requires_grad=True)
    selected = valid_token_logprobs(logits, torch.tensor([1]), 3, .6)
    (-selected.sum()).backward()
    assert logits.grad[0, 3:].tolist() == [0., 0.]
    assert selected.item() == pytest.approx((logits.detach()[0, :3] / .6).log_softmax(0)[1].item())


def test_sampled_loss_direction_teacher_detach_and_prompt_mask():
    current = torch.tensor([-100., -2., -1.], requires_grad=True)
    old = current.detach().clone().requires_grad_()
    teacher = torch.tensor([float('nan'), -1., -2.], requires_grad=True)
    loss = sampled_opd_loss(current, old, teacher, torch.tensor([0, 1, 1]))
    loss.backward()
    assert current.grad.tolist() == pytest.approx([0, -.5, .5])
    assert old.grad is None and teacher.grad is None
    with pytest.raises(ValueError):
        sampled_opd_loss(current, old, teacher, torch.zeros(3))


def test_head_view_retains_parameter_shape():
    weight = torch.randn(6, 3, requires_grad=True)
    def original(self, hidden_states, vocab_weights, input_ids, temperature):
        return valid_token_logprobs(hidden_states @ vocab_weights.T, input_ids, 4, temperature)
    forward = restrict_fused_head(original, 4, 6)
    loss = -forward(None, torch.ones(1, 3), weight, torch.tensor([2]), .6).sum()
    loss.backward()
    assert weight.shape == (6, 3)
    assert torch.count_nonzero(weight.grad[4:]) == 0


@pytest.fixture
def http_server():
    seen, lock = [], threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            with lock: seen.append(payload)
            if self.path == '/slow':
                time.sleep(.2)
            if self.path == '/error':
                self.send_response(503); self.end_headers(); return
            result = score_response()
            self.send_response(200); self.send_header('Content-Type', 'application/json'); self.end_headers()
            try: self.wfile.write(json.dumps(result).encode())
            except BrokenPipeError: pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    yield f'http://127.0.0.1:{server.server_port}', seen
    server.shutdown(); server.server_close(); thread.join()


def test_real_http_concurrency_raw_ids_and_proxy_bypass(http_server, monkeypatch):
    endpoint, seen = http_server
    monkeypatch.setenv('HTTP_PROXY', 'http://127.0.0.1:1')
    client = SGLangClient(endpoint)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: client.score([0, 1], [2, 3], 4, 6), range(8)))
    assert len(results) == len(seen) == 8
    assert all(r['input_ids'] == [0, 1, 2, 3] and 'messages' not in r and 'text' not in r for r in seen)
    assert all(r['token_ids_logprob'] == [4, 5] for r in seen)


def test_timeout_and_http_failure_propagate(http_server):
    endpoint, _ = http_server
    with pytest.raises(RuntimeError, match='timed out'):
        SGLangClient(endpoint, timeout=.03).request('/slow', {})
    with pytest.raises(RuntimeError, match='503'):
        SGLangClient(endpoint).request('/error', {})


def test_sample_contract_and_eos_not_truncated():
    client = SGLangClient('http://localhost:1')
    seen = []
    def fake(path, payload):
        seen.append(payload)
        return {'meta_info': {'finish_reason': {'type': 'stop'}, 'output_token_logprobs': [[-1., 2, None], [-1., 3, None]]}}
    client.request = fake
    config = dict(max_new_tokens=8, context_length=32, temperature=.6, valid_vocab_size=4, output_vocab_size=6)
    assert client.sample([0, 1], config, 3, 'processor', 123) == [2, 3]
    with pytest.raises(ValueError, match='stale'):
        client.sample([0, 1], config, 3, 'processor', 123, expected_weight_version='step-2')
    params = seen[0]['sampling_params']
    assert params['sampling_seed'] == 123 and params['top_p'] == 1.0 and params['top_k'] == -1
    assert params['custom_params']['token_ids'] == [4, 5]
    with pytest.raises(ValueError, match='context'):
        client.sample([0] * 30, config, 3, 'processor', 123)


def test_cpu_optimizer_checkpoint_resume_smoke(tmp_path):
    torch.manual_seed(7)
    model = torch.nn.Linear(3, 6, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    inputs = torch.randn(2, 3)
    labels = torch.tensor([1, 2])
    before = model.weight.detach().clone()
    current = valid_token_logprobs(model(inputs), labels, 4, .6)
    old = current.detach().clone()
    teacher = old + torch.tensor([.2, -.3])
    loss = sampled_opd_loss(current, old, teacher, torch.ones(2))
    loss.backward(); optimizer.step(); optimizer.zero_grad()
    assert not torch.equal(before[:4], model.weight[:4])
    checkpoint = tmp_path / 'checkpoint.pt'
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': 1}, checkpoint)
    restored = torch.nn.Linear(3, 6, bias=False)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=.01)
    state = torch.load(checkpoint, weights_only=True)
    restored.load_state_dict(state['model']); restored_optimizer.load_state_dict(state['optimizer'])
    torch.testing.assert_close(restored(inputs), model(inputs))
    assert state['step'] == 1 and restored_optimizer.state_dict()['state']


def test_import_does_not_install_training_hooks():
    original = importlib.import_module('train_add4').root_owned_lm_head
    importlib.import_module('train_add4_opd')
    assert importlib.import_module('train_add4').root_owned_lm_head is original


def test_adapter_weight_reload_and_target_alignment(tmp_path):
    from types import SimpleNamespace
    import torch.distributed as dist
    from safetensors.torch import load_file
    from tensordict import TensorDict
    import verl.utils.tensordict_utils as tu
    from add4_opd_trainer import OPDBatchAdapter, opd_or_sft_loss

    dist.init_process_group('gloo', init_method=f'file://{tmp_path}/gloo', rank=0, world_size=1)
    try:
        weight = torch.nn.Parameter(torch.ones(6, 2))
        config_writer = SimpleNamespace(save_pretrained=lambda folder: (folder / 'config.json').write_text('{}'))
        engine = SimpleNamespace(get_per_tensor_param=lambda: (iter([('lm_head.weight', weight)]), None))
        updates = []
        class Student:
            def request(self, path, payload):
                assert path == '/update_weights_from_disk'
                loaded = load_file(str(Path(payload['model_path']) / 'weights-00000.safetensors'))
                torch.testing.assert_close(loaded['lm_head.weight'], weight.detach().to(torch.bfloat16))
                updates.append(payload['weight_version'])
                return {'success': True}
            def sample(self, prompt, config, eos, processor, seed, expected_weight_version):
                assert expected_weight_version == updates[-1]
                return [2, eos]
        trainer = SimpleNamespace(engine=engine, model_config=SimpleNamespace(hf_config=config_writer, tokenizer=config_writer),
                                  training_client=SimpleNamespace())
        adapter = OPDBatchAdapter.__new__(OPDBatchAdapter)
        adapter.trainer, adapter.rank, adapter.world, adapter.step = trainer, 0, 1, 0
        adapter.config = dict(sync_dir=str(tmp_path / 'weights'), audit_dir=str(tmp_path / 'audit'), run_id='smoke',
                              temperature=.6, seed=42, max_concurrency=2, valid_vocab_size=4, output_vocab_size=6,
                              teacher_top_k=2, loss_type='teacher_topk_reverse_kl', tail_bucket=True, entropy_coef=.005,
                              export_shard_bytes=1024, student_model_type='qwen3_5')
        adapter.attempt, adapter.eos, adapter.processor = 'attempt', 3, 'processor'
        adapter.student = Student()
        adapter.teacher_identity = {'weight_version': 'teacher-v1'}
        adapter.teacher = SimpleNamespace(score_topk=lambda *args, **kwargs: {
            'ids': [[2, 1], [3, 0]], 'logprobs': [[-.8, -1.2], [-.7, -1.4]]})
        adapter.verify_services = lambda: None
        def batch():
            values = {'input_ids': [[0, 1, 2, 3, 0]], 'loss_mask': [[0, 0, 1, 1, 0]],
                      'position_ids': [[0, 1, 2, 3, 4]]}
            return TensorDict({k: torch.nested.as_nested_tensor([torch.tensor(x) for x in v], layout=torch.jagged)
                               for k, v in values.items()}, batch_size=[1])
        lengths = lambda trainer, data: data['input_ids'].offsets().diff().tolist()
        data = adapter.prepare(batch(), lengths)
        assert data['input_ids'].values().tolist() == [0, 1, 2, 3]
        assert data['loss_mask'].values().tolist() == [0, 0, 1, 1]
        torch.testing.assert_close(data['opd_teacher_logp'].values(), torch.tensor([[0., 0.], [0., 0.], [-.8, -1.2], [-.7, -1.4]]))
        assert data['opd_teacher_ids'].values().tolist() == [[0, 0], [0, 0], [2, 1], [3, 0]]
        assert 'opd_old_logp' not in data
        assert tu.get(data, 'temperature') == 1.0
        assert tu.get(data, 'return_model_output') is False
        current = torch.tensor([0., -.1, -.2, 0.], requires_grad=True)
        prediction = torch.nested.nested_tensor_from_jagged(current, torch.tensor([0, 4]))
        tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=2)
        loss, _ = opd_or_sft_loss(None, {'opd_losses': prediction}, data)
        loss.backward()
        assert current.grad.tolist() == pytest.approx([0, .5, .5, 0])
        with torch.no_grad(): weight.add_(1)
        adapter.prepare(batch(), lengths)
        assert updates == ['smoke:1', 'smoke:2']
        assert len(list((tmp_path / 'weights/smoke').iterdir())) == 2
        assert len(list((tmp_path / 'audit/smoke/attempt').glob('step-*.json'))) == 2
        with pytest.raises(RuntimeError, match='no optimizer update'):
            adapter.rank_zero(lambda: (_ for _ in ()).throw(ValueError('failed')))
    finally:
        dist.destroy_process_group()


def _distributed_failure_worker(rank, store, output):
    import datetime
    import torch.distributed as dist
    from add4_opd_trainer import OPDBatchAdapter
    dist.init_process_group('gloo', init_method=f'file://{store}', rank=rank, world_size=2,
                            timeout=datetime.timedelta(seconds=15))
    try:
        adapter = OPDBatchAdapter.__new__(OPDBatchAdapter)
        adapter.rank = rank
        try:
            adapter.rank_zero(lambda: (_ for _ in ()).throw(ValueError('test failure')))
        except RuntimeError as error:
            Path(output, f'rank-{rank}.txt').write_text(str(error))
        else:
            raise AssertionError('Expected rank-zero failure on every rank')
    finally:
        dist.destroy_process_group()


def test_two_rank_failure_is_broadcast_without_hanging(tmp_path):
    import multiprocessing
    ctx = multiprocessing.get_context('spawn')
    processes = [ctx.Process(target=_distributed_failure_worker, args=(rank, str(tmp_path / 'store'), str(tmp_path)))
                 for rank in range(2)]
    try:
        for process in processes: process.start()
        for process in processes:
            process.join(timeout=25)
            assert process.exitcode == 0
        assert all('no optimizer update' in (tmp_path / f'rank-{rank}.txt').read_text() for rank in range(2))
    finally:
        for process in processes:
            if process.is_alive(): process.terminate(); process.join(timeout=5)
