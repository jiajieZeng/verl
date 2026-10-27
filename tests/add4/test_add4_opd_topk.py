import copy
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'examples/sft/add4'))
from add4_opd_core import parse_teacher_topk
from add4_opd_client import SGLangClient
from add4_opd_engine import FIELDS, wrap_model_forward, wrap_model_inputs, wrap_model_outputs
from add4_opd_topk import chunked_topk_loss, topk_terms


def topk_response():
    # Padded IDs outrank valid IDs; top-(k + padded_count) must be filtered first.
    row = [[math.log(p), token, None] for p, token in [(.3, 4), (.25, 5), (.2, 2), (.15, 1)]]
    return {'meta_info': {'input_token_logprobs': [[math.log(.2), 2, None], [math.log(.05), 3, None]],
        'input_token_ids_logprobs': [row[:2], row[:2]], 'input_top_logprobs': [row, row]}}


def test_topk_overfetch_excludes_padding_but_preserves_valid_mass():
    result = topk_response()
    parsed = parse_teacher_topk(result, [0, 1], [2, 3], 4, 6, 2)
    assert parsed['ids'] == [[2, 1], [2, 1]]
    assert parsed['logprobs'][0] == pytest.approx([math.log(.2/.45), math.log(.15/.45)])
    assert sum(math.exp(x) for x in parsed['logprobs'][0]) == pytest.approx(.35/.45)
    for field in ('input_token_logprobs', 'input_top_logprobs', 'input_token_ids_logprobs'):
        result['meta_info'][field].insert(0, [None, 1, None] if field == 'input_token_logprobs' else None)
    assert parse_teacher_topk(result, [0, 1], [2, 3], 4, 6, 2) == parsed


@pytest.mark.parametrize('failure', ['duplicate', 'missing', 'unsorted', 'misaligned', 'wrong_token'])
def test_topk_malformed_response_fails(failure):
    result = copy.deepcopy(topk_response())
    if failure == 'duplicate': result['meta_info']['input_top_logprobs'][0][1][1] = 4
    if failure == 'missing': result['meta_info']['input_top_logprobs'][0].pop()
    if failure == 'unsorted': result['meta_info']['input_top_logprobs'][0].reverse()
    if failure == 'misaligned': result['meta_info']['input_top_logprobs'].pop()
    if failure == 'wrong_token': result['meta_info']['input_token_logprobs'][0][1] = 0
    with pytest.raises(ValueError): parse_teacher_topk(result, [0, 1], [2, 3], 4, 6, 2)


def test_http_topk_contract_has_no_template_or_sampling_truncation():
    seen = []
    client = SGLangClient('http://127.0.0.1:1')
    client.request = lambda path, payload: seen.append((path, payload)) or topk_response()
    client.score_topk([0, 1], [2, 3], 4, 6, 2)
    path, request = seen[0]
    assert path == '/generate' and request['input_ids'] == [0, 1, 2, 3]
    assert request['top_logprobs_num'] == 4 and request['token_ids_logprob'] == [4, 5]
    assert request['sampling_params'] == {'max_new_tokens': 0, 'temperature': 1.0}
    assert 'text' not in request and 'messages' not in request
    with pytest.raises(ValueError, match='weight version'):
        client.score_topk([0, 1], [2, 3], 4, 6, 2, expected_weight_version='frozen')


def test_reverse_kl_includes_tail_and_differs_from_partial_or_subset_kl():
    p, q = torch.tensor([[.2, .3, .5]], dtype=torch.double), torch.tensor([[.6, .3, .1]], dtype=torch.double)
    ids = torch.tensor([[0, 1]])
    kl, _, mass, _ = topk_terms(p.log(), ids, q[:, :2].log(), 3)
    expected = (p[:, :2] * (p[:, :2].log() - q[:, :2].log())).sum()
    assert expected.item() < 0  # Regression: the discarded partial sum is negative.
    assert kl.item() == pytest.approx((expected + .5 * math.log(.5/.1)).item())
    assert kl.item() >= 0
    assert mass.item() == pytest.approx(.5)
    forward = (q[:, :2] * (q[:, :2].log() - p[:, :2].log())).sum()
    pn, qn = p[:, :2] / .5, q[:, :2] / .9
    subset = (pn * (pn.log() - qn.log())).sum()
    for other in (forward, subset, expected): assert kl.item() != pytest.approx(other.item())


@pytest.mark.parametrize('chunk_size', [1, 2, 128])
@pytest.mark.parametrize('entropy_coef', [0., .005])
def test_chunked_loss_and_gradients_match_dense_autograd(chunk_size, entropy_coef):
    torch.manual_seed(11)
    hidden = torch.randn(1, 6, 3, dtype=torch.double, requires_grad=True)
    weight = torch.randn(7, 3, dtype=torch.double, requires_grad=True)
    ids = torch.tensor([[1, 2], [2, 4], [0, 1], [1, 3], [2, 4], [3, 4]])
    teacher = torch.tensor([[-.8, -1.4]] * 6, dtype=torch.double, requires_grad=True)
    mask = torch.tensor([0, 1, 1, 0, 1, 0])
    output = chunked_topk_loss(hidden, weight, ids, teacher, mask, 5, chunk_size, entropy_coef)
    upstream = torch.tensor([.1, .3, .7, .2, .9, .4], dtype=torch.double)
    grads = torch.autograd.grad((output[0] * upstream).sum(), (hidden, weight, teacher), allow_unused=True)
    kl, entropy, mass, teacher_mass = topk_terms(hidden.reshape(-1, 3) @ weight.T, ids, teacher, 5)
    expected = (kl - entropy_coef * entropy) * mask
    dense_grads = torch.autograd.grad((expected * upstream).sum(), (hidden, weight))
    for value, ref in zip(output, (expected, kl * mask, entropy * mask, mass * mask, teacher_mass * mask)):
        torch.testing.assert_close(value, ref)
    for grad, ref in zip(grads[:2], dense_grads): torch.testing.assert_close(grad, ref)
    assert grads[2] is None
    assert grads[1][5:].count_nonzero() == 0
    assert grads[0].reshape(-1, 3)[~mask.bool()].count_nonzero() == 0


def test_chunked_reverse_kl_gradcheck():
    h = torch.randn(3, 2, dtype=torch.double, requires_grad=True)
    w = torch.randn(5, 2, dtype=torch.double, requires_grad=True)
    ids = torch.tensor([[0, 1], [1, 2], [0, 2]])
    q = torch.full((3, 2), -1.5, dtype=torch.double)
    assert torch.autograd.gradcheck(lambda a, b: chunked_topk_loss(a, b, ids, q, torch.ones(3), 3, 2, .005)[0], (h, w))


def test_packed_engine_hooks_real_optimizer_checkpoint_and_validation(tmp_path):
    from tensordict import TensorDict
    import verl.utils.tensordict_utils as tu
    from add4_opd_trainer import opd_or_sft_loss
    input_rows = [torch.tensor([0, 1, 2, 3]), torch.tensor([1, 2, 3])]
    masks = [torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1, 1])]
    ids = [torch.tensor([[0, 0], [0, 0], [2, 1], [3, 0]]), torch.tensor([[0, 0], [2, 1], [3, 0]])]
    qs = [torch.tensor([[0., 0.], [0., 0.], [-.8, -1.2], [-.7, -1.4]]),
          torch.tensor([[0., 0.], [-.8, -1.2], [-.7, -1.4]])]
    data = TensorDict({name: torch.nested.as_nested_tensor(values, layout=torch.jagged)
                      for name, values in [('input_ids', input_rows), ('loss_mask', masks),
                                           ('opd_teacher_ids', ids), ('opd_teacher_logp', qs)]}, batch_size=[2])
    tu.assign_non_tensor(data, batch_num_tokens=4, dp_size=1)
    original_inputs = lambda self, batch: ({'input_ids': batch['input_ids'].values()[None]}, {'pad_size': 0})
    inputs, args = wrap_model_inputs(original_inputs)(None, data)
    assert inputs['opd_mask'].tolist() == [0, 1, 1, 0, 1, 1, 0]
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(6, 3)
            self.lm_head = torch.nn.Linear(3, 6, bias=False)
        def model(self, tokens, **kwargs): return (self.embedding(tokens),)
    model = Model()
    validation_calls = []
    original = lambda self, **kwargs: validation_calls.append(kwargs) or 'original validation'
    forward = wrap_model_forward(original, 4, 6, 2, .005)
    output = forward(model, **inputs)
    from torch.utils._pytree import tree_flatten
    assert any(x is output.opd_losses for x in tree_flatten(output)[0])
    converted = wrap_model_outputs(lambda *args: 'original outputs')(None, output, args, data, None)
    assert set(converted) == set(FIELDS)
    loss, metrics = opd_or_sft_loss(None, converted, data)
    assert 'opd/reverse_kl' in metrics
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    before = model.lm_head.weight.detach().clone()
    loss.backward()
    assert model.lm_head.weight.grad[4:].count_nonzero() == 0
    optimizer.step()
    assert not torch.equal(model.lm_head.weight, before)
    checkpoint = tmp_path / 'topk.pt'
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, checkpoint)
    restored = Model()
    saved = torch.load(checkpoint, weights_only=True)
    restored.load_state_dict(saved['model'])
    torch.testing.assert_close(forward(restored, **inputs).opd_losses, forward(model, **inputs).opd_losses)
    assert forward(model, input_ids=torch.tensor([[0, 1]])) == 'original validation'
    assert len(validation_calls) == 1 and 'opd_ids' not in validation_calls[0]


def test_top128_real_head_width_without_context_times_vocab_storage():
    # Real 248320-row head on CPU, tiny hidden width; actual teacher support width 128.
    h = torch.randn(1, 3, 2, requires_grad=True)
    w = torch.randn(248320, 2, requires_grad=True)
    ids = torch.arange(128).repeat(3, 1)
    q = torch.full((3, 128), -8.)
    loss = chunked_topk_loss(h, w, ids, q, torch.tensor([0, 1, 1]), 248077, 1, .005)[0].sum()
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(w.grad).all()
    assert w.grad[248077:].count_nonzero() == 0


@pytest.fixture(params=['qwen3_5', 'qwen3_5_moe'])
def service_pair(request):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread
    import json
    states, clients, servers, threads = {}, {}, [], []
    tokenizer = SimpleNamespace(all_special_tokens=['<|im_end|>'], encode=lambda text, **kwargs: [0, 1, 2, 3])
    config = dict(student_served_model_name='add4-opd-student', teacher_served_model_name='Qwen3.8-Flash-Next',
                  sglang_version='0.5.21', context_length=262144, valid_vocab_size=4, output_vocab_size=6,
                  teacher_top_k=2, sync_dir='/data/zengjiajie/opd/weights', run_id='test', student_model_type=request.param,
                  teacher_weight_version='frozen')
    def handler(state):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_GET(self): self.respond(None)
            def do_POST(self): self.respond(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            def respond(self, payload):
                state['seen'].append((self.command, self.path, payload))
                if self.path == '/model_info': result = state['identity']
                elif self.path == '/server_info': result = state['settings']
                elif self.path == '/tokenize':
                    result = {'tokens': [[0, 1, 2, 3] if not state['bad_tokens'] else [1] for _ in payload['prompt']]}
                elif self.path == '/generate':
                    result = topk_response()
                    for key in ('input_token_logprobs', 'input_token_ids_logprobs', 'input_top_logprobs'):
                        result['meta_info'][key] = result['meta_info'][key][-1:]
                    result['meta_info']['weight_version'] = state['identity']['weight_version']
                else:
                    self.send_response(404); self.end_headers(); return
                self.send_response(200); self.end_headers()
                self.wfile.write(json.dumps(result).encode())
        return Handler
    for role, model_type in [('student', config['student_model_type']), ('teacher', 'qwen4_exp')]:
        state = {'identity': {'model_type': model_type, 'served_model_name': config[role + '_served_model_name'],
                             'weight_version': 'frozen' if role == 'teacher' else 'test:initial', 'model_path': '/' + role},
                 'settings': {'version': '0.5.21', 'dtype': 'bfloat16', 'context_length': 262144,
                              'quantization': None, 'mamba_ssm_dtype': 'float32', 'enable_custom_logit_processor': role == 'student',
                              'api_key': 'must-not-be-persisted'}, 'seen': [], 'bad_tokens': False}
        server = ThreadingHTTPServer(('127.0.0.1', 0), handler(state))
        thread = Thread(target=server.serve_forever, daemon=True); thread.start()
        states[role], clients[role] = state, SGLangClient(f'http://127.0.0.1:{server.server_port}')
        servers.append(server); threads.append(thread)
    yield states, clients, tokenizer, config
    for server in servers: server.shutdown(); server.server_close()
    for thread in threads: thread.join()


def test_independent_http_services_preflight_read_only_and_sanitized(service_pair):
    from add4_opd_preflight import check_services
    states, clients, tokenizer, config = service_pair
    report = check_services(clients['student'], clients['teacher'], tokenizer, config)
    assert report['scoring_top_k'] == 2
    assert 'api_key' not in str(report) and 'must-not-be-persisted' not in str(report)
    for state in states.values():
        assert all(path in ('/model_info', '/server_info', '/tokenize', '/generate') for _, path, _ in state['seen'])
    # Resume is compatible with native SGLang's served-name change after weight reload.
    states['student']['identity'].update(model_path=config['sync_dir'] + '/test/slot-0',
                                        served_model_name=config['sync_dir'] + '/test/slot-0', weight_version='test:8')
    check_services(clients['student'], clients['teacher'], tokenizer, config)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: clients['teacher'].score_topk(
            [0, 1, 2], [3], 4, 6, 2, expected_weight_version='frozen'), range(8)))
    assert len(results) == 8 and all(result['ids'] == [[2, 1]] for result in results)


@pytest.mark.parametrize('failure', ['wrong_model', 'wrong_name', 'version', 'dtype', 'quantization',
                                    'context', 'state_dtype', 'tokenizer', 'missing_teacher_version',
                                    'wrong_teacher_version', 'missing_mask'])
def test_service_mismatch_stops_before_weight_update(service_pair, failure):
    from add4_opd_preflight import check_services
    states, clients, tokenizer, config = service_pair
    state = states['teacher']
    if failure == 'wrong_model': state['identity']['model_type'] = 'qwen3_5'
    if failure == 'wrong_name': state['identity']['served_model_name'] = 'some-other-model'
    if failure == 'version': state['settings']['version'] = '0.5.20'
    if failure == 'dtype': state['settings']['dtype'] = 'float16'
    if failure == 'quantization': state['settings']['quantization'] = 'fp8'
    if failure == 'context': state['settings']['context_length'] = 8192
    if failure == 'state_dtype': state['settings']['mamba_ssm_dtype'] = 'bfloat16'
    if failure == 'tokenizer': state['bad_tokens'] = True
    if failure == 'missing_teacher_version': state['identity']['weight_version'] = None
    if failure == 'wrong_teacher_version': state['identity']['weight_version'] = 'another-checkpoint'
    if failure == 'missing_mask': states['student']['settings']['enable_custom_logit_processor'] = False
    with pytest.raises(ValueError): check_services(clients['student'], clients['teacher'], tokenizer, config)
    assert all('update_weights' not in path for state in states.values() for _, path, _ in state['seen'])


def test_teacher_only_preflight_does_not_need_student(service_pair):
    from add4_opd_preflight import check_teacher
    states, clients, tokenizer, config = service_pair
    result = check_teacher(clients['teacher'], None, config)
    assert result['scoring_verified'] is False and result['scope'] == 'service_metadata_only'
    result = check_teacher(clients['teacher'], tokenizer, config)
    assert result['scoring_verified'] is True and result['scoring_top_k'] == 2
    assert states['student']['seen'] == []
    assert all(path in ('/model_info', '/server_info', '/tokenize', '/generate')
               for _, path, _ in states['teacher']['seen'])


def test_changed_service_identity_blocks_mutations(service_pair):
    from add4_opd_preflight import check_services
    from add4_opd_trainer import OPDBatchAdapter
    states, clients, tokenizer, config = service_pair
    report = check_services(clients['student'], clients['teacher'], tokenizer, config)
    adapter = OPDBatchAdapter.__new__(OPDBatchAdapter)
    adapter.teacher, adapter.student = clients['teacher'], clients['student']
    adapter.teacher_identity = report['identities']['teacher']
    adapter.student_identity = report['identities']['student']
    adapter.rank_zero = lambda fn: fn()
    adapter.verify_services()
    states['teacher']['identity']['weight_version'] = 'changed'
    with pytest.raises(ValueError, match='Frozen teacher'): adapter.verify_services()
    states['teacher']['identity']['weight_version'] = 'frozen'
    states['student']['identity']['model_path'] = '/other'
    with pytest.raises(ValueError, match='Student endpoint'): adapter.verify_services()


def test_metrics_sum_microbatches_then_average_data_parallel_ranks():
    import verl.utils.tensordict_utils as tu
    from add4_opd_engine import wrap_training_metrics
    output = tu.get_tensordict(tensor_dict={}, non_tensor_dict={
        'metrics': {'opd/reverse_kl': [[.1, .2], [.3, .6]], 'loss': .7}})
    result = wrap_training_metrics(lambda: output, lambda: 2)()
    assert tu.get(result, 'metrics') == pytest.approx({'opd/reverse_kl': .6, 'loss': .7})
