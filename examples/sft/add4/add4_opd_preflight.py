"""Read-only checks across independent SGLang HTTP services; no teacher filesystem access."""
from pathlib import Path


def service_identity(client):
    info = client.request('/model_info')
    return {key: info.get(key) for key in ('served_model_name', 'model_type', 'model_path', 'weight_version')}


def check_services(student, teacher, tokenizer, config):
    if tokenizer is None:
        raise ValueError('Training preflight requires the student tokenizer and teacher scoring check')
    return _check_services({'student': student, 'teacher': teacher}, tokenizer, config)


def check_teacher(teacher, tokenizer, config):
    """Check the remote teacher without a student process or teacher filesystem."""
    return _check_services({'teacher': teacher}, tokenizer, config)


def _check_services(clients, tokenizer, config):
    identities, settings = {}, {}
    for role, client in clients.items():
        expected_type = config['student_model_type'] if role == 'student' else 'qwen4_exp'
        expected_name = config[role + '_served_model_name']
        identity = service_identity(client)
        expected_names = {expected_name}
        if role == 'student':
            # SGLang 0.5.21 updates served_model_name to the new model_path on reload.
            slots = {str(Path(config['sync_dir']) / config['run_id'] / f'slot-{i}') for i in (0, 1)}
            if identity['model_path'] in slots and str(identity['weight_version']).startswith(config['run_id'] + ':'):
                expected_names.add(identity['model_path'])
        if identity['model_type'] != expected_type or identity['served_model_name'] not in expected_names:
            raise ValueError(f'{role} endpoint serves the wrong model; refusing weight update')
        if role == 'teacher' and (not config.get('teacher_weight_version') or
                                 identity['weight_version'] != config['teacher_weight_version']):
            raise ValueError('Teacher weight_version differs from the configured frozen teacher')
        identities[role] = identity
        raw = client.request('/server_info')
        # Never persist the whole server_info: it can contain unrelated launch credentials.
        fields = ('version', 'dtype', 'context_length', 'quantization', 'mamba_ssm_dtype',
                  'attention_backend', 'sampling_backend', 'tp_size', 'ep_size', 'enable_custom_logit_processor')
        settings[role] = {name: raw.get(name) for name in fields}
        if raw.get('version') != config['sglang_version'] or raw.get('dtype') != 'bfloat16':
            raise ValueError(f'{role} SGLang version/dtype differs from the agreed inference profile')
        if raw.get('quantization') is not None or raw.get('mamba_ssm_dtype') != 'float32':
            raise ValueError(f'{role} quantization or recurrent-state dtype differs from the inference profile')
        if not isinstance(raw.get('context_length'), int) or raw['context_length'] < config['context_length']:
            raise ValueError(f'{role} context capacity is insufficient')
        if role == 'student' and raw.get('enable_custom_logit_processor') is not True:
            raise ValueError('Student must enable the undefined-token mask processor')
    if tokenizer is None:
        return {'identities': identities, 'inference_settings': settings, 'tokenizer_probe_count': 0,
                'scoring_verified': False, 'scope': 'service_metadata_only'}
    probes = ['中文 facts：文件 /src/a.py；修复 issue #128。', '{"facts":["x = 1\\ny = 2"]}',
              '<|im_start|>assistant\n{"facts":[]}<|im_end|>', ' spaces\t tabs\nemoji 🧠 café']
    probes.extend(tokenizer.all_special_tokens)
    expected = [tokenizer.encode(text, add_special_tokens=False) for text in probes]
    for role, client in clients.items():
        response = client.request('/tokenize', {'model': identities[role]['served_model_name'],
                                  'prompt': probes, 'add_special_tokens': False})
        if response.get('tokens') != expected:
            raise ValueError(f'{role} tokenizer differs from the training tokenizer')
    # Exercise scoring shape, padding normalization and the full requested support width.
    probe_ids = expected[2]
    if config.get('loss_type') == 'full_vocab_reverse_kl':
        from add4_opd_full import full_request, parse_full_scores
        result = clients['teacher'].request('/generate', full_request(probe_ids[:-1], probe_ids[-1:], config['output_vocab_size']))
        scores = parse_full_scores(result, probe_ids[:-1], probe_ids[-1:], config['valid_vocab_size'],
                                   config['output_vocab_size'], config['teacher_weight_version'])
        return {'identities': identities, 'inference_settings': settings, 'tokenizer_probe_count': len(probes),
                'scoring_verified': True, 'scoring_vocab_size': scores.shape[-1],
                'teacher_transport': 'native_sglang_full_vocabulary_base64_fp32'}
    score = clients['teacher'].score_topk(probe_ids[:-1], probe_ids[-1:], config['valid_vocab_size'],
                              config['output_vocab_size'], config['teacher_top_k'],
                              expected_weight_version=identities['teacher']['weight_version'])
    return {'identities': identities, 'inference_settings': settings, 'tokenizer_probe_count': len(probes),
            'scoring_top_k': len(score['ids'][0]), 'scoring_verified': True,
            'teacher_transport': 'native_sglang_http_raw_input_ids'}
