"""Bounded SGLang requests using raw token IDs, never chat templates."""
import json
import time
import urllib.error
import urllib.parse
import urllib.request

from add4_opd_core import parse_teacher_scores, parse_teacher_topk


class SGLangClient:
    def __init__(self, endpoint, timeout=600):
        parsed = urllib.parse.urlsplit(endpoint)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password or parsed.query:
            raise ValueError('Expected an HTTP endpoint without embedded credentials')
        if timeout <= 0:
            raise ValueError('HTTP timeout must be positive')
        self.endpoint, self.timeout = endpoint.rstrip('/'), timeout
        self.monitor, self.role = None, 'service'
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(self, path, payload=None):
        started, size, ok = time.perf_counter(), 0, False
        request = urllib.request.Request(self.endpoint + path,
            data=None if payload is None else json.dumps(payload).encode(),
            headers={'Content-Type': 'application/json'})
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                raw = response.read()
                size = len(raw)
                result = json.loads(raw)
                ok = isinstance(result, dict)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f'SGLang {path} returned HTTP {error.code}') from None
        except (urllib.error.URLError, TimeoutError):
            raise RuntimeError(f'SGLang {path} connection failed or timed out') from None
        finally:
            if self.monitor:
                self.monitor.http(self.role, path, time.perf_counter() - started, size, ok)
        if not isinstance(result, dict):
            raise ValueError(f'SGLang {path} returned a non-object response')
        return result

    def score(self, prompt_ids, response_ids, valid_size, output_size):
        if not prompt_ids or not response_ids or any(not 0 <= x < valid_size for x in prompt_ids + response_ids):
            raise ValueError('Expected nonempty prompt/response containing only defined token IDs')
        result = self.request('/generate', {
            'input_ids': prompt_ids + response_ids,
            'sampling_params': {'max_new_tokens': 0, 'temperature': 1.0},
            'return_logprob': True, 'return_text_in_logprobs': False,
            'logprob_start_len': len(prompt_ids) - 1,
            'token_ids_logprob': list(range(valid_size, output_size)),
        })
        return parse_teacher_scores(result, prompt_ids, response_ids, valid_size, output_size)

    def score_topk(self, prompt_ids, response_ids, valid_size, output_size, top_k, expected_weight_version=None):
        if not 0 < top_k <= valid_size or not prompt_ids or not response_ids:
            raise ValueError('Invalid teacher top-k request')
        if any(not isinstance(x, int) or not 0 <= x < valid_size for x in prompt_ids + response_ids):
            raise ValueError('Undefined token ID in teacher input')
        result = self.request('/generate', {
            'input_ids': prompt_ids + response_ids,
            'sampling_params': {'max_new_tokens': 0, 'temperature': 1.0},
            'return_logprob': True, 'return_text_in_logprobs': False,
            'logprob_start_len': len(prompt_ids) - 1,
            # Overfetch guarantees k valid IDs even if every padded row ranks first.
            'top_logprobs_num': min(output_size, top_k + output_size - valid_size),
            'token_ids_logprob': list(range(valid_size, output_size)),
        })
        if expected_weight_version is not None and result['meta_info'].get('weight_version') != expected_weight_version:
            raise ValueError('Teacher scoring used a changed or unverified weight version')
        return parse_teacher_topk(result, prompt_ids, response_ids, valid_size, output_size, top_k)

    def sample(self, prompt_ids, config, eos_id, processor, seed, expected_weight_version=None):
        if len(prompt_ids) + config['max_new_tokens'] > config['context_length']:
            raise ValueError('Prompt plus response budget exceeds context; refusing truncation')
        result = self.request('/generate', {
            'input_ids': prompt_ids,
            'sampling_params': {'max_new_tokens': config['max_new_tokens'],
                'temperature': config['temperature'], 'top_p': 1.0, 'top_k': -1,
                'min_p': 0.0, 'stop_token_ids': [eos_id], 'skip_special_tokens': False,
                'no_stop_trim': True, 'sampling_seed': seed,
                'custom_params': {'token_ids': list(range(config['valid_vocab_size'], config['output_vocab_size']))}},
            'custom_logit_processor': processor, 'return_logprob': True,
            'return_text_in_logprobs': False,
        })
        meta = result['meta_info']
        if expected_weight_version is not None and meta.get('weight_version') != expected_weight_version:
            raise ValueError('Student rollout used a stale or unverified weight version')
        ids = [row[1] for row in meta['output_token_logprobs']]
        if self.monitor:
            self.monitor.sample(ids, meta['finish_reason']['type'], eos_id)
        if meta['finish_reason']['type'] != 'stop' or not ids or ids[-1] != eos_id:
            raise ValueError('Student response did not finish at the Add EOS; refusing truncated training')
        if any(not isinstance(x, int) or not 0 <= x < config['valid_vocab_size'] for x in ids):
            raise ValueError('Student generated an undefined token')
        return ids
