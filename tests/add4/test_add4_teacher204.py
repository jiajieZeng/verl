"""The two student profiles use one pinned remote teacher and an isolated SSH forward."""
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

SFT = Path(__file__).resolve().parents[2] / 'examples/sft/add4'


def test_student_profiles_share_the_pinned_204_teacher():
    configs = [yaml.safe_load((SFT / name).read_text()) for name in
               ('opd.example.yaml', 'opd.qwen36.example.yaml', 'opd.full.qwen35.example.yaml', 'opd.full.qwen36.example.yaml')]
    for config in configs:
        assert config['teacher_url'] == 'http://127.0.0.1:18081'
        assert config['teacher_served_model_name'] == 'Qwen3.8-Flash-Next'
        assert config['teacher_weight_version'] == 'qwen38-frozen'
        assert config['sglang_version'] == '0.5.21'
    assert {c['student_model_type'] for c in configs} == {'qwen3_5', 'qwen3_5_moe'}


def test_tunnel_targets_only_204_and_removes_proxies(tmp_path):
    fake = tmp_path / 'ssh'
    fake.write_text(f'#!{sys.executable}\nimport json,os,sys\n'
                    'print(json.dumps({"args":sys.argv[1:],"proxies":'
                    '[k for k in os.environ if k.lower() in ("http_proxy","https_proxy","all_proxy")]}))\n')
    fake.chmod(0o700)
    env = dict(os.environ, PATH=str(tmp_path) + ':' + os.environ['PATH'],
               OPD_TEACHER_SSH_IDENTITY='/private/existing-key')
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        env[key] = 'http://must-not-be-used.invalid'
    result = subprocess.run(['bash', str(SFT / 'start_opd_teacher_204_tunnel.sh')],
                            env=env, capture_output=True, text=True, check=True, timeout=5)
    record = json.loads(result.stdout)
    assert record['proxies'] == []
    args = record['args']
    assert args[-1] == 'zengjiajie@47.103.58.204'
    assert args[args.index('-L') + 1] == '127.0.0.1:18081:127.0.0.1:18082'
    assert 'ExitOnForwardFailure=yes' in args and 'StrictHostKeyChecking=yes' in args
    assert args[args.index('-S') + 1] == 'none'
    assert args[args.index('-i') + 1] == '/private/existing-key'
