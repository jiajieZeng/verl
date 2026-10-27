"""Validate the configured 204 teacher over HTTP; never load or mutate model weights."""
import argparse
import json
import os
from pathlib import Path
import time

from add4_opd_client import SGLangClient
from add4_opd_preflight import check_teacher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, help='Local training tokenizer; required for the full scoring check')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=30)
    args = parser.parse_args()
    proxies = ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy')
    for key in proxies:
        os.environ.pop(key, None)
    import yaml
    config = yaml.safe_load(args.config.read_text())
    report = {'time': time.time(), 'teacher_url': config['teacher_url'],
              'proxies_absent': {key: key not in os.environ for key in proxies}}
    try:
        tokenizer = None
        if args.tokenizer:
            from transformers import AutoTokenizer
            from add4_opd_core import validate_vocabulary
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
            validate_vocabulary(tokenizer, config['valid_vocab_size'], config['output_vocab_size'])
        result = check_teacher(SGLangClient(config['teacher_url'], args.timeout), tokenizer, config)
        report.update(status='scoring_ready' if result['scoring_verified'] else 'metadata_verified', **result)
    except Exception as error:
        # Do not persist arbitrary server responses or credentials in exception strings.
        report.update(status='not_ready', error_type=type(error).__name__)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
        raise SystemExit(f'Teacher check failed ({type(error).__name__}); see {args.report}') from None
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
