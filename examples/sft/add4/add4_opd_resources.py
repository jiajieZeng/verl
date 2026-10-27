"""Read-only host GPU/disk snapshots; no model or process control."""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import time


def gpu_metrics(text, prefix):
    metrics = {}
    for row in csv.reader(text.splitlines()):
        if len(row) != 5:
            raise ValueError('Unexpected nvidia-smi output')
        index = int(row[0])
        for field, value in zip(('utilization_pct', 'memory_used_mib', 'memory_total_mib', 'temperature_c'), row[1:]):
            try:
                metrics[f'{prefix}/gpu{index}/{field}'] = float(value.strip())
            except ValueError:
                # N/A is an absent metric, never a healthy-looking zero.
                continue
    return metrics


def collect(teacher='zengjiajie@47.103.58.204', identity=None):
    query = ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu',
             '--format=csv,noheader,nounits']
    output = {'sample_time': time.time(), 'scope': 'whole_hosts_not_exclusive_training_gpus', 'metrics': {}}
    jobs = [('training_host', query)]
    if teacher:
        if not re.fullmatch(r'[A-Za-z0-9_.@-]+', teacher) or teacher.startswith('-'):
            raise ValueError('Invalid teacher SSH destination')
        identity = identity or str(Path.home()/'.ssh/dockerserver')
        jobs.append(('teacher_host', ['ssh', '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes',
            '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=5', '-i', identity, teacher, *query]))
    for role, command in jobs:
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=True)
            output['metrics'].update(gpu_metrics(result.stdout, role))
            output['metrics'][f'resources/{role}_probe_ok'] = 1
        except (OSError, subprocess.SubprocessError, ValueError):
            output['metrics'][f'resources/{role}_probe_ok'] = 0
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--teacher', default='zengjiajie@47.103.58.204')
    parser.add_argument('--identity', default=os.environ.get('OPD_TEACHER_SSH_IDENTITY'))
    parser.add_argument('--interval', type=float, default=30)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error('interval must be positive')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    while True:
        result = collect(args.teacher, args.identity)
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, indent=2) + '\n')
        temporary.replace(args.output)
        if args.once:
            print(json.dumps({'output': str(args.output), 'probes': {
                k: v for k, v in result['metrics'].items() if k.endswith('_ok')}}))
            return
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
