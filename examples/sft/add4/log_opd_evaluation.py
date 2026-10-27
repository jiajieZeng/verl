"""Attach completed benchmark counts to an OPD checkpoint in a separate W&B run."""
import argparse
import json
import os
from pathlib import Path
import re
import uuid


def evaluation_metrics(request, result):
    if result['checkpoint_step'] != request['checkpoint_step']:
        raise ValueError('Evaluation checkpoint step does not match the saved checkpoint')
    if result['checkpoint'] != request['checkpoint']:
        raise ValueError('Evaluation checkpoint path does not match the saved checkpoint')
    if not result['datasets']:
        raise ValueError('Empty evaluation')
    metrics = {'eval/checkpoint_step': request['checkpoint_step']}
    totals = dict(successes=0, attempts=0, solved_tasks=0, tasks=0)
    for name, counts in result['datasets'].items():
        if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
            raise ValueError('Use a stable dataset metric name')
        for key in totals:
            if type(counts[key]) is not int or counts[key] < 0:
                raise ValueError('Evaluation counts must be nonnegative integers')
        if counts['attempts'] != 4 * counts['tasks'] or not counts['tasks']:
            raise ValueError('avg@4 requires exactly four completed attempts per task')
        if not 0 <= counts['solved_tasks'] <= counts['tasks'] or not (
            counts['solved_tasks'] <= counts['successes'] <= 4 * counts['solved_tasks']):
            raise ValueError('Inconsistent successful attempt/task counts')
        metrics[f'eval/{name}/avg_at_4'] = counts['successes'] / counts['attempts']
        metrics[f'eval/{name}/pass_at_4'] = counts['solved_tasks'] / counts['tasks']
        for key in totals:
            totals[key] += counts[key]
    metrics['eval/all/avg_at_4'] = totals['successes'] / totals['attempts']
    metrics['eval/all/pass_at_4'] = totals['solved_tasks'] / totals['tasks']
    metrics['eval/all/tasks'] = totals['tasks']
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    request, result = json.loads(args.request.read_text()), json.loads(args.results.read_text())
    metrics = evaluation_metrics(request, result)
    if args.dry_run:
        print(json.dumps(metrics))
        return
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        os.environ.pop(key, None)
    import wandb
    run = wandb.init(project=request['project'], entity=os.environ.get('WANDB_ENTITY'),
        id='opd-eval-' + uuid.uuid4().hex[:12], group=request['training_run_id'],
        job_type='checkpoint-evaluation', mode='online',
        config={'training_run_id': request['training_run_id'], 'checkpoint': request['checkpoint'],
                'result_source': str(args.results.resolve())})
    run.define_metric('eval/checkpoint_step')
    run.define_metric('eval/*', step_metric='eval/checkpoint_step')
    run.log(metrics)
    url = run.url
    run.finish()
    args.results.with_suffix('.wandb-receipt.json').write_text(json.dumps(dict(
        url=url, training_run_id=request['training_run_id'], metrics=metrics), indent=2) + '\n')
    print(json.dumps({'url': url}))


if __name__ == '__main__':
    main()
