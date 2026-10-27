"""Upload scalar SFT console metrics without restarting the training process."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import time

PROXIES = ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy')


def metric_events(text):
    # Ignore an unfinished last line while the trainer is writing.
    for line in text.rsplit('\n', 1)[0].splitlines():
        match = re.search(r'\bstep:\s*(\d+)\s+-\s+', line)
        if not match:
            continue
        step = int(match[1])
        metrics = {}
        for key, value in re.findall(r'((?:train|val|perf)/[^\s:]+):\s*([^\s]+)', line[match.end():]):
            try:
                number = float(value)
            except ValueError:
                continue
            if math.isfinite(number):
                metrics[key] = number
        if not metrics:
            continue
        digest = hashlib.sha256(json.dumps([step, metrics], sort_keys=True).encode()).hexdigest()
        yield step, metrics, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--credentials', type=Path, required=True)
    parser.add_argument('--project', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--id', required=True)
    args = parser.parse_args()
    for key in PROXIES:
        os.environ.pop(key, None)
    for raw in args.credentials.read_text().splitlines():
        parts = shlex.split(raw, comments=True)
        if parts and parts[0] == 'export':
            parts = parts[1:]
        if len(parts) != 1 or '=' not in parts[0]:
            continue
        key, value = parts[0].split('=', 1)
        if key in ('WANDB_API_KEY', 'WANDB_BASE_URL', 'WANDB_ENTITY'):
            os.environ[key] = value
    output = args.run_dir / 'wandb-sync'
    output.mkdir(exist_ok=True)
    os.environ.update(WANDB_MODE='online', WANDB_DISABLE_GIT='true', WANDB_CONSOLE='off',
                      WANDB_CACHE_DIR=str(output / 'cache'), WANDB_CONFIG_DIR=str(output / 'config'),
                      XDG_CACHE_HOME=str(output / 'cache'))
    import wandb
    config = json.loads(args.config.read_text())
    run = wandb.init(entity=os.environ['WANDB_ENTITY'], project=args.project, id=args.id,
                     name=args.name, resume='allow', mode='online', dir=str(output),
                     config={'training': config, 'metric_source': 'existing train.log scalar bridge',
                             'raw_data_uploaded': False}, save_code=False)
    run.define_metric('trainer/global_step')
    for pattern in ('train/*', 'val/*', 'perf/*'):
        run.define_metric(pattern, step_metric='trainer/global_step')
    seen = set()
    if run.resumed:
        remote = wandb.Api().run(f'{run.entity}/{run.project}/{run.id}')
        seen.update(row['sync/event_id'] for row in remote.scan_history(keys=['sync/event_id']))
    last_step = int(run.summary.get('sync/last_training_step', 0))
    print(json.dumps({'url': run.url, 'id': run.id, 'resumed': run.resumed}), flush=True)
    while True:
        logfile = args.run_dir / 'train.log'
        if logfile.exists():
            for step, metrics, event_id in metric_events(logfile.read_text(errors='replace')):
                if event_id in seen:
                    continue
                run.log({**metrics, 'trainer/global_step': step, 'sync/event_id': event_id})
                seen.add(event_id)
                last_step = max(last_step, step)
        statefile = args.run_dir / 'status.json'
        state = json.loads(statefile.read_text()) if statefile.exists() else {}
        run.summary.update({'sync/last_training_step': last_step, 'sync/events_queued': len(seen),
                            'sync/training_phase': state.get('phase', 'unknown')})
        status = {'url': run.url, 'run_id': run.id, 'entity': run.entity, 'project': run.project,
                  'last_training_step': last_step, 'events_queued': len(seen),
                  'training_phase': state.get('phase'), 'checked_at_unix': time.time(),
                  'proxies_absent': all(key not in os.environ for key in PROXIES)}
        temporary = output / 'status.tmp'
        temporary.write_text(json.dumps(status, indent=2) + '\n')
        temporary.replace(output / 'status.json')
        if state.get('phase') in ('completed', 'failed'):
            run.finish(exit_code=0 if state['phase'] == 'completed' else 1)
            return
        time.sleep(15)


if __name__ == '__main__':
    main()
