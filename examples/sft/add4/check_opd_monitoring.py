"""Small CPU + W&B round trip; never starts a model service or GPU training."""
import argparse
import json
import os
from pathlib import Path
import time
import uuid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
        os.environ.pop(key, None)
    run_id = 'opd-monitor-smoke-' + uuid.uuid4().hex[:12]
    os.environ.update(WANDB_RUN_ID=run_id, WANDB_RESUME='never', WANDB_DIR=str(args.output),
                      WANDB_MODE='offline' if args.offline else 'online')
    import torch
    import wandb
    from verl.utils.tracking import Tracking
    from add4_opd_monitor import OPDMonitor, MonitoredTracking
    from add4_opd_topk import chunked_topk_loss
    monitor = OPDMonitor(args.output)
    tracker = MonitoredTracking(Tracking('codemem-add4-opd-monitoring-smoke', run_id,
        default_backend=['wandb'], config={'synthetic_cpu_smoke_only': True, 'trainer': {}}), monitor)
    run_path, url = wandb.run.path, wandb.run.url
    h = torch.ones(1, 1, requires_grad=True)
    w = torch.tensor([[.5], [-.5]], requires_grad=True)
    optimizer = torch.optim.SGD([h, w], lr=.01)
    for step in (1, 2):
        monitor.begin(step)
        with monitor.phase('train'):
            optimizer.zero_grad()
            result = chunked_topk_loss(h, w, torch.tensor([[0]]), torch.tensor([[0.]]), torch.ones(1), 2)
            result[0].sum().backward()
            norm = torch.nn.utils.clip_grad_norm_([h, w], 1.)
            optimizer.step()
        tracker.log({'train/loss': result[0].item(), 'train/grad_norm': norm.item(),
                     'opd/reverse_kl': result[1].item(), 'opd/teacher_tail_floor_rate': 1.}, step)
    # Same-step validation must survive the W&B event index mapping.
    tracker.log({'smoke/same_step_event': 1}, 2)
    tracker.finish()
    report = dict(run_id=run_id, run_path=run_path, url=url, mode=os.environ['WANDB_MODE'],
                  cpu_steps=2, uploaded_and_read_back=False)
    if not args.offline:
        keys = ['opd/step', 'opd/reverse_kl', 'opd/teacher_tail_floor_rate']
        deadline = time.monotonic() + 45
        while True:
            remote = wandb.Api(timeout=20).run(run_path)
            rows = list(remote.scan_history(keys=keys))
            if len(rows) == 2 and remote.summary.get('smoke/same_step_event') == 1:
                report['uploaded_and_read_back'] = True
                report['history'] = rows
                break
            if time.monotonic() >= deadline:
                (args.output / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')
                raise RuntimeError('W&B upload/readback did not confirm both steps and same-step event')
            time.sleep(2)
    (args.output / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
