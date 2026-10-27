"""OPD telemetry: bounded scalar summaries, durable events and W&B delivery."""
from collections import Counter, defaultdict
from contextlib import contextmanager
import json
import math
from pathlib import Path
import shutil
import threading
import time


def facts_valid(text):
    try:
        value = json.loads(text)
        return (isinstance(value, dict) and set(value) == {'facts'}
                and isinstance(value['facts'], list)
                and all(isinstance(x, str) for x in value['facts']))
    except (ValueError, TypeError):
        return False


class OPDMonitor:
    def __init__(self, directory, enabled=True, tokenizer=None):
        self.enabled, self.tokenizer = enabled, tokenizer
        self.directory = Path(directory)
        if enabled:
            self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.step = 0
        self.tracking = None
        self.totals = Counter()
        self.begin(0)

    def begin(self, step):
        previous_duration = time.perf_counter() - self.started if hasattr(self, 'started') and self.step > 0 else None
        self.step = step
        self.started = time.perf_counter()
        self.sums = Counter()
        self.values = defaultdict(list)
        if previous_duration is not None:
            self.sums['time/previous_step_with_validation_checkpoint_s'] = previous_duration

    def count(self, key, amount=1):
        if self.enabled:
            with self.lock:
                self.sums[key] += amount

    @contextmanager
    def phase(self, name):
        start = time.perf_counter()
        try:
            yield
        except Exception:
            self.count(f'errors/{name}')
            raise
        finally:
            self.count(f'time/{name}_s', time.perf_counter() - start)

    def http(self, role, path, elapsed, size, ok):
        if not self.enabled:
            return
        operation = 'score' if role == 'teacher' and path == '/generate' else (
            'rollout' if path == '/generate' else 'reload' if path == '/update_weights_from_disk' else 'probe')
        prefix = f'http/{role}_{operation}'
        with self.lock:
            self.sums[f'{prefix}_requests'] += 1
            self.sums[f'{prefix}_errors'] += int(not ok)
            self.sums[f'{prefix}_bytes'] += size
            self.sums[f'{prefix}_service_s'] += elapsed
            # Bound memory even for full-vocabulary scoring with many chunks.
            samples = self.values[f'{prefix}_latency_s']
            if len(samples) < 10000:
                samples.append(elapsed)

    def sample(self, ids, finish, eos):
        if not self.enabled:
            return
        with self.lock:
            self.sums['generation/attempts'] += 1
            self.sums['generation/eos_count'] += int(finish == 'stop' and bool(ids) and ids[-1] == eos)
            self.sums['generation/truncated_count'] += int(finish == 'length')
            self.values['generation/response_tokens'].append(len(ids))
            if self.tokenizer is not None:
                text = self.tokenizer.decode(ids, skip_special_tokens=True)
                self.sums['generation/format_checked'] += 1
                self.sums['generation/format_valid_count'] += int(facts_valid(text))

    def records(self, records):
        self.count('generation/samples', len(records))
        self.count('generation/prompt_tokens', sum(len(r['prompt_ids']) for r in records))
        self.count('generation/response_tokens_total', sum(len(r['response_ids']) for r in records))
        self.count('cache/written_bytes', sum(Path(r['teacher_scores']['path']).stat().st_size
                                            for r in records if 'teacher_scores' in r))

    def snapshot(self):
        with self.lock:
            result = dict(self.sums)
            for key, values in self.values.items():
                if values:
                    ordered = sorted(values)
                    result[key + '_mean'] = sum(values) / len(values)
                    result[key + '_p95'] = ordered[math.ceil(.95 * len(values)) - 1]
                    result[key + '_max'] = ordered[-1]
            attempts = result.get('generation/attempts', 0)
            if attempts:
                result['generation/eos_rate'] = result.get('generation/eos_count', 0) / attempts
                result['generation/truncated_rate'] = result.get('generation/truncated_count', 0) / attempts
            checked = result.get('generation/format_checked', 0)
            if checked:
                result['generation/format_valid_rate'] = result.get('generation/format_valid_count', 0) / checked
            elapsed = time.perf_counter() - self.started
            result['time/step_to_metrics_s'] = elapsed
            result['perf/response_tokens_per_s'] = result.get('generation/response_tokens_total', 0) / max(elapsed, 1e-9)
            result['opd/step'] = self.step
            result['time/weight_export_s'] = max(0., result.get('time/weight_sync_s', 0) - result.get('time/student_reload_s', 0))
            result.update(self.totals)
            return result

    def disk(self, paths):
        for name, path in paths.items():
            path = Path(path)
            while not path.exists():
                path = path.parent
            usage = shutil.disk_usage(path)
            self.count(f'disk/{name}_free_gb', usage.free / 1024**3)

    def resources(self, path, required=False):
        try:
            record = json.loads(Path(path).read_text())
            age = time.time() - record['sample_time']
            fresh = -5 <= age <= 120
            probes_ok = all(record['metrics'].get(f'resources/{role}_probe_ok') == 1
                            for role in ('training_host', 'teacher_host'))
            if required and (not fresh or not probes_ok):
                raise ValueError('GPU monitor snapshot is stale or a host probe failed')
            self.count('resources/snapshot_age_s', age)
            self.count('resources/snapshot_fresh', int(fresh))
            if fresh:
                for key, value in record['metrics'].items():
                    self.count(key, value)
        except (OSError, ValueError, KeyError):
            self.count('errors/resource_snapshot')
            if required:
                raise ValueError('Start add4_opd_resources.py on the training host before monitored OPD') from None

    def emit(self, metrics, event='step'):
        if not self.enabled:
            return
        # Nonfinite values remain visible as errors without invalid JSON.
        safe = {k: v if not isinstance(v, float) or math.isfinite(v) else None for k, v in metrics.items()}
        with (self.directory / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(event=event, step=self.step, time=time.time(), metrics=safe), allow_nan=False) + '\n')


class MonitoredTracking:
    def __init__(self, tracking, monitor):
        self.tracking, self.monitor = tracking, monitor
        self.event_index = 0
        monitor.tracking = self
        wandb = tracking.logger.get('wandb')
        if wandb:
            wandb.define_metric('opd/step')
            wandb.define_metric('*', step_metric='opd/step')

    def log(self, data, step, **kwargs):
        self.monitor.step = step
        metrics = dict(data)
        if 'train/loss' in data:
            metrics.update(self.monitor.snapshot())
            norm = float(data.get('train/grad_norm', 0))
            skipped = int(not math.isfinite(norm))
            self.monitor.totals['errors/nonfinite_grad_total'] += skipped
            self.monitor.totals['train/skipped_updates_total'] += skipped
            metrics.update(self.monitor.totals)
            metrics['train/update_skipped'] = skipped
        metrics['opd/step'] = step
        self.monitor.emit(metrics)
        # Separate event sequence permits baseline, train, validation, checkpoint
        # and failure at the same optimizer step without W&B dropping late rows.
        self.tracking.log(metrics, step=self.event_index, **kwargs)
        self.event_index += 1

    def failure(self, error):
        self.monitor.totals['errors/fatal_total'] += 1
        metrics = self.monitor.snapshot()
        metrics['errors/fatal'] = 1
        metrics['errors/type'] = type(error).__name__
        self.log(metrics, self.monitor.step)

    def finish(self, exit_code=0):
        self.tracking.finish(exit_code=exit_code)


def phase(adapter, name):
    from contextlib import nullcontext
    monitor = getattr(getattr(adapter, 'trainer', None), '_opd_monitor', None)
    return monitor.phase(name) if monitor else nullcontext()


def observed(adapter, name, fn, *args, **kwargs):
    with phase(adapter, name):
        return fn(*args, **kwargs)
