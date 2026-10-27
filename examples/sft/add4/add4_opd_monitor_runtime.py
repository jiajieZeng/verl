"""Install telemetry only in the OPD trainer, preserving the VERL SFT loop."""
import json
import os
from pathlib import Path
import time
import uuid

from add4_opd_monitor import OPDMonitor, MonitoredTracking


def build_loaders(trainer, build):
    # Replicate validation, not training, when DP cannot partition it exactly.
    # Every rank executes the same 398 forwards; DP means count each row once.
    # This costs extra compute but avoids dropped/reweighted examples and uneven
    # FSDP collective counts. One example per rank gives sequence-mean CE.
    validation = trainer.val_dataset
    trainer.val_dataset = None
    try:
        build()
    finally:
        trainer.val_dataset = validation
    if validation is not None:
        from torch.utils.data import SequentialSampler
        from torchdata.stateful_dataloader import StatefulDataLoader
        trainer.val_sampler = SequentialSampler(validation)
        trainer.val_dataloader = StatefulDataLoader(validation, batch_size=1,
            sampler=trainer.val_sampler, collate_fn=trainer.collate_fn,
            num_workers=trainer.config.data.num_workers, drop_last=False)


def baseline(trainer, tracking):
    import torch
    import torch.distributed as dist
    import verl.utils.tensordict_utils as tu
    meta = dict(use_remove_padding=trainer.config.model.use_remove_padding,
        use_dynamic_bsz=trainer.config.data.use_dynamic_bsz,
        max_token_len_per_gpu=trainer.config.data.max_token_len_per_gpu,
        micro_batch_size_per_gpu=trainer.config.data.micro_batch_size_per_gpu,
        temperature=1.0, global_batch_size=trainer.global_batch_size,
        pad_mode=trainer.config.data.pad_mode, pad_token_id=trainer.model_config.tokenizer.pad_token_id)
    losses = []
    with trainer._opd_monitor.phase('baseline_validation'):
        for batch in trainer.val_dataloader:
            result = trainer.training_client.infer_batch(tu.get_tensordict(tensor_dict=batch, non_tensor_dict=meta))
            if trainer.engine.is_mp_src_rank_with_outputs():
                losses.append(tu.get(result, 'metrics')['loss'])
        if trainer.engine.is_mp_src_rank_with_outputs():
            loss = torch.tensor(losses, device=trainer.device_name).mean()
            group = trainer.engine.get_data_parallel_group()
            if group is not None:
                dist.all_reduce(loss, op=dist.ReduceOp.AVG, group=group)
            if tracking:
                tracking.log({'val/loss': loss.item(), 'val/unique_samples': len(trainer.val_dataset),
                              'val/baseline': 1}, trainer.resume_global_step)
    dist.barrier()


def run_monitored(trainer, sft, fit):
    import torch
    import torch.distributed as dist
    from omegaconf import OmegaConf
    import verl.utils.tensordict_utils as tu
    enabled = trainer.engine.is_mp_src_rank_with_outputs() and trainer.engine.get_data_parallel_rank() == 0
    attempt = [uuid.uuid4().hex[:12] if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(attempt, src=0)
    run_id = f'opd-{trainer.config.opd.run_id}-{attempt[0]}'
    directory = Path(trainer.config.trainer.default_local_dir).parent / 'monitoring' / run_id
    monitor = trainer._opd_monitor = OPDMonitor(directory, enabled, trainer.model_config.tokenizer)
    monitor.step = trainer.resume_global_step
    original_tracking, original_train = sft.Tracking, trainer.training_client.train_batch
    original_save = trainer.ckpt_handler.save_checkpoint
    tracking = None
    exit_code = 0

    def train_batch(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        with monitor.phase('train'):
            result = original_train(*args, **kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        if torch.cuda.is_available():
            peaks = torch.tensor([torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()],
                                device=trainer.device_name, dtype=torch.float64)
            dist.all_reduce(peaks, op=dist.ReduceOp.MAX)
            metrics = tu.get(result, 'metrics')
            metrics.update({'perf/all_rank_peak_allocated_gb': peaks[0].item() / 1024**3,
                            'perf/all_rank_peak_reserved_gb': peaks[1].item() / 1024**3})
            tu.assign_non_tensor(result, metrics=metrics)
        metrics = tu.get(result, 'metrics')
        norm = float(metrics.get('grad_norm', 0))
        metrics['train/grad_clip_triggered'] = int(norm > trainer.config.optim.clip_grad)
        tu.assign_non_tensor(result, metrics=metrics)
        return result

    def save_checkpoint(*args, **kwargs):
        start = time.perf_counter()
        result = original_save(*args, **kwargs)
        if enabled:
            step = kwargs.get('step', args[0] if args else monitor.step)
            # A durable handoff for separately authorized downstream evaluation.
            queue = directory / 'eval-queue'
            queue.mkdir(exist_ok=True)
            checkpoint = str(Path(trainer.config.trainer.default_local_dir) / f'global_step_{step}')
            (queue / f'step-{step:08d}.json').write_text(json.dumps(dict(
                training_run_id=run_id, project=trainer.config.trainer.project_name,
                checkpoint_step=step, checkpoint=checkpoint,
                status='pending_external_evaluation'), indent=2) + '\n')
            tracking.log({'time/checkpoint_s': time.perf_counter() - start,
                          'time/step_through_checkpoint_s': time.perf_counter() - monitor.started,
                          'eval/pending_checkpoint_step': step}, step)
        return result

    try:
        startup = [None]
        if enabled:
            try:
                os.environ['WANDB_RUN_ID'] = run_id
                os.environ['WANDB_RESUME'] = 'never'
                os.environ.setdefault('WANDB_DIR', str(directory))
                if 'wandb' in trainer.config.trainer.logger:
                    os.environ.setdefault('WANDB_MODE', 'online')
                    if os.environ['WANDB_MODE'] != 'online':
                        raise ValueError('Monitored OPD requires online W&B; verify upload before training')
                tracking = MonitoredTracking(original_tracking(
                    project_name=trainer.config.trainer.project_name,
                    experiment_name=trainer.config.trainer.experiment_name,
                    default_backend=trainer.config.trainer.logger,
                    config=OmegaConf.to_container(trainer.config, resolve=True)), monitor)
                (directory / 'manifest.json').write_text(json.dumps(dict(
                    run_id=run_id, resume_global_step=trainer.resume_global_step,
                    validation_samples=len(trainer.val_dataset) if trainer.val_dataset is not None else 0,
                    validation_reduction='replicated_sequence_mean',
                    http_proxy_variables_absent=all(k not in os.environ for k in (
                        'HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'))), indent=2) + '\n')
                sft.Tracking = lambda **kwargs: tracking
                if trainer.config.trainer.get('require_resource_snapshot', False):
                    monitor.resources(Path(trainer.config.trainer.default_local_dir).parent/'resources.json', required=True)
            except Exception as error:
                startup[0] = type(error).__name__
        dist.broadcast_object_list(startup, src=0)
        if startup[0] is not None:
            raise RuntimeError(f"OPD monitoring startup failed on rank 0 ({startup[0]})")
        trainer.training_client.train_batch = train_batch
        trainer.ckpt_handler.save_checkpoint = save_checkpoint
        if trainer.config.trainer.get('validate_before_training', False):
            if trainer.val_dataloader is None:
                raise ValueError('Baseline validation requires a fixed validation dataset')
            baseline(trainer, tracking)
        return fit()
    except BaseException as error:
        exit_code = 1
        if tracking:
            try:
                tracking.failure(error)
            except Exception:
                monitor.emit({'errors/fatal': 1, 'errors/type': type(error).__name__}, 'failure')
        raise
    finally:
        sft.Tracking = original_tracking
        trainer.training_client.train_batch = original_train
        trainer.ckpt_handler.save_checkpoint = original_save
        if tracking:
            tracking.finish(exit_code=exit_code)


def prepare_monitored(trainer, prepare):
    monitor = getattr(trainer, '_opd_monitor', None)
    if monitor is None:
        return prepare()
    monitor.begin(getattr(getattr(trainer, '_opd_adapter', None), 'step', trainer.resume_global_step) + 1)
    paths = {name: trainer.config.opd[name] for name in ('sync_dir', 'audit_dir', 'full_score_dir')
             if name in trainer.config.opd}
    monitor.disk(paths)
    if monitor.enabled:
        monitor.resources(Path(trainer.config.trainer.default_local_dir).parent/'resources.json')
    with monitor.phase('prepare'):
        return prepare()
