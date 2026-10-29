"""Add-7 SFT entry: reuse Add-4 head fix and VERL packed loss/checkpoints."""
import json
import os
from pathlib import Path
import time
from functools import wraps


def build_loaders(self):
    from torchdata.stateful_dataloader import StatefulDataLoader
    from verl.utils.dataset.dataset_utils import SFTTensorCollator
    from sampler import BalancedSampler
    config = self.config
    if config.trainer.n_gpus_per_node != 8 or config.engine.ulysses_sequence_parallel_size != 4 or config.data.train_batch_size != 64:
        raise ValueError('This recipe requires 8 GPUs, SP4, global batch64')
    if not config.data.use_dynamic_bsz or not config.model.use_remove_padding:
        raise ValueError('Dynamic packing and removal of padding must be enabled')
    dp, rank = self.engine.get_data_parallel_size(), self.engine.get_data_parallel_rank()
    self.global_batch_size = config.data.train_batch_size
    self.train_batch_size_per_dp = self.global_batch_size // dp
    self.collate_fn = SFTTensorCollator(config.data.pad_mode)
    if self.val_dataset is not None:
        overlap = {r['group'] for r in self.train_dataset.rows} & {r['group'] for r in self.val_dataset.rows}
        if overlap:
            raise ValueError(f'Train/val source overlap: {len(overlap)} groups')
    def make(dataset, batch_size, shuffle):
        sampler = BalancedSampler(dataset.lengths, batch_size, dp, rank,
                                  seed=config.trainer.seed, shuffle=shuffle,
                                  longest_first=bool(os.environ.get('ADD_SMOKE_STEPS')) and shuffle)
        loader = StatefulDataLoader(dataset, batch_size=batch_size // dp, sampler=sampler,
            collate_fn=self.collate_fn, num_workers=config.data.num_workers,
            pin_memory=False, drop_last=False)
        return sampler, loader
    self.train_sampler, self.train_dataloader = make(self.train_dataset, self.global_batch_size, True)
    if self.val_dataset is not None:
        self.val_sampler, self.val_dataloader = make(self.val_dataset, config.data.validation_batch_size, False)
    else:
        self.val_sampler, self.val_dataloader = None, None
    if self.rank == 0:
        print('ADD7_COVERAGE ' + json.dumps(dict(train=len(self.train_dataset),
            val=len(self.val_dataset) if self.val_dataset is not None else 0,
            dp=dp, global_batch=self.global_batch_size, steps_per_epoch=len(self.train_dataloader),
            dropped=0, duplicated=0, sampler='within-global-batch workload balance')), flush=True)


def main():
    import torch
    import torch.distributed as dist
    import verl.utils.fsdp_utils as fsdp_utils
    from examples.sft.add4.train_add4 import root_owned_lm_head
    fsdp_utils._select_fsdp2_wrap_targets = root_owned_lm_head(fsdp_utils._select_fsdp2_wrap_targets)
    from verl.utils import tensordict_utils as tu
    import verl.trainer.sft_trainer as sft
    sft.SFTTrainer._build_dataloader = build_loaders
    if os.environ.get('ADD_SMOKE_STEPS'):
        fit = sft.SFTTrainer.fit
        @wraps(fit)
        def smoke(self):
            self.total_training_steps = int(os.environ['ADD_SMOKE_STEPS'])
            self.val_dataloader = None
            self.save_freq = self.total_training_steps
            return fit(self)
        sft.SFTTrainer.fit = smoke
    # Measure the slowest rank, including optimizer update, and sum tokens once per DP group.
    from verl.workers.engine_workers import TrainingWorker
    original_train = TrainingWorker.train_batch
    @wraps(original_train)
    def measured(self, data):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        output = original_train(self, data)
        torch.cuda.synchronize()
        values = torch.tensor([time.monotonic()-started, torch.cuda.max_memory_allocated()/2**30,
                               torch.cuda.max_memory_reserved()/2**30], device='cuda')
        dist.all_reduce(values, op=dist.ReduceOp.MAX)
        tokens = data['input_ids'].offsets().diff().sum().to('cuda')
        dist.all_reduce(tokens, group=self.engine.get_data_parallel_group())
        if self.engine.is_mp_src_rank_with_outputs():
            metrics = tu.get(output, 'metrics')
            metrics.update({'train/step_seconds': values[0].item(),
                            'train/tokens_per_second': tokens.item()/values[0].item(),
                            'train/max_allocated_gib': values[1].item(),
                            'train/max_reserved_gib': values[2].item()})
        return output
    TrainingWorker.train_batch = measured
    # Small per-step audit: sample boundaries remain separate inside a packed microbatch.
    import verl.workers.engine.fsdp.transformer_impl as impl
    prepare = impl.prepare_micro_batches
    def audited(*args, **kwargs):
        batches, indices = prepare(*args, **kwargs)
        rank = dist.get_rank()
        if rank % int(os.environ.get('ADD_SP_SIZE', '4')) == 0:
            print('ADD7_PACKING ' + json.dumps(dict(rank=rank,
                microbatches=len(batches), examples=[len(b['input_ids']) for b in batches],
                packed_tokens=[int(b['input_ids'].offsets().diff().sum()) for b in batches])), flush=True)
        return batches, indices
    impl.prepare_micro_batches = audited
    original_log = sft.Tracking.log
    @wraps(original_log)
    def log(self, data, step, backend=None):
        with (Path(os.environ['ADD_RUN_DIR'])/'metrics.jsonl').open('a') as out:
            out.write(json.dumps(dict(time=time.time(), step=step, metrics=data), default=float)+'\n')
        return original_log(self, data, step, backend)
    sft.Tracking.log = log
    sft.main()


if __name__ == '__main__':
    main()
