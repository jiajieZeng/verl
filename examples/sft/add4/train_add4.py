"""Explicit VERL SFT entry; importing this module never starts training."""
from functools import wraps


def root_owned_lm_head(selector):
    """Fused Qwen loss reads head.weight without calling the head's forward.

    Keep the head under root FSDP hooks so its weight is materialized before
    fused loss accesses it. The other layer/embedding shard targets stay intact.
    """
    @wraps(selector)
    def select(model, classes):
        if getattr(model.config, 'model_type', None) != 'qwen3_5':
            raise ValueError('This SFT entry requires Qwen3.5 dense')
        head = model.lm_head
        return [module for module in selector(model, classes) if module is not head]
    return select


def independent_validation_batch(build):
    """Decouple validation batches without dropping or duplicating examples."""
    @wraps(build)
    def build_loaders(self):
        dp_size = self.engine.get_data_parallel_size()
        train_batch = self.config.data.train_batch_size
        val_batch = self.config.data.validation_batch_size
        if train_batch <= 0 or train_batch % dp_size or len(self.train_dataset) % dp_size:
            raise ValueError('Training batch and dataset must divide data-parallel size; refusing sample loss')
        if val_batch <= 0 or val_batch % dp_size:
            raise ValueError('Validation batch must divide data-parallel size')
        if self.val_dataset is not None and len(self.val_dataset) % dp_size:
            raise ValueError('Validation sampler would drop examples; refusing incomplete validation')
        build(self)
        previous = self.train_dataloader
        self.train_dataloader = type(previous)(
            dataset=self.train_dataset,
            batch_size=train_batch // dp_size,
            sampler=previous.sampler,
            collate_fn=self.collate_fn,
            num_workers=self.config.data.num_workers,
            pin_memory=False,
            drop_last=False,
            pin_memory_device=previous.pin_memory_device,
        )
        if self.val_dataloader is not None:
            previous = self.val_dataloader
            self.val_dataloader = type(previous)(
                dataset=self.val_dataset,
                batch_size=val_batch // dp_size,
                sampler=self.val_sampler,
                collate_fn=self.collate_fn,
                num_workers=self.config.data.num_workers,
                pin_memory=False,
                drop_last=False,
                pin_memory_device=previous.pin_memory_device,
            )
    return build_loaders


def validation_before_training(fit):
    """Log the untouched base model with the same validation reduction as VERL."""
    @wraps(fit)
    def run(self):
        if self.config.trainer.get('validate_before_training', False):
            import torch
            import verl.utils.tensordict_utils as tu
            meta = dict(use_remove_padding=self.config.model.use_remove_padding,
                        use_dynamic_bsz=self.config.data.use_dynamic_bsz,
                        max_token_len_per_gpu=self.config.data.max_token_len_per_gpu,
                        micro_batch_size_per_gpu=self.config.data.micro_batch_size_per_gpu,
                        temperature=1.0, global_batch_size=self.global_batch_size,
                        pad_mode=self.config.data.pad_mode, pad_token_id=self.model_config.tokenizer.pad_token_id)
            losses = []
            log_rank = self.engine.is_mp_src_rank_with_outputs() and self.engine.get_data_parallel_rank() == 0
            if log_rank:
                print(f'Baseline validation: {len(self.val_dataloader)} batches before any optimizer update', flush=True)
            for index, batch in enumerate(self.val_dataloader, 1):
                data = tu.get_tensordict(tensor_dict=batch, non_tensor_dict=meta)
                output = self.training_client.infer_batch(data)
                if self.engine.is_mp_src_rank_with_outputs():
                    losses.append(tu.get(output, 'metrics')['loss'])
                if log_rank and index % 25 == 0:
                    print(f'Baseline validation: {index}/{len(self.val_dataloader)} batches', flush=True)
            if self.engine.is_mp_src_rank_with_outputs():
                loss = torch.mean(torch.tensor(losses, device=self.device_name))
                group = self.engine.get_data_parallel_group()
                if group is not None:
                    torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG, group=group)
                if self.engine.get_data_parallel_rank() == 0:
                    print(f'step:{self.resume_global_step} - val/loss:{loss.item()}', flush=True)
            torch.distributed.barrier()
        return fit(self)
    return run


def main():
    import verl.utils.fsdp_utils as fsdp_utils
    fsdp_utils._select_fsdp2_wrap_targets = root_owned_lm_head(
        fsdp_utils._select_fsdp2_wrap_targets
    )
    import verl.trainer.sft_trainer as sft
    sft.SFTTrainer._build_dataloader = independent_validation_batch(sft.SFTTrainer._build_dataloader)
    sft.SFTTrainer.fit = validation_before_training(sft.SFTTrainer.fit)
    sft.main()


if __name__ == '__main__':
    main()
