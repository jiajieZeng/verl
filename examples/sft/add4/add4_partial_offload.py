"""Opt-in per-layer FSDP2 CPU offload for the Add SFT entry."""
import re

import torch
from torch.distributed.fsdp import CPUOffloadPolicy


def install_partial_offload(layer_indices):
    import verl.utils.fsdp_utils as utils
    import verl.workers.engine.fsdp.transformer_impl as engine

    indices = set(layer_indices)
    original_apply = engine.apply_fsdp2
    original_load = engine.fsdp2_load_full_state_dict

    def apply(model, fsdp_kwargs, config):
        if fsdp_kwargs.get('offload_policy') is not None:
            raise ValueError('Partial offload requires global CPU offload disabled')
        targets = {
            name: module for name, module in model.named_modules()
            if (match := re.search(r'(?:^|\.)layers\.(\d+)$', name))
            and int(match.group(1)) in indices
            and module.__class__.__name__ == 'Qwen3_5MoeDecoderLayer'
        }
        if len(targets) != len(indices):
            raise ValueError(f'Expected {len(indices)} decoder layers; found {list(targets)}')
        target_ids = {id(module) for module in targets.values()}
        original_shard = utils.fully_shard
        wrapped = set()

        def shard(module, **kwargs):
            if id(module) in target_ids:
                kwargs['offload_policy'] = CPUOffloadPolicy(pin_memory=True)
                wrapped.add(id(module))
            return original_shard(module, **kwargs)

        utils.fully_shard = shard
        try:
            result = original_apply(model, fsdp_kwargs, config)
        finally:
            utils.fully_shard = original_shard
        if wrapped != target_ids:
            raise ValueError('Selected CPU-offloaded layers must be individual FSDP units')
        # Bypass nn.Module registration: these are already children of the model.
        object.__setattr__(model, '_add4_cpu_offload_layers', targets)
        print(f'Add SFT partial CPU offload: {list(targets)}', flush=True)
        return result

    def load(model, full_state, device_mesh=None, cpu_offload=None):
        result = original_load(model, full_state, device_mesh, cpu_offload)
        for module in model._add4_cpu_offload_layers.values():
            # Loading broadcasts on CUDA; restore only selected shards to CPU
            # before FSDP lazy initialization. Buffers remain on CUDA.
            module.to('cpu')
            for buffer in module.buffers():
                buffer.data = buffer.data.to(torch.cuda.current_device())
        cpu_count = sum(p.numel() for p in model.parameters() if p.device.type == 'cpu')
        total = sum(p.numel() for p in model.parameters())
        print(f'Add SFT residency: CPU {cpu_count:,}/{total:,} parameters ({cpu_count/total:.2%})', flush=True)
        return result

    engine.apply_fsdp2 = apply
    engine.fsdp2_load_full_state_dict = load
