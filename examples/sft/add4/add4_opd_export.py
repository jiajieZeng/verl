"""Bounded HF shards for student reload, including many small per-expert MoE tensors."""
from pathlib import Path


class StudentShardWriter:
    def __init__(self, folder, max_bytes):
        if max_bytes <= 0:
            raise ValueError('Student export shard budget must be positive')
        self.folder, self.max_bytes = Path(folder), max_bytes
        self.pending, self.pending_bytes = {}, 0
        self.mapping, self.total_bytes, self.shard_count = {}, 0, 0

    def add(self, name, tensor):
        import torch
        if name in self.mapping or name in self.pending:
            raise ValueError('Duplicate parameter in student weight export')
        dtype = torch.bfloat16 if tensor.is_floating_point() else tensor.dtype
        # Own storage: a view into one packed expert tensor must not retain that whole tensor.
        tensor = tensor.detach().to(device='cpu', dtype=dtype, copy=True).contiguous()
        size = tensor.numel() * tensor.element_size()
        if self.pending and self.pending_bytes + size > self.max_bytes:
            self.flush()
        self.pending[name] = tensor
        self.pending_bytes += size
        if self.pending_bytes >= self.max_bytes:
            self.flush()

    def flush(self):
        from safetensors.torch import save_file
        if not self.pending:
            return
        filename = f'weights-{self.shard_count:05d}.safetensors'
        temporary = self.folder / (filename + '.tmp')
        save_file(self.pending, str(temporary))
        temporary.replace(self.folder / filename)
        self.mapping.update({name: filename for name in self.pending})
        self.total_bytes += self.pending_bytes
        self.shard_count += 1
        self.pending, self.pending_bytes = {}, 0
