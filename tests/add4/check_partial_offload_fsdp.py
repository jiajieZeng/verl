"""Two-GPU check: mixed CPU/CUDA FSDP loading, accumulation, clipping and Adam."""
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'examples/sft/add4'))
from add4_partial_offload import install_partial_offload
from add4_streaming_adamw import StreamingAdamW
from verl.utils.fsdp_utils import fsdp2_clip_grad_norm_
import verl.workers.engine.fsdp.transformer_impl as engine


class Qwen3_5MoeDecoderLayer(torch.nn.Sequential):
    def __init__(self):
        super().__init__(torch.nn.Linear(64, 64), torch.nn.GELU())


class Model(torch.nn.Module):
    _no_split_modules = ['Qwen3_5MoeDecoderLayer']

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=False)
        self.layers = torch.nn.ModuleList([Qwen3_5MoeDecoderLayer() for _ in range(2)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def main():
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('cpu:gloo,cuda:nccl')
    device = torch.device('cuda', rank)
    torch.manual_seed(536)
    a, b = Model().to(device), Model().to(device)
    initial = {key: value.clone() for key, value in a.state_dict().items()}
    b.load_state_dict(initial)
    engine.apply_fsdp2(b, {}, {})
    install_partial_offload([1])
    engine.apply_fsdp2(a, {}, {})
    engine.fsdp2_load_full_state_dict(a, initial)
    assert all(p.device.type == 'cpu' for p in a.layers[1].parameters())
    assert all(p.device.type == 'cuda' for p in a.layers[0].parameters())
    actual = StreamingAdamW(a.parameters(), lr=.01, weight_decay=.1)
    expected = torch.optim.AdamW(b.parameters(), lr=.01, weight_decay=.1, foreach=False, fused=False)
    for step in range(3):
        xs = [torch.randn(12, 64, device=device) for _ in range(2)]
        norms = []
        for model, optimizer in ((a, actual), (b, expected)):
            optimizer.zero_grad(set_to_none=True)
            for x in xs:
                model(x).square().mean().backward()
            norms.append(fsdp2_clip_grad_norm_(model.parameters(), max_norm=.01).full_tensor())
            optimizer.step()
        torch.testing.assert_close(norms[0], norms[1], rtol=1e-6, atol=1e-8)
        for pa, pb in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(pa.to_local().cpu(), pb.to_local().cpu(), rtol=1e-5, atol=1e-7)
        assert all(p.device.type == 'cpu' for p in a.layers[1].parameters())
        assert all(p.device.type == 'cuda' for p in a.layers[0].parameters())
    dist.barrier()
    if rank == 0:
        print('PASS: mixed CPU/CUDA FSDP2, loading, 3 updates with 2 accumulated microbatches and clipping.', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
