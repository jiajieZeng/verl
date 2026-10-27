"""Two-GPU FSDP2 equivalence check; invoke with torchrun --nproc_per_node=2."""
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'examples/sft/add4'))
from add4_streaming_adamw import StreamingAdamW


def main():
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl')
    device = torch.device('cuda', local_rank)
    torch.manual_seed(536)
    a = torch.nn.Sequential(torch.nn.Linear(64, 32), torch.nn.GELU(), torch.nn.Linear(32, 8)).to(device)
    b = torch.nn.Sequential(torch.nn.Linear(64, 32), torch.nn.GELU(), torch.nn.Linear(32, 8)).to(device)
    b.load_state_dict(a.state_dict())
    for model in (a, b):
        fully_shard(model[0])
        fully_shard(model[2])
        fully_shard(model)
    actual = StreamingAdamW(a.parameters(), lr=.01, weight_decay=.1)
    expected = torch.optim.AdamW(b.parameters(), lr=.01, weight_decay=.1, foreach=False, fused=False)
    for step in range(3):
        x = torch.randn(12, 64, device=device)
        for model, optimizer in ((a, actual), (b, expected)):
            optimizer.zero_grad(set_to_none=True)
            model(x).square().mean().backward()
            optimizer.step()
        for pa, pb in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(pa.to_local(), pb.to_local(), rtol=0, atol=0)
            for key in ('exp_avg', 'exp_avg_sq'):
                sa, sb = actual.state[pa][key], expected.state[pb][key]
                assert sa.to_local().device.type == 'cpu'
                torch.testing.assert_close(sa.to_local(), sb.to_local().cpu(), rtol=0, atol=0)
        assert all(p.to_local().device.type == 'cuda' for p in a.parameters())
    dist.barrier()
    if local_rank == 0:
        print('PASS: FSDP2 2 GPUs, 3 updates exactly match AdamW; parameters on CUDA, moments on CPU.', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
