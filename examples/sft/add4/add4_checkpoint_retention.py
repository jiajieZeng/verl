"""Opt-in Add SFT retention: latest N complete saves plus every epoch end."""
import json
from pathlib import Path
import re
import shutil

MARKER = 'add4-complete.json'


def complete_and_prune(root, step, steps_per_epoch, keep_latest, world_size):
    root = Path(root)
    if root.is_symlink() or keep_latest < 1 or steps_per_epoch < 1:
        raise ValueError('Invalid checkpoint retention root or policy')
    current = root / f'global_step_{step}'
    if current.is_symlink() or not current.is_dir():
        raise ValueError('Expected a real checkpoint directory')
    for rank in range(world_size):
        for prefix in ('model', 'optim', 'extra_state'):
            shard = current / f'{prefix}_world_size_{world_size}_rank_{rank}.pt'
            if not shard.is_file() or shard.stat().st_size == 0:
                raise ValueError(f'Incomplete checkpoint: {shard}')
    if not any(current.glob('data_*.pt')):
        raise ValueError('Missing dataloader state')
    marker = current / MARKER
    temporary = marker.with_suffix('.tmp')
    temporary.write_text(json.dumps({'step': step, 'epoch_end': step % steps_per_epoch == 0}) + '\n')
    temporary.replace(marker)

    completed = []
    for path in root.iterdir():
        match = re.fullmatch(r'global_step_([1-9][0-9]*)', path.name)
        if not match or path.is_symlink() or not path.is_dir() or not (path / MARKER).is_file():
            continue
        record = json.loads((path / MARKER).read_text())
        saved_step = int(match[1])
        if record['step'] != saved_step:
            raise ValueError(f'Checkpoint marker mismatch: {path}')
        completed.append((saved_step, path, record['epoch_end']))
    completed.sort()
    retained = {number for number, _, _ in completed[-keep_latest:]}
    retained.update(number for number, _, epoch_end in completed if epoch_end)
    removed = []
    for number, path, _ in completed:
        if number not in retained:
            shutil.rmtree(path)
            removed.append(number)
    return {'retained': sorted(retained), 'removed': removed}


def install_epoch_retention(handler, steps_per_epoch, keep_latest):
    import torch.distributed as dist

    if keep_latest < 1:
        raise ValueError('ADD4_KEEP_LATEST_CHECKPOINTS must be positive')
    # Built-in rotation must not remove epoch checkpoints before our handler runs.
    handler.max_ckpt_to_keep = None
    original_save = handler.save_checkpoint

    def save_checkpoint(step):
        result = original_save(step=step)
        # The SPMD handler returns only after all ranks saved their shards/data.
        error = [None]
        if dist.get_rank() == 0:
            try:
                report = complete_and_prune(handler.default_local_dir, step, steps_per_epoch,
                                            keep_latest, dist.get_world_size())
                print(f'Add checkpoint retention: {report}', flush=True)
            except Exception as exc:
                error[0] = f'{type(exc).__name__}: {exc}'
        dist.broadcast_object_list(error, src=0)
        if error[0]:
            raise RuntimeError(error[0])
        return result

    handler.save_checkpoint = save_checkpoint
