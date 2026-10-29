"""CPU validation of complete caches and the resolved training recipe."""
import argparse
import json
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf
from dataset import Add7CachedDataset
from sampler import BalancedSampler


def validate(root):
    config = OmegaConf.create(dict(pad_mode='no_padding', truncation='error', max_length=262144))
    datasets = {s: Add7CachedDataset(str(root/'prepared'/s), None, config) for s in ('train', 'val')}
    if len(datasets['train']) != 12008 or len(datasets['val']) != 398:
        raise ValueError('Expected 12008 train and independent 398 val')
    overlap = {r['group'] for r in datasets['train'].rows} & {r['group'] for r in datasets['val'].rows}
    if overlap:
        raise ValueError(f'Train/val overlap: {len(overlap)}')
    info = {}
    for split, ds in datasets.items():
        ranks = [list(BalancedSampler(ds.lengths, 64 if split=='train' else 2, 2, r,
                                    shuffle=split=='train')) for r in range(2)]
        indices = ranks[0] + ranks[1]
        if sorted(indices) != list(range(len(ds))):
            raise ValueError('Dropped or duplicated samples')
        info[split] = dict(rows=len(ds), mean_tokens=float(np.mean(ds.lengths)),
            max_tokens=max(ds.lengths), total_tokens=sum(ds.lengths),
            supervised_tokens=sum(r['loss_end']-r['loss_start'] for r in ds.rows),
            exact_coverage=True)
    report = dict(ready=True, training_started=False, gpu_tested=False, world_size=8,
        sp=4, dp=2, global_batch=64, max_tokens_per_microbatch=262144,
        steps_per_epoch=188, epochs=5, total_steps=940, train_val_group_overlap=0, datasets=info)
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--report', type=Path)
    a = p.parse_args()
    result = validate(a.root)
    if a.report:
        a.report.write_text(json.dumps(result, indent=2)+'\n')
