"""Keep each shuffled global batch; balance its compute across equal-size DP groups."""
import random
from torch.utils.data import Sampler


def partition_batch(indices, lengths, replicas):
    if len(indices) % replicas:
        raise ValueError('Exact coverage requires batch divisible by DP size')
    size = len(indices) // replicas
    groups, loads = [[] for _ in range(replicas)], [0] * replicas
    for i in sorted(indices, key=lambda i: lengths[i], reverse=True):
        rank = min((r for r in range(replicas) if len(groups[r]) < size), key=lambda r: (loads[r], r))
        groups[rank].append(i)
        loads[rank] += lengths[i] * (lengths[i] + 24576)
    return groups


class BalancedSampler(Sampler):
    def __init__(self, lengths, batch_size, replicas, rank, seed=1, shuffle=True, longest_first=False):
        if not lengths or batch_size <= 0 or replicas <= 0 or not 0 <= rank < replicas:
            raise ValueError('Invalid sampler configuration')
        if len(lengths) % replicas or batch_size % replicas:
            raise ValueError('No drop/duplicate policy: dataset and batch must divide DP size')
        self.lengths, self.batch_size = lengths, batch_size
        self.replicas, self.rank, self.seed, self.shuffle = replicas, rank, seed, shuffle
        self.longest_first = longest_first
        self.epoch = 0

    def __len__(self):
        return len(self.lengths) // self.replicas

    def set_epoch(self, epoch):
        self.epoch = epoch

    # Deliberately non-stateful: StatefulDataLoader restores its yielded-index
    # count; the trainer sets the restored epoch before constructing the iterator.

    def __iter__(self):
        indices = list(range(len(self.lengths)))
        if self.longest_first:
            indices.sort(key=lambda i: self.lengths[i], reverse=True)
        elif self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            yield from partition_batch(indices[start:start+self.batch_size], self.lengths, self.replicas)[self.rank]
