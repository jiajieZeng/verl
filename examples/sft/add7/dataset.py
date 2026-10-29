"""Memory-mapped token cache: no repeated tokenization or dense max-length padding."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class Add7CachedDataset(Dataset):
    def __init__(self, parquet_files, tokenizer, config, processor=None, max_samples=-1):
        self.root = Path(parquet_files)
        self.manifest = json.loads((self.root/'manifest.json').read_text())
        m = self.manifest
        if (m.get('format') != 'add7-noresults-token-cache-v1' or not m.get('complete')
                or m.get('enable_thinking') is not False or m.get('tool_results_remaining') != 0):
            raise ValueError('Unverified cache')
        if config.pad_mode != 'no_padding' or config.truncation != 'error':
            raise ValueError('Requires no_padding and no truncation')
        if max_samples != -1:
            raise ValueError('Do not silently subset this dataset')
        self.rows = m['rows']
        self.lengths = [r['token_count'] for r in self.rows]
        if len(self.rows) != m['count'] or max(self.lengths) > config.max_length:
            raise ValueError('Invalid row count/context')
        offset = 0
        for row in self.rows:
            if row['offset'] != offset or not 0 < row['loss_start'] < row['loss_end'] < row['token_count']:
                raise ValueError('Invalid cache offsets/supervision span')
            offset += row['token_count']
        if offset != m['total_tokens'] or (self.root/'tokens.bin').stat().st_size != offset * 4:
            raise ValueError('Incomplete token cache')
        self.tokens = np.memmap(self.root/'tokens.bin', mode='r', dtype='<i4')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        r = self.rows[index]
        ids = torch.tensor(self.tokens[r['offset']:r['offset']+r['token_count']].astype(np.int64))
        mask = torch.zeros_like(ids)
        mask[r['loss_start']:r['loss_end']] = 1
        return dict(input_ids=ids, position_ids=torch.arange(len(ids)), loss_mask=mask)
