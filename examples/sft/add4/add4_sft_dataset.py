"""VERL reader for the exported non-thinking Add-4 SFT Parquet."""
import importlib.util
from pathlib import Path

import torch
import torch.nn.functional as F
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset

_spec = importlib.util.spec_from_file_location('add4_parquet_encoding', Path(__file__).with_name('export_add4_parquet.py'))
_encoding = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_encoding)


class Add4SFTDataset(MultiTurnSFTDataset):
    def __getitem__(self, item):
        row = self.dataframe.iloc[item]
        if bool(row['enable_thinking']):
            raise ValueError('This Add-4 dataset requires thinking disabled')
        ids, start, end = _encoding.encode_pair(self.tokenizer, list(self.messages[item]))
        if (len(ids), start, end) != (row['token_count'], row['loss_start'], row['loss_end']):
            raise ValueError('Tokenizer/template differs from the exported dataset')
        if len(ids) > self.max_length:
            raise ValueError(f"{row['id']}: {len(ids)} tokens exceeds {self.max_length}; never truncate")
        ids = torch.tensor(ids, dtype=torch.long)
        mask = torch.zeros_like(ids); mask[start:end] = 1
        positions = torch.arange(ids.numel(), dtype=torch.long)
        if self.processor is not None and 'Qwen2VLImageProcessor' in self.processor.image_processor.__class__.__name__:
            positions = positions.unsqueeze(0).repeat(4, 1)
        result = dict(input_ids=ids, position_ids=positions, loss_mask=mask)
        if self.pad_mode == 'right':
            padding = self.max_length - ids.numel()
            result['attention_mask'] = F.pad(torch.ones_like(ids), (0, padding))
            result['input_ids'] = F.pad(ids, (0, padding), value=self.tokenizer.pad_token_id or 0)
            result['position_ids'] = F.pad(positions, (0, padding))
            result['loss_mask'] = F.pad(mask, (0, padding))
        elif self.pad_mode != 'no_padding':
            raise ValueError('Unsupported pad mode')
        return result
