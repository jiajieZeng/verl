"""Full-vocabulary batch adapter using the existing VERL OPD lifecycle."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil

import torch
import torch.distributed as dist

from add4_opd_core import prompt_from_sft_ids
from add4_opd_full import score_full_to_file
from add4_opd_trainer import OPDBatchAdapter
from add4_opd_monitor import phase, observed


class FullOPDBatchAdapter(OPDBatchAdapter):
    def prepare(self, data, sequence_lengths):
        import verl.utils.tensordict_utils as tu
        from tensordict import NonTensorData
        self.step += 1
        self.verify_services()
        with phase(self, 'weight_sync'):
            version = self.sync_weights()
        try:
            prompts = [prompt_from_sft_ids(ids.tolist(), mask.tolist())
                       for ids, mask in zip(data['input_ids'].unbind(), data['loss_mask'].unbind(), strict=True)]
            local = {'ok': True, 'prompts': prompts}
        except Exception as error:
            local = {'ok': False, 'error': type(error).__name__}
        gathered = [None] * self.world
        dist.all_gather_object(gathered, local)
        if not all(item['ok'] for item in gathered):
            raise ValueError('Invalid Add prompt on a training rank')
        groups = [item['prompts'] for item in gathered]
        # These are temporary, run/attempt-owned cache slots, never teacher-host files.
        folder = (Path(self.config['full_score_dir']) / self.config['run_id'] /
                  self.attempt / f'slot-{self.step % 2}')

        def generate_all():
            flat = [prompt for group in groups for prompt in group]
            folder.mkdir(parents=True, exist_ok=True)
            required = len(flat) * self.config['max_new_tokens'] * self.config['valid_vocab_size'] * 4
            if shutil.disk_usage(folder).free < required:
                raise RuntimeError('Insufficient disk for a full-vocabulary score slot')
            def one(pair):
                index, prompt = pair
                response = observed(self, 'rollout_service', self.student.sample, prompt, self.config, self.eos, self.processor,
                    seed=self.config['seed'] + self.step * len(flat) + index, expected_weight_version=version)
                scores = observed(self, 'teacher_service', score_full_to_file, self.teacher, prompt, response, self.config,
                                           folder / f'sample-{index:06d}.npy')
                return dict(prompt_ids=prompt, response_ids=response, sample_index=index, teacher_scores=scores)
            with ThreadPoolExecutor(max_workers=self.config['max_concurrency']) as pool:
                records = list(pool.map(one, enumerate(flat)))
            audit = Path(self.config['audit_dir']) / self.config['run_id'] / self.attempt
            audit.mkdir(parents=True, exist_ok=True)
            with (audit / f'step-{self.step:08d}.json').open('x') as stream:
                json.dump(dict(step=self.step, student_weight_version=version,
                    teacher_weight_version=self.config['teacher_weight_version'],
                    loss_type='full_vocab_reverse_kl', valid_vocab_size=self.config['valid_vocab_size'],
                    temperature=self.config['temperature'], distillation_temperature=1.0,
                    entropy_coef=self.config['entropy_coef'], teacher_score_storage='two_slot_fp32_cache',
                    samples=records), stream)
            return records

        with phase(self, 'generate_score_wall'):
            records = self.rank_zero(generate_all)
        monitor = getattr(self.trainer, '_opd_monitor', None)
        if monitor:
            monitor.records(records)
        self.verify_services()
        # Only small IDs/paths are broadcast. Every local rank reads its response rows from disk.
        available = all(Path(r['teacher_scores']['path']).is_file() for r in records)
        flags = [None] * self.world
        dist.all_gather_object(flags, available)
        if not all(flags):
            raise ValueError('Teacher score cache must be visible to every training rank')
        offset = sum(len(group) for group in groups[:self.rank])
        records = records[offset:offset + len(prompts)]
        axes = data['position_ids'].shape[1] if data['position_ids'].dim() == 3 else None
        ids, masks, refs, positions = [], [], [], []
        for record in records:
            prompt, response = record['prompt_ids'], record['response_ids']
            length = len(prompt) + len(response)
            ids.append(torch.tensor(prompt + response, dtype=torch.long))
            masks.append(torch.tensor([0] * len(prompt) + [1] * len(response), dtype=torch.long))
            refs.append(torch.cat([torch.full((len(prompt), 2), -1, dtype=torch.long),
                                   torch.tensor([[record['sample_index'], i] for i in range(len(response))])]))
            pos = torch.arange(length, dtype=torch.long)
            positions.append(pos.unsqueeze(0).repeat(axes, 1) if axes else pos)
        for key, values in [('input_ids', ids), ('loss_mask', masks), ('position_ids', positions), ('opd_full_rows', refs)]:
            data[key] = torch.nested.as_nested_tensor(values, layout=torch.jagged)
        tu.assign_non_tensor(data, temperature=1.0, compute_loss=True, return_model_output=False,
                             opd_full_store=NonTensorData(str(folder)),
                             global_token_num=NonTensorData(sequence_lengths(self.trainer, data)))
        return data
