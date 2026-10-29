import json
import os
from pathlib import Path
import random
import sys
import tempfile
import unittest

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'examples/sft/add7'))
from sampler import BalancedSampler, partition_batch
from prepare_data import check_input, encode_pair, export
from dataset import Add7CachedDataset


def sample(identity='a', split='train'):
    payload = dict(verification=dict(status='passed'), trajectory=dict(tool_results_removed=True,
        events=[dict(message=dict(content=[dict(type='tool_use', name='read', input={'path': 'x'})]))]))
    return dict(id=identity, split=split, split_group=identity, messages=[
        dict(role='system', content='Extract memory.'),
        dict(role='user', content=json.dumps(payload)),
        dict(role='assistant', content='{"summary":"保留原文", "facts":["fixed x"]}')])


class CoverageTests(unittest.TestCase):
    def test_recipe_is_five_epochs_with_wandb(self):
        root = Path(__file__).resolve().parents[2]
        config = OmegaConf.load(root/'examples/sft/add7/qwen35.8gpu.yaml')
        self.assertEqual(config.trainer.total_epochs, 5)
        self.assertEqual(config.trainer.logger, ['console', 'wandb'])
        launcher = (root/'examples/sft/add7/launch_204.sh').read_text()
        self.assertNotIn('WANDB_MODE=disabled -e TOKENIZERS', launcher)
        self.assertIn('WANDB_MODE=${WANDB_MODE:-online}', launcher)
        self.assertIn('-e WANDB_API_KEY', launcher)

    def test_global_batch_membership_and_tail(self):
        lengths = [(i*113)%190000+1000 for i in range(12008)]
        all_ranks = [list(BalancedSampler(lengths, 64, 2, r)) for r in range(2)]
        expected = list(range(len(lengths))); random.Random(1).shuffle(expected)
        self.assertEqual(sorted(all_ranks[0]+all_ranks[1]), list(range(len(lengths))))
        for b in range(188):
            actual = all_ranks[0][b*32:(b+1)*32] + all_ranks[1][b*32:(b+1)*32]
            self.assertEqual(set(actual), set(expected[b*64:(b+1)*64]))
        self.assertEqual(len(all_ranks[0][187*32:]), 20)

    def test_epoch_resume_and_validation(self):
        lengths = list(range(1,399))
        a = BalancedSampler(lengths, 64, 2, 0); b = BalancedSampler(lengths, 64, 2, 0)
        previous = list(a); a.set_epoch(2); b.set_epoch(2)
        self.assertEqual(list(a), list(b)); self.assertNotEqual(previous, list(a))
        ranks = [list(BalancedSampler(lengths, 2, 2, r, shuffle=False)) for r in range(2)]
        self.assertEqual(sorted(sum(ranks, [])), list(range(398)))
        with self.assertRaises(ValueError): BalancedSampler([1,2,3], 64, 2, 0)

    def test_workload_balance(self):
        lengths = [100000,1000,95000,2000,90000,3000,85000,4000]
        groups = partition_batch(list(range(8)), lengths, 2)
        work = lambda group: sum(lengths[i]*(lengths[i]+24576) for i in group)
        self.assertLess(abs(work(groups[0])-work(groups[1])),
                        abs(work(range(0,8,2))-work(range(1,8,2))))

    def test_smoke_uses_longest_global_batch(self):
        lengths = list(range(1,81))
        ranks = [list(BalancedSampler(lengths, 64, 2, r, longest_first=True)) for r in range(2)]
        self.assertEqual(set(ranks[0][:32]+ranks[1][:32]), set(range(16,80)))

    def test_stateful_loader_resume(self):
        from torchdata.stateful_dataloader import StatefulDataLoader
        sampler = BalancedSampler(list(range(1,81)), 64, 2, 0)
        sampler.set_epoch(2)
        loader = StatefulDataLoader(list(range(80)), batch_size=32, sampler=sampler, drop_last=False)
        it = iter(loader); next(it); state = loader.state_dict(); expected = next(it)
        restored_sampler = BalancedSampler(list(range(1,81)), 64, 2, 0)
        restored_sampler.set_epoch(2)
        second = StatefulDataLoader(list(range(80)), batch_size=32,
            sampler=restored_sampler, drop_last=False)
        second.load_state_dict(state)
        self.assertTrue(torch.equal(next(iter(second)), expected))


class EncodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.tokenizer = AutoTokenizer.from_pretrained('/model', local_files_only=True)

    def test_exact_target_and_cache(self):
        row = sample()
        ids, a, b = encode_pair(self.tokenizer, row['messages'])
        self.assertEqual(self.tokenizer.decode(ids[a:b], clean_up_tokenization_spaces=False),
                         row['messages'][-1]['content']+'<|im_end|>')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); src = root/'messages.jsonl'; out = root/'cache'
            src.write_text(json.dumps(row)+'\n')
            export(src, out, self.tokenizer, 'train', 1, 262144, 1)
            ds = Add7CachedDataset(str(out), self.tokenizer,
                OmegaConf.create(dict(pad_mode='no_padding', truncation='error', max_length=262144)))
            result = ds[0]
            self.assertEqual(result['input_ids'].tolist(), ids)
            self.assertEqual(result['loss_mask'].nonzero().flatten().tolist(), list(range(a,b)))
            self.assertEqual(result['position_ids'].tolist(), list(range(len(ids))))
            with self.assertRaises(FileExistsError): export(src, out, self.tokenizer, 'train', 1, 262144, 1)

    def test_reject_results_unverified_thinking(self):
        row = sample(); check_input(row, 'train')
        for change in ('thinking', 'results', 'unverified'):
            row = sample(); payload = json.loads(row['messages'][1]['content'])
            if change=='thinking': row['enable_thinking']=True
            elif change=='results': payload['trajectory']['events'][0]['message']['content'].append({'type':'tool_result'})
            else: payload['verification']['status']='unverified'
            row['messages'][1]['content']=json.dumps(payload)
            with self.assertRaises(ValueError): check_input(row, 'train')


class PackingLossTests(unittest.TestCase):
    def test_microbatch_partition_and_token_normalization(self):
        from verl.utils.dataset.dataset_utils import SFTTensorCollator
        from verl.utils.seqlen_balancing import rearrange_micro_batches
        from verl.utils import tensordict_utils as tu
        from verl.workers.utils.losses import sft_loss
        seqs = [11, 23, 37, 8, 17, 29]
        rows=[]
        for n in seqs:
            mask=torch.zeros(n, dtype=torch.long); mask[-5:-1]=1
            rows.append(dict(input_ids=torch.arange(n), position_ids=torch.arange(n), loss_mask=mask))
        packed = tu.get_tensordict(tensor_dict=SFTTensorCollator('no_padding')(rows),
                                  non_tensor_dict=dict(pad_mode='no_padding', dp_size=1, batch_num_tokens=24))
        batches, indices = rearrange_micro_batches(packed, max_token_len=50, same_micro_num_in_dp=False)
        self.assertEqual(sorted(sum(indices,[])), list(range(len(rows))))
        for batch in batches:
            self.assertLessEqual(int(batch['input_ids'].offsets().diff().sum()), 50)
            for pos in batch['position_ids'].unbind(): self.assertEqual(pos[0].item(),0)
        # Token-weighted loss is invariant under packing; prompt tokens have zero gradients.
        values = [torch.linspace(-3, -.1, n, requires_grad=True) for n in seqs]
        def loss_for(indexes):
            data = tu.get_tensordict(tensor_dict=SFTTensorCollator('no_padding')([rows[i] for i in indexes]),
                                    non_tensor_dict=dict(pad_mode='no_padding', dp_size=1, batch_num_tokens=24))
            output = {'log_probs':torch.nested.as_nested_tensor([values[i] for i in indexes], layout=torch.jagged)}
            return sft_loss(None, output, data)[0]
        packed_loss=sum(loss_for(idx) for idx in indices)
        single_loss=sum(loss_for([i]) for i in range(len(rows)))
        torch.testing.assert_close(packed_loss, single_loss)
        packed_loss.backward()
        for value, row in zip(values, rows):
            torch.testing.assert_close(value.grad, -torch.roll(row['loss_mask'],-1).float()/24)


if __name__=='__main__': unittest.main()
