"""Encode complete, accepted no-results Add-7 JSONL once; preserve answer bytes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import itertools
import json
from pathlib import Path


def encode_pair(tokenizer, messages):
    if [m['role'] for m in messages] != ['system', 'user', 'assistant']:
        raise ValueError('Expected system,user,assistant')
    answer = messages[-1]['content']
    obj = json.loads(answer)
    if not isinstance(obj, dict) or set(obj) != {'summary', 'facts'}:
        raise ValueError('Expected Add-7 summary/facts JSON')
    if not isinstance(obj['summary'], str) or not isinstance(obj['facts'], list):
        raise ValueError('Invalid summary/facts types')
    if not all(isinstance(f, str) and f.strip() for f in obj['facts']):
        raise ValueError('Invalid fact')
    prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    full = tokenizer.apply_chat_template(messages, tokenize=False,
        add_generation_prompt=False, enable_thinking=False)
    target = answer + '<|im_end|>'
    if full != prefix + target + '\n':
        raise ValueError('Non-thinking template altered answer or prefix')
    encoded = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded['input_ids'], encoded['offset_mapping']
    left, right = len(prefix), len(prefix) + len(target)
    selected = [i for i, (a, b) in enumerate(offsets) if a >= left and b <= right and b > a]
    if not selected or selected != list(range(selected[0], selected[-1] + 1)):
        raise ValueError('Invalid completion span')
    start, end = selected[0], selected[-1] + 1
    if offsets[start][0] != left or offsets[end-1][1] != right:
        raise ValueError('Token crosses supervision boundary')
    if tokenizer.decode(ids[start:end], skip_special_tokens=False,
                        clean_up_tokenization_spaces=False) != target:
        raise ValueError('Supervised answer changed')
    return ids, start, end


def check_input(row, split):
    if row.get('enable_thinking', False) is not False:
        raise ValueError('Thinking must be disabled for SFT')
    if row.get('split') != split:
        raise ValueError(f'Expected {split} split')
    if not row.get('id') or not row.get('split_group'):
        raise ValueError('Missing identity/group')
    payload = json.loads(row['messages'][1]['content'])
    if payload['verification']['status'] not in ('passed', 'failed'):
        raise ValueError('Unverified source')
    trajectory = payload['trajectory']
    if trajectory.get('tool_results_removed') is not True:
        raise ValueError('Not the no-tool-results batch')
    for event in trajectory['events']:
        if event.get('type') == 'attachment' or 'toolUseResult' in event:
            raise ValueError('Observation/attachment remains')
        blocks = event.get('message', {}).get('content')
        if isinstance(blocks, list) and any(b.get('type') == 'tool_result' for b in blocks):
            raise ValueError('tool_result remains')


def export(source, output, tokenizer, split, expected, context, workers):
    import numpy as np
    # Refuse both overwritten caches and partially completed attempts.
    output.mkdir(parents=True, exist_ok=False)
    rows, seen, offset = [], set(), 0
    def encode(line):
        row = json.loads(line)
        check_input(row, split)
        ids, start, end = encode_pair(tokenizer, row['messages'])
        if len(ids) > context:
            raise ValueError(f"{row['id']}: {len(ids)} > {context}; never truncate")
        return row, ids, start, end
    with source.open() as src, (output/'tokens.bin').open('wb') as dst, ThreadPoolExecutor(workers) as pool:
        while lines := list(itertools.islice(src, 16)):
            for row, ids, start, end in pool.map(encode, lines):
                if row['id'] in seen:
                    raise ValueError('Duplicate sample')
                seen.add(row['id'])
                np.asarray(ids, dtype='<i4').tofile(dst)
                rows.append(dict(id=row['id'], group=row['split_group'], offset=offset,
                    token_count=len(ids), loss_start=start, loss_end=end))
                offset += len(ids)
            if len(rows) % 160 == 0:
                print(json.dumps(dict(split=split, encoded=len(rows))), flush=True)
    if len(rows) != expected:
        raise ValueError(f'Expected {expected} rows, got {len(rows)}; incomplete delivery')
    mapped = np.memmap(output/'tokens.bin', dtype='<i4', mode='r')
    if len(mapped) != offset:
        raise ValueError('Cache length mismatch')
    # Read back every target and compare to original answer, not reserialized JSON.
    with source.open() as src:
        for meta, line in zip(rows, src, strict=True):
            row = json.loads(line)
            a, b = meta['offset'] + meta['loss_start'], meta['offset'] + meta['loss_end']
            if tokenizer.decode(mapped[a:b].tolist(), skip_special_tokens=False,
                    clean_up_tokenization_spaces=False) != row['messages'][-1]['content'] + '<|im_end|>':
                raise ValueError('Persisted target differs')
    report = dict(format='add7-noresults-token-cache-v1', complete=True, split=split,
        source=str(source.resolve()), source_bytes=source.stat().st_size, rows=rows,
        count=len(rows), total_tokens=offset, max_tokens=max(r['token_count'] for r in rows),
        context=context, enable_thinking=False, tool_results_remaining=0,
        all_answers_read_back=True, loss='original summary/facts JSON + im_end only')
    (output/'manifest.json').write_text(json.dumps(report, ensure_ascii=False) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'rows'}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tokenizer', required=True)
    p.add_argument('--split', choices=['train', 'val'], required=True)
    p.add_argument('--expected', type=int, required=True)
    p.add_argument('--context', type=int, default=262144)
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True, trust_remote_code=False)
    export(a.source, a.output, tokenizer, a.split, a.expected, a.context, a.workers)


if __name__ == '__main__':
    main()
