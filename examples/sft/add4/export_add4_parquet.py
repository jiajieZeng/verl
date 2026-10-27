#!/usr/bin/env python3
"""Export complete Add-4 pairs as non-thinking, facts-only SFT Parquet."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import itertools
import json
from pathlib import Path
import re
import shutil


def read(path):
    return json.loads(Path(path).read_text())


def save(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def dumps(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(',', ':'))


def strict_facts(obj):
    if not isinstance(obj, dict) or set(obj) != {'facts'}:
        raise ValueError('Expected facts-only JSON')
    if not isinstance(obj['facts'], list) or not all(isinstance(x, str) and x.strip() for x in obj['facts']):
        raise ValueError('Expected nonempty fact strings')
    return obj


def flatten_facts(value):
    """Verify the existing lossless wrapper normalization used by validation labels."""
    if isinstance(value, str) and value.strip():
        return [value]
    if isinstance(value, list):
        return [s for x in value for s in flatten_facts(x)]
    if isinstance(value, dict) and len(value) == 1 and next(iter(value)) in ('text', 'content', 'fact'):
        return flatten_facts(next(iter(value.values())))
    raise ValueError('Unsupported historical fact wrapper')


def encode_pair(tokenizer, messages):
    """Keep the official template; mask the entire prompt and empty think prefix."""
    if [m['role'] for m in messages] != ['system', 'user', 'assistant']:
        raise ValueError('Expected exactly system, user, assistant')
    answer = messages[-1]['content']
    strict_facts(json.loads(answer))
    prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False,
                                         enable_thinking=False)
    target = answer + '<|im_end|>'
    if full != prefix + target + '\n':
        raise ValueError('Official template changed the facts or the non-thinking prefix')
    encoded = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded['input_ids'], encoded['offset_mapping']
    left, right = len(prefix), len(prefix) + len(target)
    selected = [i for i, (start, end) in enumerate(offsets) if start >= left and end <= right and end > start]
    if not selected or selected != list(range(selected[0], selected[-1] + 1)):
        raise ValueError('Completion token span is empty or discontinuous')
    start, end = selected[0], selected[-1] + 1
    if offsets[start][0] != left or offsets[end - 1][1] != right:
        raise ValueError('Token crosses the prompt/completion boundary')
    if tokenizer.decode(ids[start:end], skip_special_tokens=False, clean_up_tokenization_spaces=False) != target:
        raise ValueError('Loss span differs from facts plus EOS')
    return ids, start, end


def issue_key(source_id):
    value = source_id.lower()
    match = re.fullmatch(r'([^/]+)/([^#]+)#(\d+)', value)
    if match:
        return f'{match[1]}__{match[2]}-{match[3]}'
    return value


def accepted_label(folder, model, memory=None, normalized=False):
    response = read(folder / 'response.json')
    choice = response['choices'][0]
    if response['model'] != model or choice['finish_reason'] != 'stop':
        raise ValueError('Wrong teacher or incomplete response')
    raw = json.loads(choice['message']['content'])
    label = read(memory) if memory else raw
    strict_facts(label)
    expected = {'facts': flatten_facts(raw['facts'])} if normalized and set(raw) == {'facts'} else raw
    if expected != label:
        raise ValueError('Accepted label differs from the original response')
    return label


def pair_rows(train_root, validation_root):
    meta = read(train_root / 'metadata.json')
    system = (train_root / 'prompts/ordinary_add.md').read_text()
    vm = read(validation_root / 'manifest.json')
    if vm['model'] != meta['model'] or (validation_root / 'prompts/ordinary_add.md').read_text() != system:
        raise ValueError('Validation must use the same teacher and Add-4 prompt')
    for row in read(train_root / 'manifest.json'):
        label = accepted_label(Path(row['output']), meta['model'], row['memory_path'])
        yield dict(id=row['key'], split='train', dataset=row['dataset'], source_id=row['source_id'],
                   source_key=row['key'], segment_index=0, group_id=issue_key(row['source_id']),
                   enable_thinking=False, messages=[dict(role='system', content=system),
                       dict(role='user', content=Path(row['trajectory']).read_text()),
                       dict(role='assistant', content=dumps(label))],
                   provenance=dumps(dict(input_path=row['trajectory'], memory_path=row['memory_path'],
                       response_path=str(Path(row['output'])/'response.json'), teacher_model=meta['model'],
                       original_prompt_arm=row['prompt_arm'], label_wrapper_normalized=False)))
    yield from validation_rows(validation_root, system, meta['model'])


def validation_rows(validation_root, system, model):
    """Read accepted teacher labels with the original source/segment mapping."""
    vm = read(validation_root / 'manifest.json')
    if vm['model'] != model or (validation_root / 'prompts/ordinary_add.md').read_text() != system:
        raise ValueError('Validation must use the same teacher and Add-4 prompt')
    inputs = {r['key']: r for r in vm['sources']}
    for segment in vm['segments']:
        key = segment['key']; a = read(validation_root / 'accepted' / (key + '.json'))
        folder = Path(a['output'])
        label = accepted_label(folder, vm['model'], a.get('normalized_memory'), a.get('format_normalized', False))
        path = inputs[key]['trajectory']
        yield dict(id=key, split='validation', dataset=segment['dataset'], source_id=segment['source_id'],
                   source_key=segment['source_key'], segment_index=segment['segment_index'],
                   group_id=issue_key(segment['source_id']), enable_thinking=False,
                   messages=[dict(role='system', content=system),dict(role='user', content=dumps(read(path))),
                             dict(role='assistant', content=dumps(label))],
                   provenance=dumps(dict(input_path=path, memory_path=a.get('normalized_memory', str(folder/'memory.json')),
                       response_path=str(folder/'response.json'),teacher_model=vm['model'],original_prompt_arm=a['arm'],
                       label_wrapper_normalized=a.get('format_normalized', False))))


def export(args):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    if args.output.exists():
        raise ValueError('Output already exists; keep prior exports unchanged')
    args.output.mkdir(parents=True)
    tokenizer_dir = args.output / 'tokenizer'; tokenizer_dir.mkdir()
    for name in ['tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'config.json', 'vocab.json', 'merges.txt']:
        shutil.copyfile(args.tokenizer / name, tokenizer_dir / name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    schema = pa.schema([(k, pa.string()) for k in ['id','split','dataset','source_id','source_key','group_id']]
        + [('segment_index',pa.int32()),('enable_thinking',pa.bool_()),
           ('messages',pa.list_(pa.struct([('role',pa.string()),('content',pa.string())]))),
           ('token_count',pa.int32()),('loss_start',pa.int32()),('loss_end',pa.int32()),
           ('completion_tokens',pa.int32()),('within_context',pa.bool_()),('provenance',pa.string())])
    writers = {s:pq.ParquetWriter(args.output/(name+'.parquet.pending'),schema,compression='zstd')
               for s,name in [('train','train'),('validation','validation')]}
    counts=Counter(); datasets=Counter(); lengths=[]; seen=set()
    def encode(row):
        ids,start,end=encode_pair(tokenizer,row['messages'])
        row.update(token_count=len(ids),loss_start=start,loss_end=end,completion_tokens=end-start,
                   within_context=len(ids)<=args.context_limit)
        return row
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            pending=iter(pair_rows(args.train,args.validation))
            while batch:=list(itertools.islice(pending,16)):
                encoded=list(pool.map(encode,batch))
                for split in writers:
                    part=[r for r in encoded if r['split']==split]
                    if part:writers[split].write_table(pa.Table.from_pylist(part,schema=schema))
                for row in encoded:
                    identity=(row['split'],row['id'])
                    if identity in seen:raise ValueError('Duplicate sample')
                    seen.add(identity);counts[row['split']]+=1;datasets[(row['split'],row['dataset'])]+=1
                    lengths.append({k:row[k] for k in ['id','split','dataset','source_id','group_id','token_count','completion_tokens','loss_start','loss_end','within_context']})
                if len(lengths)%160==0:
                    save(args.output/'progress.json',dict(phase='exporting',rows=len(lengths),splits=dict(counts)))
                    print('exported',len(lengths),flush=True)
    finally:
        for writer in writers.values():writer.close()
    if counts!={'train':4000,'validation':220}:
        raise ValueError(f'Unexpected export counts {dict(counts)}')
    # Verify the persisted Arrow schema, row order, text and loss boundaries by readback.
    originals=iter(pair_rows(args.train,args.validation));readback=0
    for name in ['train','validation']:
        for batch in pq.ParquetFile(args.output/(name+'.parquet.pending')).iter_batches(batch_size=16):
            for row in batch.to_pylist():
                original=next(originals)
                if any(row[k]!=v for k,v in original.items()):raise ValueError('Parquet round-trip changed a pair')
                if row['enable_thinking'] is not False:raise ValueError('Thinking enabled')
                readback+=1
        (args.output/(name+'.parquet.pending')).rename(args.output/(name+'.parquet'))
    if readback!=4220 or next(originals,None) is not None:raise ValueError('Incomplete round-trip')
    train_groups={r['group_id'] for r in lengths if r['split']=='train'}
    val_groups={r['group_id'] for r in lengths if r['split']=='validation'}
    overlap=sorted(train_groups & val_groups)
    save(args.output/'overlap-audit.json',dict(method='Exact source ID, with owner/repo#number canonicalized to owner__repo-number; not semantic deduplication',overlapping_source_ids=overlap,
        rows=[r for r in lengths if r['group_id'] in overlap],policy='Preserve all requested training and validation rows; disclose overlap.'))
    save(args.output/'lengths.json',lengths)
    overflow=[r for r in lengths if not r['within_context']]
    save(args.output/'over-context.json',dict(context_limit=args.context_limit,rows=overflow,policy='All complete rows retained in Parquet. No truncation, filtering or splitting. Resolve before training with this limit.'))
    summary=dict(phase='complete',train_rows=4000,validation_rows=220,validation_datasets={d:datasets[('validation',d)] for d in ['pro','verified']},
        enable_thinking=False,loss='Only facts JSON plus im_end; mask system, historical user record, assistant header and empty thinking prefix.',
        teacher_model=read(args.train/'metadata.json')['model'],tokenizer=str(tokenizer_dir),context_limit=args.context_limit,
        over_context=len(overflow),max_tokens=max(r['token_count'] for r in lengths),overlap_sources=len(overlap),
        source_manifest=str(args.train/'manifest.json'),validation_manifest=str(args.validation/'manifest.json'),
        validation_mode='Teacher-forcing loss only; no coding runs or avg@4.',parquet_roundtrip_verified=readback,
        training_started=False,all_training_rows_retained=True)
    save(args.output/'manifest.json',summary);save(args.output/'progress.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['train','validation','tokenizer','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--context-limit',type=int,default=262144);p.add_argument('--workers',type=int,default=4)
    export(p.parse_args())
