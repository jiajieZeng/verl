import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('export_add4', ROOT / 'examples/sft/add4/export_add4_parquet.py')
export = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(export)


class CharacterTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert not tokenize and enable_thinking is False
        prefix = ''.join(m['content'] for m in messages[:2]) + '<assistant><think>\n\n</think>\n\n'
        return prefix if add_generation_prompt else prefix + messages[-1]['content'] + '<|im_end|>\n'

    def __call__(self, text, **kwargs):
        return dict(input_ids=[ord(x) for x in text], offset_mapping=[(i, i+1) for i in range(len(text))])

    def decode(self, ids, **kwargs):
        return ''.join(chr(x) for x in ids)


def messages():
    return [dict(role='system',content='Extract facts.'),
            dict(role='user',content='Historical reasoning and a tool call remain here.'),
            dict(role='assistant',content=json.dumps({'facts':['缓存 condition is retained.']},ensure_ascii=False))]


def test_loss_excludes_history_headers_and_empty_thinking():
    tokenizer=CharacterTokenizer();m=messages()
    ids,start,end=export.encode_pair(tokenizer,m)
    assert tokenizer.decode(ids[start:end])==m[-1]['content']+'<|im_end|>'
    assert 'Historical reasoning' in tokenizer.decode(ids[:start])
    assert tokenizer.decode(ids[:start]).endswith('<think>\n\n</think>\n\n')
    assert tokenizer.decode(ids[end:])=='\n'


def test_template_rewriting_label_is_rejected():
    class Bad(CharacterTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            value=super().apply_chat_template(messages,**kwargs)
            return value if kwargs['add_generation_prompt'] else value.replace('缓存','changed')
    with pytest.raises(ValueError,match='changed the facts'):
        export.encode_pair(Bad(),messages())


@pytest.mark.parametrize('obj',[{'facts':[{'content':'x'}]},{'facts':['']},{'facts':[],'summary':'x'}])
def test_bad_labels_rejected(obj):
    with pytest.raises(ValueError):export.strict_facts(obj)


def test_historical_wrapper_normalization_is_lossless():
    assert export.flatten_facts([{'content':'A'},['B',{'text':'C'}]])==['A','B','C']
    with pytest.raises(ValueError):export.flatten_facts({'summary':'not a fact'})


def test_issue_identity_overlap():
    assert export.issue_key('django/django#11087')==export.issue_key('django__django-11087')
    assert export.issue_key('django/django#11088')!=export.issue_key('django/django#11087')
