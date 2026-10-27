import copy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'examples/sft/add4'))
from add4_opd_full import full_request, parse_full_scores
from add4_opd_transport import trim_full_scoring_response
from test_add4_full import response_for


def test_trim_preserves_full_distribution_and_all_validation_metadata():
    request = full_request([0, 1], [2, 3], 6)
    original = response_for([0, 1], [2, 3])
    original['meta_info']['output_top_logprobs'] = [[[-0.2, 1, None]]]
    original['meta_info']['new_server_field'] = {'keep': True}
    snapshot = copy.deepcopy(original)
    trimmed = trim_full_scoring_response(request, original)
    assert original == snapshot
    expected = copy.deepcopy(original)
    del expected['meta_info']['output_top_logprobs']
    assert trimmed == expected
    np.testing.assert_array_equal(
        parse_full_scores(original, [0, 1], [2, 3], 4, 6, 'frozen'),
        parse_full_scores(trimmed, [0, 1], [2, 3], 4, 6, 'frozen'))


@pytest.mark.parametrize('change', [
    {'sampling_params': {'max_new_tokens': 1}},
    {'stream': True}, {'top_logprobs_num': 2},
    {'return_flat_raw_top_logprobs_b64': False},
    {'input_ids': [[0, 1, 2, 3]]},
])
def test_other_requests_pass_through_untouched(change):
    request = full_request([0, 1], [2, 3], 6)
    request.update(change)
    original = response_for([0, 1], [2, 3])
    original['meta_info']['output_top_logprobs'] = [[[-0.2, 1, None]]]
    assert trim_full_scoring_response(request, original) is original
