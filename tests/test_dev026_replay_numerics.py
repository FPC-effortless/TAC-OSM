"""Float tolerance must never relax predictions or physical outcome checks."""
import copy
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from dev026_replay_numerics import compare_diagonal  # noqa: E402

BASE = {'logits':[[1.3, -1.3]],'predictions':[0],
        'controls':{'selected':{'entity':1,'total_cost':.6,'decision_probabilities':[.2]}}}


def test_allow_small_inference_rounding_only():
    other=copy.deepcopy(BASE)
    other['logits'][0][0] += 1e-7
    other['controls']['selected']['decision_probabilities'][0] += 1e-8
    compare_diagonal(other, BASE)


@pytest.mark.parametrize('change',['prediction','cost','large_logit','nonfinite'])
def test_reject_scientifically_material_or_invalid_change(change):
    other=copy.deepcopy(BASE)
    if change=='prediction':other['predictions'][0]=1
    elif change=='cost':other['controls']['selected']['total_cost']+=1e-10
    elif change=='large_logit':other['logits'][0][0]+=.01
    else:other['logits'][0][0]=float('nan')
    with pytest.raises(AssertionError):compare_diagonal(other,BASE)
