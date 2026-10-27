"""Fail-closed mathematical accounting, never model-skill evidence."""
import copy
import json
import pytest
from tac_osm.research_operator_budget import signature_work,prove_pair,digest,verify_profile


def op(name,inputs,outputs,calls=1):
    sig=json.dumps([name,[["torch.float32",s] for s in inputs],[["torch.float32",s] for s in outputs]])
    return dict(stage='forward',signature=sig,calls=calls)


def pair(extra=None):
    base=[op('aten::mm.',[[16,100],[100,100]],[[16,100]],100)]
    return {'operators':base}, {'operators':base+([extra] if extra else [])}


def test_unknown_differing_math_blocks():
    a,b=pair(op('aten::mysterious.',[[16]],[[16]]))
    with pytest.raises(ValueError,match='unbounded'):prove_pair(a,b)


def test_common_unknown_cancels_but_complete_difference_is_reported():
    a,b=pair(op('aten::add.Tensor',[[16]],[[16]]))
    proof=prove_pair(a,b)
    assert proof['relative_full_math_gap_upper_bound']>0
    assert len(proof['complete_unequal_operators'])==1


def test_excess_compute_and_duplicate_rows_block():
    a,b=pair(op('aten::mul.Tensor',[[1000000]],[[1000000]],100))
    with pytest.raises(ValueError,match='exceeds'):prove_pair(a,b)
    a['operators']*=2
    with pytest.raises(ValueError,match='duplicate'):prove_pair(a,b)


def test_matrix_and_weight_invariance_required():
    ops=pair()[0]['operators']
    profile=dict(schema='tac-osm.complete-operator-profile.v1',profiles=[dict(seed=seed,arm=arm,kind=kind,
        weights_before='x',weights_after='x',parameters=99390,finite_gradient_tensors=43,
        operators=ops,operator_sha256=digest(ops)) for seed in (1,2) for arm in ('assoc_historic_query','fast_slow_historic_query') for kind in ('first','warm')])
    assert len(verify_profile(profile))==4
    bad=copy.deepcopy(profile);bad['profiles'][0]['weights_after']='changed'
    with pytest.raises(ValueError,match='weights'):verify_profile(bad)
    profile['profiles'][0]['operator_sha256']='fabricated'
    with pytest.raises(ValueError,match='fingerprint'):verify_profile(profile)


def test_reduction_bound_and_matmul():
    s=op('aten::sum.',[[16,8]], [[]])['signature']
    assert signature_work(s)==128
    s=op('aten::bmm.',[[2,3,4],[2,4,5]], [[2,3,5]])['signature']
    assert signature_work(s)==240
