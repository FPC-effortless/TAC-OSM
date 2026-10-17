"""DEV-043 frozen-transfer controls: split isolation, pair ranking and non-leakage."""
import importlib.util
from pathlib import Path

import pytest
torch=pytest.importorskip("torch")
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("dev043",ROOT/"scripts/run_head_transfer_dev043.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_split_namespace_and_pair_budget():
    train=m.split_cases("head-train")
    calibration=m.split_cases("calibration")
    evaluation=m.split_cases("evaluation")
    assert len(train)==180 and len(calibration)==45 and len(evaluation)==144
    b,_=m.assemble(train,0)
    a,_=m.assemble(train,100)
    d,_=m.assemble(calibration,100)
    e,_=m.assemble(evaluation,100)
    sets=[{x.fingerprint for x in cases} for cases in (a,b,d,e)]
    assert [len(x) for x in sets]==[360,360,90,288]
    assert all(not sets[i]&sets[j] for i in range(4) for j in range(i))
    assert not any("dev042" in c.fingerprint for c in a)


def test_mixed_head_fraction_counts():
    train=m.split_cases("head-train")
    a,c=m.assemble(train,0); b,d=m.assemble(train,100)
    dummy=torch.zeros(360,8,3,44)
    source=dict(original=dict(x=dummy,y=torch.zeros(360,8,3,dtype=torch.long),conditions=c),
                order=dict(x=dummy+1,y=torch.ones(360,8,3,dtype=torch.long),conditions=d))
    for fraction,expected in [(0,0),(10,36),(50,180)]:
        mixed=m.mix_train(source,fraction)
        assert sum(z["mode"]=="order" for z in mixed["conditions"])==expected
    assert len(a)==len(b)==360


def test_positive_calibration_preserves_pairwise_logit_order():
    logits=torch.tensor([[[[0.,6.]]],[[[0.,5.]]]],dtype=torch.float32)
    for params in (dict(scales=[.1,2,1],bias=[0,0,0]),
                   dict(scales=[3,1,1],bias=[-10,0,0])):
        p=m.calibrated(logits,params)
        assert p[0,0,0]>p[1,0,0]


def test_calibration_method_names_are_explicit():
    assert m.METHODS==("identity","temperature_positive","platt_positive")


def test_pairwise_ranking_is_threshold_free():
    rows=[]
    for pair in range(144):
        deadline=("before","at","after")[pair//48]
        for partner in (0,1):
            y=int(partner==0) if deadline!="before" else 1
            p=.999 if partner==0 else .998
            rows.append(dict(deadline=deadline,partner=partner,target_entity=0,
                probabilities=[[p,0.,0.]]+[[0.,0.,0.]]*7,
                labels=[[y,0,1]]+[[0,0,0]]*7))
    met=m.metrics(rows)
    assert met["pairwise_ranking"]==1
    assert met["hard_mature_wait"]==.5


def test_disjointness_gate_rejects_reused_fingerprints():
    d=dict(conditions=[dict(fingerprint="dup")])
    with pytest.raises(AssertionError):m.disjoint([d,d])


def test_pairwise_oracle_is_labeled_retrospective():
    rows=[]
    for p,lab in [(.95,1),(.94,0)]:
        rows.append(dict(probabilities=[[p,0,0]]+[[0,0,0]]*7,
             labels=[[lab,0,0]]+[[0,0,0]]*7))
    diagnostic=m.retrospective_pairwise_oracle(rows)
    assert "hindsight" in diagnostic["note"].lower() or "test labels" in diagnostic["note"].lower()
