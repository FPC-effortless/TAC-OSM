"""Diagnostic input boundaries, frozen weights and causal controls."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from tac_osm.development import memory_diagnostics as diag  # noqa: E402
from tac_osm.development.noisy_memory import NoisyMemoryModel,parameter_hash  # noqa: E402


@pytest.fixture(autouse=True)
def cpu():
    torch.set_num_threads(2)
    torch.manual_seed(43)


def frozen_model():
    model = NoisyMemoryModel().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def test_probe_training_preserves_frozen_parent_and_changes_probe():
    model = frozen_model()
    before = parameter_hash(model)
    probe,record = diag.train_probe(model,43,smoke=True)
    assert parameter_hash(model)==before==record['parent_hash']
    assert record['initial_hash']!=record['final_hash']
    assert record['nonzero_finite_gradient']
    assert probe.net[0].in_features==32
    assert all(p.grad is None for p in model.parameters())


def test_probe_refuses_unfrozen_model():
    with pytest.raises(ValueError,match='frozen'):
        diag.train_probe(NoisyMemoryModel().eval(),43,smoke=True)


def test_nonoracle_predictions_ignore_evaluator_target_slot_and_label(monkeypatch):
    model = frozen_model()
    probe = diag.ReadProbe(32).eval()
    pair = diag.make_pair(45,32,1,0.0,.08)
    normal = diag.retrieval_row(model,probe,45,32,1,0.0,.08)
    changed = replace(pair,left=replace(pair.left,target=1-pair.left.target,target_slot=(pair.left.target_slot+1)%32),
                       right=replace(pair.right,target=1-pair.right.target,target_slot=(pair.right.target_slot+1)%32))
    monkeypatch.setattr(diag,'make_pair',lambda *a:changed)
    altered = diag.retrieval_row(model,probe,45,32,1,0.0,.08)
    for arm in ('native','uniform_read','no_compute','read_probe','reset'):
        assert normal['predictions'][arm]==altered['predictions'][arm]


def test_competing_writes_match_between_partners_and_have_real_payloads():
    address = torch.randn(1,12).expand(2,-1)
    a,v = diag.competing_writes(7,address,16,1.0)
    assert a.shape==(2,16,12) and v.shape==(2,16)
    assert torch.equal(a[0],a[1]) and torch.equal(v[0],v[1])
    assert not torch.equal(a[0,0],address[0])
    assert set(v.flatten().tolist())=={0,1}


@pytest.mark.parametrize('delay',(0,1,4,16,32))
def test_delayed_carry_controls_keep_exact_paired_ceiling(delay):
    row = diag.carry_row(frozen_model(),77,delay,1.0)
    assert row['write_calls']==33+delay
    for arm in ('reset','no_commit'):
        assert sum(x==t for x,t in zip(row['predictions'][arm],row['target']))==1
    correct = sum(x==t for x,t in zip(row['predictions']['verified'],row['target']))
    swapped = sum(x==t for x,t in zip(row['predictions']['swapped'],row['target']))
    assert correct+swapped==2


def test_calibration_ties_select_highest_threshold():
    rows = [{'valid':False,'probability':.4},{'valid':True,'probability':.95}]
    threshold,table = diag.calibrate(rows)
    assert threshold==.9 and len(table)==5
    assert diag.verifier_errors(rows,threshold)['weighted_error']==0


def test_calibration_and_evaluation_never_share_random_cases():
    model = frozen_model()
    calibration = diag.verifier_cases(model,43,calibration=True)
    evaluation = diag.verifier_cases(model,43,calibration=False)
    assert not {r['seed'] for r in calibration}&{r['seed'] for r in evaluation}
    assert len(evaluation)==384
    assert sum(r['valid'] for r in evaluation)==192


def test_error_rates_report_tradeoff_instead_of_only_accuracy():
    rows = [{'valid':False,'probability':.7},{'valid':True,'probability':.8}]
    assert diag.verifier_errors(rows,.6)['false_accept_rate']==1
    assert diag.verifier_errors(rows,.9)['false_reject_rate']==1
    with pytest.raises(ValueError):
        diag.verifier_errors([rows[0]],.6)


def test_probe_features_do_not_include_oracle_or_current_bit():
    model = frozen_model()
    address = torch.randn(2,12)
    state = torch.randn(2,4,8,24)
    read = diag.raw_reads(model,state,address)
    assert read.shape==(2,32)
    assert torch.allclose(read[0],diag.raw_reads(model,state[:1],address[:1])[0],atol=1e-6)


def test_frozen_development_protocol_retains_original_content():
    p = diag.protocol()
    assert p['status']=='FROZEN_DEVELOPMENT_DESIGN_NOT_CONFIRMATORY'
    assert p['carry_delays']==[0,1,4,16,32]
    assert p['probe']['steps']==400
    assert p['parent_head']=='a089a12d46de6b9b06779f880f5a1e689c9ed48a'
