"""DEV-011: frozen finite-memory information-probe mechanism tests."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))

import pytest
import torch

from tac_osm.development.protected_memory import ProtectionModel
from tac_osm.development.noisy_memory import parameter_hash
from tac_osm.development.protection_benchmark import make_pair, tensors
from run_frozen_plm_probes_dev import (
    DecoderProbe, frozen_features, evaluate_probe, encode_examples,
    training_cases, fit_probe, feature_for,
)


def test_probe_input_shapes_match_fixed_768_memory():
    writer=ProtectionModel(mode="always_update")
    pair=make_pair(11,32,4,0.,.08,"retention")
    a,v,q,c=tensors([pair.left,pair.right])
    with torch.no_grad():
        state=writer.encode(a,v,trace=False)
        feats=frozen_features(writer,state,q,c)
    assert state.shape==(2,4,8,24)
    assert feats["keyed"].shape==(2,46)
    assert feats["state"].shape==(2,782)
    for kind in ("keyed_linear","keyed_mlp","state_mlp"):
        assert DecoderProbe(kind)(feature_for(kind,feats)).shape==(2,2)


def test_invalid_shape_fails_without_silent_history_expansion():
    model=DecoderProbe("state_mlp")
    with pytest.raises(ValueError,match="shape"):
        model(torch.zeros(2,46))
    with pytest.raises(ValueError):
        DecoderProbe("oracle_label")


def test_frozen_writer_not_optimized_and_probe_receives_gradients():
    torch.set_num_threads(2)
    writer=ProtectionModel(mode="always_update")
    writer.eval()
    for parameter in writer.parameters():
        parameter.requires_grad_(False)
    before=parameter_hash(writer)
    rows,features=encode_examples(writer,training_cases(43,True))
    model,_=fit_probe("keyed_mlp",features,rows,seed=43,smoke=True)
    assert parameter_hash(writer)==before
    assert all(p.grad is None for p in writer.parameters())
    assert all(not p.requires_grad for p in writer.parameters())
    assert any(p.abs().sum()>0 for p in model.parameters())


def test_opposite_past_pairs_have_identical_public_present():
    pair=make_pair(71,128,16,1.,.25,"acquisition")
    a,v,q,c=tensors([pair.left,pair.right])
    assert torch.equal(a[0],a[1]) and torch.equal(q[0],q[1])
    assert torch.equal(c[0],c[1])
    assert int((v[0]!=v[1]).sum())==1
    assert pair.left.target==1-pair.right.target


def test_zero_state_cannot_distinguish_opposite_past_pairs():
    writer=ProtectionModel(mode="always_update")
    rows,features=encode_examples(writer,training_cases(59,True))
    probe=DecoderProbe("state_mlp")
    results=evaluate_probe(probe,features,rows,zero_memory=True)
    assert results["accuracy"]==0.5
    assert all(r["prediction"][0]==r["prediction"][1] for r in results["rows"])


def test_swap_states_across_matched_pair_is_no_oracle_access():
    writer=ProtectionModel(mode="always_update")
    rows,features=encode_examples(writer,training_cases(43,True))
    probe=DecoderProbe("keyed_linear")
    normal=evaluate_probe(probe,features,rows,zero_memory=False)
    swapped=evaluate_probe(probe,features,rows,zero_memory=True)
    assert normal["episodes"]==swapped["episodes"]==8
    assert all(r["target"][0]!=r["target"][1] for r in normal["rows"])
    assert swapped["accuracy"]==0.5
