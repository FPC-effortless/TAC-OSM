"""ME051 software-only safeguards: constructed data NEVER qualifies model performance."""
import zipfile
import numpy as np
import pytest
import torch
from tac_osm.me051_long_retention import (
    ARMS,HORIZONS,RealRetentionPredictor,real_quarter_target,genuine_rating_loss)
from scripts import me051_train_real_long_retention as study


def _numeric_batch(H=256,batch_size=3):
    torch.manual_seed(51051)
    b=batch_size
    movie=torch.zeros(b,256,dtype=torch.long)
    rating=torch.zeros(b,256)
    gap=torch.zeros(b,256)
    valid=torch.zeros(b,256)
    movie[:,-H:]=torch.randint(1,3953,(b,H))
    rating[:,-H:]=(torch.randint(1,6,(b,H)).float()-3.)/2.
    gap[:,-H:]=torch.rand(b,H)
    valid[:,-H:]=1.
    return {"movie":movie,"rating":rating,"gap":gap,"valid":valid,
            "candidate":torch.randint(1,3953,(b,)),
            "target":torch.randint(1,6,(b,)).float()}


def test_me051_real_retention_quarters_only_use_existing_observed_past():
    x=_numeric_batch(16,1)
    x["rating"][:,-16:-12]=-1.
    x["rating"][:,-4:]=1.
    assert torch.allclose(real_quarter_target(x["rating"],x["valid"],True),torch.tensor([1.]))
    assert torch.allclose(real_quarter_target(x["rating"],x["valid"],False),torch.tensor([5.]))
    with pytest.raises(ValueError):
        real_quarter_target(x["rating"],torch.zeros_like(x["valid"]),True)


def test_me051_native_old_recent_are_exact_equal_capacity_and_optimizable():
    torch.manual_seed(51051)
    a=RealRetentionPredictor("tac_old")
    b=RealRetentionPredictor("tac_recent")
    assert a.num_active()==b.num_active()
    b.load_state_dict(a.state_dict(),strict=True)
    x=_numeric_batch(256,2)
    pred,probe=a(x["movie"],x["rating"],x["gap"],x["valid"],x["candidate"])
    pred2,probe2=b(x["movie"],x["rating"],x["gap"],x["valid"],x["candidate"])
    assert pred.shape==probe.shape==(2,)
    assert torch.equal(pred,pred2) and torch.equal(probe,probe2)
    for model,old in ((a,True),(b,False)):
        loss,_,_=genuine_rating_loss(model,x,old)
        loss.backward()
        assert torch.isfinite(loss)
        assert model.state.transition.weight.grad is not None
        assert model.retrieve.weight.grad is not None
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.parameters() if p.requires_grad)


def test_me051_all_comparators_have_valid_numeric_gradients_on_16_and_64_real_contract():
    for H in (16,64):
        x=_numeric_batch(H,2)
        for arm in ("gru_old","gru_recent","context_recent"):
            m=RealRetentionPredictor(arm)
            loss,_,_=genuine_rating_loss(m,x,oldest=arm.endswith("_old"))
            loss.backward()
            assert torch.isfinite(loss)
            assert all(p.grad is not None and torch.isfinite(p.grad).all()
                       for p in m.parameters() if p.requires_grad)


def test_me051_frozen_FIT_labels_and_tied_timestamp_cannot_leak_test(tmp_path,monkeypatch):
    z=tmp_path/"synthetic-file-contract-only.zip"
    lines=[]
    for user in range(1,4):
        for i in range(40):
            stamp=900_000+i*10
            if i==29:stamp=900_000+28*10
            lines.append(f"{user}::{1+i}::{1+i%5}::{stamp}\n")
    with zipfile.ZipFile(z,"w") as archive:
        archive.writestr("ml-1m/ratings.dat","".join(lines))
    monkeypatch.setattr(study,"RECORDS",120)
    monkeypatch.setattr(study,"USERS",3)
    monkeypatch.setattr(study,"MIN_FIT_ROWS",20)
    state,assignment=study.build_metadata(z)
    assert study.apply_ratings(z,state,assignment,"FIT")==84
    assert all((rec["rating"][rec["fit"]:]<0).all() for rec in state.values())
    # A later observation at exactly the preceding timestamp is unavailable.
    for data in state.values():
        duplicate=np.flatnonzero(np.diff(data["time"])==0)
        assert len(duplicate)==1
        assert data["prior"][duplicate[0]+1]==data["prior"][duplicate[0]]
    example=study.eligible(state,"FIT",16)[0]
    batch=study.make_batch(state,[example],16)
    assert int(batch["valid"].sum())==16
    assert float(batch["target"][0]) in (1.,2.,3.,4.,5.)
    with pytest.raises(RuntimeError,match="future or unknown"):
        study.make_batch(state,[(1,35)],16)
    assert study.apply_ratings(z,state,assignment,"FINAL")==120
    assert all((rec["rating"]>0).all() for rec in state.values())
