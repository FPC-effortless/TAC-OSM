"""Causal, non-leakage, and training checks for DEV-006 (not capability evidence)."""
from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from tac_osm.development.predictive_world import (  # noqa: E402
    ARMS, PredictiveWorldModel, arm_loss, future_target, make_world_batch, smoke_train,
)


def test_counterfactual_future_requires_past_and_action():
    past = torch.tensor([0,1]); current = torch.tensor([1,1]); action = torch.tensor([0,0])
    actual = future_target(past,current,action)
    assert actual.shape == (2,3)
    assert not torch.equal(actual[0],actual[1])
    changed = future_target(past,current,torch.ones(2,dtype=torch.long))
    assert not torch.equal(actual,changed)
    # Public-only optimal conditional mean has exact irreducible pair error.
    c = current.float()*2-1; a = action.float()*2-1
    public = torch.stack((.4*a*c,.8*c,.25*c),dim=-1)
    assert torch.isclose((actual-public).square().mean(),torch.tensor(1.01/3),atol=1e-6)


def test_future_target_rejects_invalid_values_and_shapes():
    with pytest.raises(ValueError):
        future_target(torch.tensor([0,2]),torch.tensor([0,1]),torch.tensor([1,1]))
    with pytest.raises(ValueError):
        future_target(torch.tensor([[0,1]]),torch.tensor([[0,1]]),torch.tensor([[0,1]]))


def test_batch_pairing_and_no_inference_leakage():
    batch = make_world_batch(43,0,smoke=True)
    assert len(batch.inference())==5
    assert batch.answer.shape==(8,) and batch.next_latent.shape==(8,3)
    for k in range(0,8,2):
        assert batch.answer[k] != batch.answer[k+1]
        assert batch.current[k] == batch.current[k+1]
        assert batch.action[k] == batch.action[k+1]
        assert torch.equal(batch.query[k],batch.query[k+1])
        assert not torch.equal(batch.next_latent[k],batch.next_latent[k+1])
    assert len(set(batch.tasks))==2


def test_world_head_shapes_and_pretraining_integrity():
    model = PredictiveWorldModel().eval()
    batch = make_world_batch(43,0,smoke=True)
    with torch.no_grad():
        state = model.encode(batch.addresses,batch.values,trace=False)
        task,pred = model.predict_both(state,batch.query,batch.current,batch.action)
    assert state.shape == (8,4,8,24)
    assert task.shape == (8,2) and pred.shape == (8,3)
    assert torch.isfinite(pred).all()
    with pytest.raises(ValueError):
        model.predict_both(state,batch.query,batch.current,torch.full_like(batch.action,2))


def test_losses_and_writer_gradients_on_smoke():
    torch.set_num_threads(2)
    out = smoke_train(43,steps=2)
    assert set(out)==set(ARMS)
    for result in out.values():
        assert len(result['smoke_losses'])==2
        assert all(torch.isfinite(torch.tensor(result['smoke_losses'])))
        assert 0 <= result['smoke_task_accuracy']<=1
        assert result['smoke_latent_mse']>=0


def test_auxiliary_corruption_changes_training_target():
    batch = make_world_batch(43,0,smoke=True)
    task_logits = torch.zeros(8,2)
    # Nonconstant predictions let mismatched targets be detected.
    pred = batch.next_latent.clone()
    _,_,matched = arm_loss('joint',task_logits,pred,batch.answer,batch.next_latent)
    corrupted,_,_ = arm_loss('joint_shuffled',task_logits,pred,batch.answer,batch.next_latent)
    assert matched.item()==0.0
    assert corrupted.item()>torch.log(torch.tensor(2.)).item()
