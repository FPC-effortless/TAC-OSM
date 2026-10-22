"""ME052 numerical/data-contract tests ONLY; no synthetic task performance evidence."""
import numpy as np
import pytest
import torch

from scripts import me052_train_real_robot as exp


def fixtures():
    examples={}
    for ep in range(10):
        n=305
        frames=np.zeros((n,14),dtype=np.float32)
        commands=np.zeros((n,14),dtype=np.float32)
        flags=np.zeros(n,dtype=np.bool_)
        frames[:,0]=np.arange(n)*.001
        commands[:,0]=np.arange(n)*.002
        flags[272:274]=True
        examples[ep]={"state":torch.from_numpy(frames),
             "action":torch.from_numpy(commands),
             "human":torch.from_numpy(flags),"n":n,
             "source_sha":"0"*64}
    return examples


def test_me052_native_and_established_rival_have_matching_active_parameter_counts():
    original=exp.count(exp.network("native_tac"))
    rival=exp.count(exp.network("gru_world"))
    assert original==37985
    assert rival==38295
    assert abs(original-rival)/max(original,rival)<.01
    assert exp.SEEDS==(52001,52002,52003,52004,52005)
    assert exp.HISTORY==256 and exp.BATCH==2


def test_me052_actual_episode_schedule_is_identical_per_architecture_and_real_only():
    records=fixtures()
    schedule,stats=exp.sample_schedule(records,exp.SEEDS[0])
    assert len(schedule)==exp.UPDATES
    assert stats["source_genuine_eligible_windows"]>100
    assert stats["source_genuine_onset_windows"]>=10
    schedule2,stats2=exp.sample_schedule(records,exp.SEEDS[0])
    assert schedule==schedule2 and stats==stats2
    for actual_pair in schedule:
        s,a,h=exp.get_batch(records,actual_pair)
        assert s.shape==(2,265,14)
        assert a.shape==(2,264,14)
        assert h.shape==(2,265,1)


def test_me052_real_sequence_loss_and_feedback_backpropagate_for_both_models():
    records=fixtures()
    samples=((0,0),(1,0))
    states,actions,human=exp.get_batch(records,samples)
    for arm in exp.ARMS:
        model=exp.network(arm)
        state=model.initial(2)
        with torch.no_grad():
            for t in range(256):
                state,*_=exp.single_step(model,state,states,actions,human,t)
        total=None
        for t in range(256,264):
            state,pred,target,onset,mask=exp.single_step(
                model,state,states,actions,human,t)
            losses=model.genuine_logged_losses(pred,target,onset,mask)
            term=losses["observed_physical_mse"]+.2*losses["masked_real_takeover_bce"]
            total=term if total is None else total+term
        total.backward()
        assert torch.isfinite(total)
        assert any(p.grad is not None for p in model.parameters())


def test_me052_train_stream_has_genuine_onset_not_mislabeled_during_human_control():
    records=fixtures()
    states,actions,human=exp.get_batch(records,((0,0),(1,0)))
    assert bool(human[0,272]) and bool(human[0,273])
    # In true source: first human frame t+1=272 is onset.
    target=((~human[:,271]) & human[:,272]).float()
    masked=(~human[:,271])
    assert masked.all() and target.sum().item()==2
    assert not (~human[:,272]).any()
