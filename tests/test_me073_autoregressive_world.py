"""Mathematical/causality fixtures only. Capability requires original real data."""
import copy
import torch
from scripts.me073_autoregressive_world import derive_preprocessing,pack,observed
from tac_osm.workflow_me073 import (TrajectoryModel,teacher_path,generate,weighted_quantiles,clock_nll)
from tac_osm.workflow_me071 import detach_belief
from tests.test_me072_marked_world import fixture as previous_fixture


def fixture():
    cases,pre,_=previous_fixture();pre=derive_preprocessing(cases,pre)
    torch.manual_seed(73);model=TrajectoryModel(pre)
    with torch.no_grad():
        for layer in (model.mark_head,model.existence_head,model.clock_head):layer.weight.normal_(std=.02)
    return cases,pre,model


def test_incremental_future_targets_and_boundary_stop_mask():
    cases,_,model=fixture();data=pack(cases,[0],68,2);args,_=observed(model,data)
    _,_,mask,_,_=teacher_path(model,*args,data[5].flatten(0,1),data[6].flatten(0,1),data[7].flatten(0,1),data[8].flatten())
    assert mask[0].tolist()==[True,True,False,False,False,False,False,False]
    assert mask[1].tolist()==[True,False,False,False,False,False,False,False]
    assert data[6][0,0,0]==cases[0]['gap'][-1]
    assert not data[6][0,1].any()


def test_actual_future_labels_cannot_change_generated_futures():
    cases,_,model=fixture();data=pack(cases,[0],length=6);args,_=observed(model,data,all_steps=False)
    original,_=generate(model,*args,particles=32)
    altered=copy.deepcopy(cases);altered[0]['activity'][6:]=1;altered[0]['gap'][6:]=999
    changed=pack(altered,[0],length=6);args2,_=observed(model,changed,all_steps=False)
    again,_=generate(model,*args2,particles=32)
    torch.testing.assert_close(original,again,atol=0,rtol=0)


def test_rollouts_normalize_survive_monotonically_and_accumulate_nonnegative_time():
    cases,_,model=fixture();data=pack(cases,[0,1],length=6);args,state=observed(model,data,all_steps=False)
    saved=[x.clone() for x in state];f,paths=generate(model,*args,particles=64)
    n=model.transition_prior.shape[-1]
    torch.testing.assert_close(f[:,:,:n+1].sum(-1),torch.ones_like(f[:,:,0]),atol=1e-6,rtol=1e-6)
    assert (f[:,1:,n]>=f[:,:-1,n]-1e-6).all()
    assert (paths[:,:,1:,1]>=paths[:,:,:-1,1]).all()
    assert (f[:,:,n+2:n+4]>=f[:,:,n+1:n+3]).all()
    for x,y in zip(state,saved):torch.testing.assert_close(x,y,atol=0,rtol=0)


def test_sampled_previous_activity_and_delay_change_future_native_state():
    _,_,model=fixture();belief=model.initial(2);marks=torch.tensor([1,2]);seconds=torch.tensor([0.,3600.])
    updated,encoded=model.advance(belief,marks,seconds)
    assert not torch.equal(updated.fast[0],updated.fast[1])
    assert not torch.equal(encoded[0],encoded[1])
    assert belief.fast.abs().sum()==0


def test_clock_parameters_are_bounded_even_for_extreme_head_values():
    cases,pre,model=fixture();args,_=observed(model,pack(cases,[0],length=2),all_steps=False)
    with torch.no_grad():model.clock_head.bias.fill_(1e6)
    law=model.step(*args,0);current=args[2];next_mark=torch.ones_like(current)
    clock=model.clock(law,current,next_mark)
    base=model.clock_locations[current,next_mark]
    assert (clock.location-base<=2.+1e-5).all()
    assert (clock.log_scale.exp()<=1.25+1e-6).all()
    # Long measured waits still receive a finite likelihood, never clipped72h.
    occurrence,density=clock_nll(clock,torch.full_like(current,1000.*3600.,dtype=torch.float32))
    assert torch.isfinite(occurrence+density).all()


def test_all_trained_parameters_receive_finite_gradients():
    cases,_,model=fixture();data=pack(cases,[0,1],length=4);args,_=observed(model,data)
    loss,*_=teacher_path(model,*args,data[5].flatten(0,1),data[6].flatten(0,1),data[7].flatten(0,1),data[8].flatten())
    loss.backward();assert torch.isfinite(loss)
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in model.parameters())


def test_real_observation_carry_matches_full_history():
    cases,_,model=fixture();_,a=observed(model,pack(cases,[0],length=32))
    _,tail=observed(model,pack(cases,[0],32,32),detach_belief(a))
    _,whole=observed(model,pack(cases,[0],length=64))
    for x,y in zip(tail,whole):torch.testing.assert_close(x,y,atol=1e-6,rtol=1e-6)


def test_weighted_clock_quantiles_honor_survival_selection():
    values=torch.tensor([[0.,10.,100.,1000.]])
    weights=torch.tensor([[.01,.98,.005,.005]]).log()
    q=weighted_quantiles(values,weights)
    torch.testing.assert_close(q,torch.tensor([[10.,10.,10.]]))


def test_initial_context_depends_only_on_first_recorded_observation():
    cases,_,model=fixture();a=pack(cases,[0],length=6)
    changed=copy.deepcopy(cases);changed[0]['features'][1:]=1234
    b=pack(changed,[0],length=6)
    for index in (9,10,11):torch.testing.assert_close(a[index],b[index],atol=0,rtol=0)
    state=model.initial_from_context(model.event_encoding(*a[9:12]))
    assert state.fast.abs().max()<=.5 and state.slow.abs().max()<=.5


def test_free_run_readout_separates_actual_futures_from_record_boundaries():
    from scripts.me073_autoregressive_world import metrics
    cases,_,model=fixture();data=pack(cases,[0],length=69)
    args,_=observed(model,data,all_steps=False)
    forecasts,paths=generate(model,*args,particles=32)
    _,joint,mask,_,_=teacher_path(model,*args,data[5][:,-1],data[6][:,-1],data[7][:,-1],data[8][:,-1])
    rows=metrics(forecasts,paths,data,joint.detach(),mask)
    assert rows['1']['future_record_cases']==1
    for horizon in ('2','4','8'):
        assert rows[horizon]['future_record_cases']==0
        assert rows[horizon]['record_boundary_cases']==1
        assert rows[horizon]['mean_interval_score_hours'] is None
    assert all(row['generated_path_time_monotone'] for row in rows.values())
