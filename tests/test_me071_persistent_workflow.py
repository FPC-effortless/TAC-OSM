"""Generated fixtures verify implementation contracts, never real capability."""
import copy
import numpy as np
import pytest
import torch

from scripts.me069_workflow_source import fit_preprocessing,encode_cases
from scripts.me071_persistent_workflow import forward_chunk,pack,metrics,frozen_preprocessing
from scripts.me069_workflow_source import digest
from scripts.validate_research_parity import check_manifest,ParityError
from tac_osm.workflow_me071 import (PersistentWorkflowModel,EntityMemory,detach_belief,
    assimilate_masked,predictive_loss,fit_temperature,cqr_correction,calibrated_seconds)


def fixture():
    raw=[{"id":str(i),"events":[["A|COMPLETE" if t%2 else "B|START","r",i*10000+t*t]
        for t in range(70)]} for i in range(2)]
    pre=fit_preprocessing(raw)
    pre["log_wait_quantile_prior"]=[[0.,1.,5.]]*(len(pre["activities"])+1)
    cases=encode_cases(raw,pre)
    for c in cases:
        c["history_log_prior"]=np.asarray([pre["transition_log_probability"][a] for a in c["activity"]],np.float32)
    torch.manual_seed(71);model=PersistentWorkflowModel(pre)
    with torch.no_grad():model.output.weight.normal_(std=.01)
    return cases,pre,model


def test_detached_chunk_carry_preserves_values_and_truncates_gradients():
    cases,_,m=fixture();a,b=pack(cases,[0,1],0,32),pack(cases,[0,1],32,32)
    _,first=forward_chunk(m,a);carried=detach_belief(first)
    assert all(not x.requires_grad and x.grad_fn is None for x in carried)
    _,tail=forward_chunk(m,b,carried)
    _,whole=forward_chunk(m,pack(cases,[0,1],0,64))
    for x,y in zip(tail,whole):torch.testing.assert_close(x,y,atol=1e-6,rtol=1e-6)
    _,reset=forward_chunk(m,b)
    assert not torch.allclose(reset.memory,tail.memory)


def test_quantiles_are_ordered_and_all_model_parameters_train():
    cases,_,m=fixture();data=pack(cases,[0,1])
    prediction,_=forward_chunk(m,data)
    q=prediction.log_wait_quantiles
    assert (q[:,0]>=0).all() and (q[:,1]>=q[:,0]).all() and (q[:,2]>=q[:,1]).all()
    loss=predictive_loss(prediction,data[5].flatten(),data[6].flatten(),data[7].flatten())
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_future_events_do_not_change_earlier_forecasts():
    cases,_,m=fixture();data=pack(cases,[0])
    original,_=forward_chunk(m,data);changed=list(data)
    changed[0]=data[0].clone();changed[0][:,16:]=1
    later,_=forward_chunk(m,tuple(changed))
    torch.testing.assert_close(original.activity_logits[:16],later.activity_logits[:16],rtol=0,atol=0)
    torch.testing.assert_close(original.log_wait_quantiles[:16],later.log_wait_quantiles[:16],rtol=0,atol=0)


def test_masked_assimilation_accumulates_elapsed_time_and_hides_observations():
    cases,_,m=fixture();data=pack(cases,[0],0,8)
    encoded=m.event_encoding(*data[:3]);mask=torch.ones(1,8,dtype=torch.bool);mask[:,2:6]=False
    original=assimilate_masked(m,encoded,data[3],mask)
    mutated=encoded.clone();mutated[:,2:6]=100
    hidden=assimilate_masked(m,mutated,data[3],mask)
    for a,b in zip(original,hidden):torch.testing.assert_close(a,b,rtol=0,atol=0)
    selected=[0,1,6,7];hours=data[3][:,selected].clone()
    hours[:,2]=data[3][:,2:7].sum(1)
    manual=assimilate_masked(m,encoded[:,selected],hours,torch.ones(1,4,dtype=torch.bool))
    for a,b in zip(original,manual):torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_entity_interleaving_preserves_independent_state_and_rejects_time_reversal():
    cases,_,m=fixture();data=pack(cases,[0,1],0,8);encoded=m.event_encoding(*data[:3])
    serial=EntityMemory(m);mixed=EntityMemory(m)
    for entity in range(2):
        for t in range(8):serial.observe(str(entity),encoded[entity:entity+1,t],float(t*t))
    for t in range(8):
        for entity in range(2):mixed.observe(str(entity),encoded[entity:entity+1,t],float(t*t))
    for entity in ("0","1"):
        for a,b in zip(serial.states[entity],mixed.states[entity]):torch.testing.assert_close(a,b,rtol=0,atol=0)
    with pytest.raises(ValueError,match="out-of-order"):mixed.observe("0",encoded[:1,0],-1.)
    before=mixed.states["1"].fast.clone();mixed.reset("0")
    torch.testing.assert_close(before,mixed.states["1"].fast)


def test_temperature_fits_only_supplied_labels_and_keeps_identity_option():
    logits=torch.tensor([[8.,0.],[8.,0.],[8.,0.],[8.,0.]])
    labels=torch.tensor([0,0,0,1])
    t=fit_temperature(logits,labels)
    assert t>1.
    assert torch.nn.functional.cross_entropy(logits/t,labels)<torch.nn.functional.cross_entropy(logits,labels)


def test_cqr_finite_sample_rank_and_float64_seconds_conversion():
    q=np.tile([0.,1.,2.],(19,1));wait=np.expm1(np.arange(19)/5)
    expected=np.sort(np.maximum(q[:,0]-np.log1p(wait),np.log1p(wait)-q[:,2]))[17]
    correction=cqr_correction(q,wait)
    assert correction==pytest.approx(expected)
    seconds=calibrated_seconds(torch.tensor(q,dtype=torch.float32),correction)
    assert seconds.dtype==torch.float64 and (seconds[:,2]>=seconds[:,0]).all()
    with pytest.raises(ValueError):cqr_correction(q[:1],wait[:1])


def test_calibrated_pinball_scores_the_adjusted_quantiles():
    array=np.array([[-.5,-1.,1.,.2,1.,2.]])
    wait=np.array([np.expm1(3.)]);labels=np.array([0])
    result=metrics(array,labels,wait,correction=.5)
    adjusted=np.array([0.,1.,2.5]);delta=3.-adjusted
    expected=np.maximum(np.array([.05,.5,.95])*delta,(np.array([.05,.5,.95])-1)*delta)
    np.testing.assert_allclose(result['log1p_pinball_q05_q50_q95'],expected)


def test_development_lane_cannot_admit_multiple_architectures_or_missing_originals():
    plan={"schema_version":"tac-osm.parity.v1","study_id":"ME071_TEST_ONLY","study_type":"capability_development",
        "registered_before_first_fit":True,"no_architecture_superiority_claim":True,"architecture_count":1,
        "scientific_purpose":"build one integrated native architecture on authentic data",
        "real_world_source":{"observed_real_outcomes":True,"sha256":"a"*64,"license":"actual source terms"},
        "seeds":[1,2],"trainable_parameters":100000,"deployed_parameters":100000,
        "training_budget":{"epochs":2,"batch_size":2,"backprop_events":32,"expected_targets_per_seed":100},
        "repository_storage_plan":"original model bytes in tracked replay bundle before scoring",
        "split_sha256":{k:"b"*64 for k in ("FIT","TEMP","CQR","REPORT")}}
    assert check_manifest(plan)["eligible_for_architecture_superiority"] is False
    changed=copy.deepcopy(plan);changed["architecture_count"]=2
    with pytest.raises(ParityError):check_manifest(changed)
    with pytest.raises(ParityError):check_manifest(plan,"final")


def test_restore_uses_exact_frozen_statistics_and_rejects_identity_changes():
    original={'activities':{'a':1},'resources':{'r':1},'time_prior':[[1.]]}
    reconstructed=copy.deepcopy(original);reconstructed['time_prior'][0][0]+=1e-12
    restored=frozen_preprocessing(original,reconstructed,digest(original))
    assert restored is original and restored['time_prior'][0][0]==1.
    changed=copy.deepcopy(original);changed['time_prior'][0][0]=2.
    with pytest.raises(ValueError,match='checksum'):
        frozen_preprocessing(changed,reconstructed,digest(original))
    reconstructed['activities']['future']=2
    with pytest.raises(ValueError,match='vocabulary'):
        frozen_preprocessing(original,reconstructed,digest(original))
