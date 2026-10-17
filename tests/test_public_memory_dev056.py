import inspect
import importlib.util
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from tac_osm.development.learned_stack import PublicEvent
from tac_osm.development.public_memory_deadline import (
    DeadlineModel,PublicHazardWorld,PublicFilter,oracle_transitions,collect,public_features)


def test_closed_mask_holds_every_state_tensor_and_skips_backend():
    torch.set_num_threads(1)
    model=DeadlineModel(8561001,'masked_mtsk').eval()
    with torch.no_grad():
        model.write_gate.weight.zero_();model.write_gate.bias.fill_(-30)
        public=torch.randn(3,11)
        state=model.initial(public)
        state=(torch.randn_like(state[0]),state[1])
        with patch.object(model.memory,'_step',wraps=model.memory._step) as backend:
            result=model.update(state,public)
            backend.assert_not_called()
        assert model.last_write_count==0
        for old,new in zip(state,result):torch.testing.assert_close(old,new,rtol=0,atol=0)


def test_forced_open_mask_matches_unmasked_recurrence():
    plain=DeadlineModel(8561002,'mtsk').eval()
    masked=DeadlineModel(8561002,'masked_mtsk').eval()
    masked.load_state_dict(plain.state_dict(),strict=False)
    with torch.no_grad():
        masked.write_gate.weight.zero_();masked.write_gate.bias.fill_(30)
        ds,_=collect(8561003,3,length=4)
        a,b=plain.initial(ds[0][:,0]),masked.initial(ds[0][:,0])
        for t in range(4):
            a=plain.update(a,ds[0][:,t]);b=masked.update(b,ds[0][:,t])
            for old,new in zip(a,b):torch.testing.assert_close(old,new,rtol=0,atol=0)


def test_gate_gets_task_gradients_and_parameters_match():
    ds,_=collect(8561004,4,length=4)
    model=DeadlineModel(8561004,'masked_mtsk')
    loss=torch.nn.functional.binary_cross_entropy_with_logits(model(ds[0],ds[1]),ds[2])
    loss.backward()
    assert model.write_gate.weight.grad.abs().sum()>0
    gru=DeadlineModel(8561004,'gru_matched')
    for arm in ('mtsk','masked_mtsk'):
        assert abs(DeadlineModel(8561004,arm).trainable_count-gru.trainable_count)/gru.trainable_count<=.02


def test_public_oracle_equals_explicit_eight_state_bayes_update():
    table=oracle_transitions()
    np.testing.assert_allclose(table.sum(-1),1)
    filt=PublicFilter(table,.25)
    filt.update(PublicEvent(0,(1,0),(0,0),0),initial=True)
    before=filt.belief.copy()
    event=PublicEvent(1,(0,1),(0,0),1)
    brute=np.zeros((2,4))
    for z in range(2):
        for s in range(4):
            for nz in range(2):
                for ns in range(4):
                    sensor=(.8 if ns//2==0 else .2)*(.8 if ns%2==1 else .2)
                    brute[nz,ns]+=before[z,s]*table[z,1,s,ns]*(.75 if nz==z else .25)*sensor
    brute/=brute.sum()
    filt.update(event)
    np.testing.assert_allclose(filt.belief,brute,atol=1e-14)
    assert list(inspect.signature(filt.update).parameters)==['event','initial']


def test_public_hazard_replay_and_masked_incremental_restoration():
    a,b=PublicHazardWorld(8561005,0,'fast'),PublicHazardWorld(8561005,0,'fast')
    for action in [1,2,0,3,4]*5:
        assert a.step(action)==b.step(action)
    assert public_features(a.last,.25)[-2]==.25
    ds,_=collect(8561006,3,length=4)
    model=DeadlineModel(8561006,'masked_mtsk').eval()
    with torch.no_grad():
        batch=model(ds[0],ds[1]);state=model.initial(ds[0][:,0]);outputs=[]
        for t in range(4):
            state=model.update(state,ds[0][:,t])
            outputs.append(model.predict(state,ds[1][:,t],ds[0][:,t,-1]*20))
        torch.testing.assert_close(batch,torch.stack(outputs,1),rtol=1e-5,atol=1e-6)
        restored=DeadlineModel(42,'masked_mtsk').eval();restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(batch,restored(ds[0],ds[1]),rtol=0,atol=0)


def test_streaming_boundary_accepts_late_public_ticks_and_rejects_labels():
    import pytest
    path=Path(__file__).resolve().parents[1]/'scripts/demo_public_memory_dev056.py'
    spec=importlib.util.spec_from_file_location('dev056_demo',path)
    demo=importlib.util.module_from_spec(spec);spec.loader.exec_module(demo)
    receipt={'action':1,'sensors':[0,1],'exact':[0,0],'tick':23,'hazard':.25}
    event,hazard=demo.parse_event(receipt)
    assert event.tick==23 and hazard==.25
    with pytest.raises(ValueError):demo.parse_event({**receipt,'regime':0})
    with pytest.raises(ValueError):demo.parse_event({**receipt,'hazard':.1})
