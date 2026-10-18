"""Same learned inference across empty history, padding, receipts and FIFO evictions."""
from pathlib import Path
import sys
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_belief_dev062 import gate_fit,make
from tac_osm.development.belief_agent import configurations
from tac_osm.development.cached_belief_agent import CachedBeliefAgent
from data_belief_dev062 import BeliefWorld


@pytest.mark.parametrize('arm',['tac','no_casm','gru','mlp'])
def test_incremental_state_exactly_matches_full_prefix_with_overflow(arm):
    torch.set_num_threads(1);s=configurations()[arm];base=make(arm,s,192620,gate_fit()).double().eval();cached=CachedBeliefAgent(arm,s['width'],s['aux'],192620).double().eval();cached.load_state_dict(base.state_dict());worlds=[BeliefWorld(19962601,i) for i in range(2)];keys=torch.stack([torch.tensor(w.identities) for w in worlds]).double();a=base.initial(keys);b=cached.initial(keys);d=torch.stack([torch.tensor(w.candidates()[0]) for w in worlds]).double();prices=torch.tensor([[.1,.1],[.1,.1]],dtype=torch.float64)
    def check():
        with torch.no_grad():
            x=base.decide(a,d,prices);y=cached.decide(b,d,prices)
        for key in x:torch.testing.assert_close(y[key],x[key],atol=1e-8,rtol=1e-8)
        assert torch.equal(base.choice(x),cached.choice(y))
    check()
    with torch.no_grad():
        for tick in range(80):
            e=torch.stack([torch.tensor(w.record(2 if tick%2 else 3,tick%2,bit=tick%2,action=2)) for w in worlds]).double();present=torch.tensor([True,tick%3!=0]);a=base.observe(a,e,present);b=cached.observe(b,e,present)
            if tick%7==0 or tick==79:check()


def test_raw_prefix_prefill_is_refreshed_before_first_cached_query():
    torch.set_num_threads(1);s=configurations()['tac'];base=make('tac',s,12,gate_fit()).double().eval();cached=CachedBeliefAgent('tac',s['width'],s['aux']).double().eval();cached.load_state_dict(base.state_dict());w=BeliefWorld(19962602,0);keys=torch.tensor(w.identities)[None].double();a=base.initial(keys);b=cached.initial(keys)
    with torch.no_grad():
        for e in w.events:a=base.observe(a,torch.tensor(e)[None].double())
        b['events']=a['events'].clone();b['present']=a['present'].clone();d=torch.tensor(w.candidates()[0])[None].double();prices=torch.tensor([[.1,.1]],dtype=torch.float64);x=base.decide(a,d,prices);y=cached.decide(b,d,prices)
    torch.testing.assert_close(x['posterior'],y['posterior'],atol=1e-8,rtol=1e-8)
    assert b['cache_ready'].all()
    assert not any(p.requires_grad for p in base.gate.parameters())
