"""Available-packet scheduling preserves the learned neural controller."""
from pathlib import Path
import sys
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_belief_dev062 import make,gate_fit
from tac_osm.development.belief_agent import configurations
from tac_osm.development.batched_belief_agent import BatchedBeliefAgent,BatchedCachedBeliefAgent
from data_belief_dev062 import BeliefWorld


@pytest.mark.parametrize('kind',[BatchedBeliefAgent,BatchedCachedBeliefAgent])
@pytest.mark.parametrize('arm',['tac','no_casm','gru','mlp'])
def test_packet_and_deferred_stream_match_after_noise_padding_and_eviction(kind,arm):
    torch.set_num_threads(1);s=configurations()[arm];base=make(arm,s,199629,gate_fit()).double().eval();m=kind(arm,s['width'],s['aux']).double().eval();m.load_state_dict(base.state_dict());ws=[BeliefWorld(1996291,i) for i in range(2)];keys=torch.stack([torch.tensor(w.identities) for w in ws]).double();a=base.initial(keys);b=m.initial(keys);d=torch.stack([torch.tensor(w.candidates()[0]) for w in ws]).double();price=torch.tensor([[.1,.1],[.1,.1]],dtype=torch.float64)
    def check():
        x=base.decide(a,d,price);y=m.decide(b,d,price)
        for k in x:torch.testing.assert_close(x[k],y[k],atol=1e-8,rtol=1e-8)
        assert torch.equal(base.choice(x),m.choice(y))
    with torch.no_grad():
        check()
        for offset in range(0,96,4):
            events=torch.stack([torch.stack([torch.tensor(w.record(5 if (offset+t)%5==0 else 3,(offset+t)%2,bit=t%2)) for t in range(4)]) for w in ws]).double();present=torch.tensor([[True]*4,[True,False,True,False]])
            for t in range(4):a=base.observe(a,events[:,t],present[:,t])
            if offset%8:b=m.observe_many(b,events,present)
            else:
                for t in range(4):b=m.observe(b,events[:,t],present[:,t])
            check()
        torch.testing.assert_close(a['events'],b['events'],atol=0,rtol=0)
        assert torch.equal(a['present'],b['present'])
