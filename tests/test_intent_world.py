import torch
from tac_osm.variants.intent_world import IntentWorld
from tac_osm.variants.bounded_value_world import BoundedValueWorld
from scripts import me090_train_world as R


def test_reference_targets_are_labels_and_do_not_change_past_inputs():
    row={'state':torch.arange(30.)[:,None].expand(-1,14)/10,'action':torch.arange(30.)[:,None].expand(-1,14)/5,
         'human':torch.zeros(30,dtype=torch.bool),'human_prefix':torch.zeros(31,dtype=torch.int64),'elapsed':torch.ones(30,1)/15}
    rows={('recorded',1):row};x,a,y,v,dt,ref=R.batch(rows,[('recorded',1,8)],history=8)
    assert torch.equal(a[0,-1],row['action'][7])
    assert torch.equal(ref[0,-1,0],row['action'][8])
    changed={**row,'action':row['action'].clone()};changed['action'][8:]+=100
    alternate=R.batch({('recorded',1):changed},[('recorded',1,8)],history=8)
    assert torch.equal(x,alternate[0]) and torch.equal(a,alternate[1])
    assert not torch.equal(ref,alternate[-1])


def test_joint_reference_graph_is_causal_active_and_incrementally_equivalent():
    torch.manual_seed(90090);torch.set_num_threads(1);m=IntentWorld().double()
    source=BoundedValueWorld().double();R.load_initial(m,source.state_dict())
    assert sum(p.numel() for p in m.parameters())==104886
    x=torch.randn(2,9,14,dtype=torch.float64)*.01;a=torch.randn_like(x)*.01;dt=torch.full((2,9,1),1/15,dtype=x.dtype)
    full,final=m(x,a,return_state=True,elapsed=dt);state=None;parts=[]
    for t in range(9):
        out,state=m.step(x[:,t],a[:,t],state,elapsed=dt[:,t]);parts.append(out)
    for k in full:torch.testing.assert_close(full[k],torch.stack([v[k] for v in parts],1),rtol=1e-10,atol=1e-10)
    changed=x.clone();changed[:,5:]+=1;alter=m(changed,a,elapsed=dt)
    for k in full:torch.testing.assert_close(full[k][:,:5],alter[k][:,:5],rtol=0,atol=0)
    (full['reference_prediction'].square().mean()+full['reference_change_logit'].square().mean()+full['reference_lead'].square().mean()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.reference_head.parameters())
