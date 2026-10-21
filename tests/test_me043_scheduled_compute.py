import torch
from tac_osm.me038_innovation_forecast import build_model as old_model
from tac_osm.me043_scheduled_compute import ARMS,build_model,objective


def test_scheduled_native_learning_really_skips_seven_updates_and_retains_cheap_weights():
    torch.set_num_threads(1)
    net=build_model('scheduled_001',96,43)
    old=old_model('context_cross',96,43)
    for name,value in old.state_dict().items():assert torch.equal(net.state_dict()[name],value)
    x=torch.randn(2,96,3);y=torch.randn(2,96,3);calls=[]
    hook=net.memory.register_forward_pre_hook(lambda *_:calls.append(1))
    opt=torch.optim.AdamW(net.parameters(),lr=.001)
    for step in range(1,17):
        opt.zero_grad(set_to_none=True);loss,_=objective(net,x,y,step);loss.backward();opt.step()
        assert len(calls)==step//8
    assert net.memory.transition.weight.grad is not None
    assert net.memory.transition.weight.grad.abs().sum()>0
    hook.remove()


def test_scheduled_all_comparators_use_active_capacity_within_one_percent():
    for h in (96,720):
        counts=[sum(p.numel() for p in build_model(arm,h,43).parameters()) for arm in ARMS]
        assert max(counts)/min(counts)<1.01
