import pytest
import torch
from tac_osm.variants.clock_value_world import ClockValueWorld
from tac_osm.variants.clock_value_execution import ClockValueCache


def test_entity_interleaving_matches_full_and_retry_is_pure():
    torch.manual_seed(85086);torch.set_num_threads(1)
    model=ClockValueWorld().double().eval();runtime=ClockValueCache(model,model_revision='test-model',preprocessing_revision='test-input',max_entities=2)
    x=torch.randn(2,6,14,dtype=torch.float64)*.01;a=torch.zeros_like(x);dt=torch.full((2,6,1),1/15,dtype=x.dtype);dt[:,3]=1.
    calls=[];handle=model.register_forward_hook(lambda *args:calls.append(1))
    # step() deliberately calls the pure tensor graph directly; count step calls.
    original=model.step;counter=[0]
    def counted(*args,**kwargs):counter[0]+=1;return original(*args,**kwargs)
    model.step=counted
    for t in range(6):
        for entity in range(2):
            got=runtime.step(str(entity),t,x[entity:entity+1,t],a[entity:entity+1,t],elapsed=dt[entity:entity+1,t])
            expected=model.forecast_last(x[entity:entity+1,:t+1],a[entity:entity+1,:t+1],elapsed=dt[entity:entity+1,:t+1])
            for k in got:torch.testing.assert_close(got[k],expected[k],rtol=1e-10,atol=1e-10)
    count=counter[0];retry=runtime.step('0',5,x[:1,5],a[:1,5],elapsed=dt[:1,5]);retry['current'].zero_()
    again=runtime.step('0',5,x[:1,5],a[:1,5],elapsed=dt[:1,5]);assert counter[0]==count
    assert again['current'].abs().sum()>0
    with pytest.raises(ValueError):runtime.step('0',5,x[:1,5],a[:1,5],elapsed=dt[:1,5]*2)
    with pytest.raises(ValueError):runtime.step('0',7,x[:1,5],a[:1,5],elapsed=dt[:1,5])
    with pytest.raises(RuntimeError):runtime.step('third',0,x[:1,5],a[:1,5],elapsed=dt[:1,5])
    runtime.reset('1');runtime.step('third',0,x[:1,5],a[:1,5],elapsed=dt[:1,5])
    with torch.no_grad():next(model.parameters()).add_(.01)
    with pytest.raises(RuntimeError):runtime.step('0',6,x[:1,5],a[:1,5],elapsed=dt[:1,5])
    assert not runtime._entries
    handle.remove()
