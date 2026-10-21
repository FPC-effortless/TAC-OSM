import pytest
torch=pytest.importorskip('torch')
from tac_osm.me034_runtime import StablePipeline
from tac_osm.me032_conditional_memory import FrozenContext,CommonHead
from tac_osm.me034_stable_inputs import build_model


def pair(arm):
    torch.manual_seed(2);e=torch.randn(16000,8)
    c=FrozenContext(e,0.,1.,torch.full((16000,),.5),torch.zeros(26),torch.ones(26))
    s=build_model(arm,CommonHead().state_dict(),e,34001,torch.full((16000,),.5),torch.zeros(12),torch.ones(12),0.)
    return c,s


@pytest.mark.parametrize('arm',['native_frozen','history_frozen','gru_frozen'])
def test_shared_pipeline_is_exact_and_preserves_originals(arm):
    c,s=pair(arm);plain=StablePipeline(c,s);shared=StablePipeline(c,s,True)
    x=(torch.ones(2,128,dtype=torch.long),torch.rand(2,128,3),torch.ones(2,dtype=torch.long),torch.ones(2,1))
    torch.testing.assert_close(shared(*x),plain(*x),rtol=0,atol=0)
    assert sum(p.numel() for p in plain.parameters())-sum(p.numel() for p in shared.parameters())==128000
    torch.testing.assert_close(torch.jit.script(shared)(*x),shared(*x),rtol=0,atol=1e-7)
    assert c.emb.weight.data_ptr()!=shared.context.emb.weight.data_ptr()


def test_shared_pipeline_rejects_mutable_or_different_tables():
    c,s=pair('native_adapt')
    with pytest.raises(ValueError,match='immutable'):StablePipeline(c,s,True)
    c,s=pair('native_frozen')
    with torch.no_grad():s.memory.core.emb.weight[0,0]+=1
    with pytest.raises(ValueError,match='equal'):StablePipeline(c,s,True)
