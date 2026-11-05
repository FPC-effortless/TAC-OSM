"""No-fit independently constructed CSL125R2 scales; no nested weight sharing."""
import torch
import pytest
from tac_osm.csl125_r2 import CSL125R2
from scripts.profile_csl125_r2_scales import configs

@pytest.mark.parametrize('size,expected',[('1m',1_037_060),('10m',9_739_421),('125m',123_822_574)])
def test_actual_meta_parameter_capacity_all_heads_included(size,expected):
    with torch.device('meta'):
        model=CSL125R2(configs()[size],device='meta')
    assert model.parameter_count()==expected
    assert model.parameter_count()<=125_000_000
    assert len(model.blocks)==configs()[size].layers
    assert (sum(p.numel() for n,p in model.named_parameters() if '.write_gate.' in n)
            == configs()[size].layers*configs()[size].role_count*(configs()[size].width+1))

@pytest.mark.parametrize('size',["1m","10m"])
def test_untrained_exits_are_source_executable(size):
    cfg=configs()[size]
    model=CSL125R2(cfg).eval()
    with torch.no_grad():
        tokens=torch.randint(0,cfg.vocab_size,(1,9))
        outputs,_=model(tokens,all_rungs=True,emit_token_logits=False)
    assert list(outputs)==list(cfg.rungs)
    assert all(torch.isfinite(outputs[r]['tool_logits']).all() for r in cfg.rungs)
