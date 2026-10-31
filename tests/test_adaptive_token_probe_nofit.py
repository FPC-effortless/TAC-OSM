"""Actual-source dense BYTE adapter checks. No optimizer or training."""
import pytest
import torch
from tac_osm.adaptive_token_controls import DenseByteGRU,TokenComparisonArm
from tac_osm.adaptive_token_probe import dense_byte_adapter
from tac_osm.research_probe import probe_memory,probe_dependencies,probe_entity_isolation


def sample():
    return {'previous_ids':torch.tensor([[256,97,98,99,100],[256,120,121,122,42]])}


def test_every_streamed_output_matches_actual_dense_source():
    torch.manual_seed(42)
    model=DenseByteGRU().eval()
    adapter=dense_byte_adapter()
    data=sample()
    adapter.validate(model,data)
    state=adapter.start(model,data)
    with torch.no_grad():
        expected=model(data['previous_ids'])
        for t in range(adapter.steps(data)):
            state=adapter.advance(model,state,data,t)
            actual=adapter.forecast(model,state,data)['next_byte_logits']
            assert torch.equal(actual,expected[:,t])


def test_actual_memory_gradient_entity_and_patch_probes():
    model=DenseByteGRU().eval()
    adapter=dense_byte_adapter()
    data=sample()
    memory=probe_memory(model,data,adapter)
    assert memory['state_components']==['current_byte','hidden']
    assert memory['query_does_not_write_state']
    assert memory['candidate_permutation']['status']=='INCONCLUSIVE_NOT_DECLARED'
    deps=probe_dependencies(model,data,adapter)
    assert all(v['connected'] and v['finite'] for v in deps['heads']['next_byte_logits'].values())
    assert probe_entity_isolation(model,data,adapter)['status']=='PASS'
    state=adapter.start(model,data)
    changed=adapter.patch_state(state,'hidden',state['hidden']+1)
    assert not torch.equal(changed['hidden'],state['hidden'])


def test_adapter_never_claims_adaptive_model_or_future_target_coverage():
    adapter=dense_byte_adapter()
    with pytest.raises(ValueError):
        adapter.validate(TokenComparisonArm('adaptive_codebook',seed=1).backbone,sample())
    with pytest.raises(ValueError):
        adapter.validate(DenseByteGRU(),{**sample(),'targets':torch.zeros(1)})
