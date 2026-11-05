"""Original-graph generic probe adapter; synthetic diagnostic only."""
import torch
import pytest
from tac_osm.csl125_r1 import CSL125R1,CSL125R1Config
from tac_osm.csl125_r1_probe_adapter import csl125_r1_probe_adapter as adapter
from tac_osm.research_probe import probe_memory,probe_dependencies,probe_entity_isolation


def setup():
    torch.manual_seed(321)
    c=CSL125R1Config(vocab_size=256,width=24,layers=4,ff_width=48,
         role_count=3,state_width=8,rungs=(2,4),attention_layers=(2,4),
         query_heads=4,kv_heads=2,attention_window=7)
    m=CSL125R1(c).eval()
    sample={'prefix':torch.randint(0,256,(2,3)),
            'query_token':torch.randint(0,256,(2,1)),
            'stream_ids':('a','b'),'depth':4}
    return m,sample


def test_adapter_memory_and_isolation_on_actual_r1_computation():
    torch.set_num_threads(1)
    m,s=setup()
    result=probe_memory(m,s,adapter)
    assert result['query_does_not_write_state']
    assert 'layer0.fast_mass' in result['state_components']
    assert 'kv3.k' in result['state_components']
    assert result['candidate_permutation']['status']=='INCONCLUSIVE_NOT_DECLARED'
    isolated=probe_entity_isolation(m,s,adapter)
    assert isolated['status']=='PASS'


def test_adapter_gradients_include_state_and_exclude_missing_candidate_axis():
    torch.set_num_threads(1)
    m,s=setup()
    report=probe_dependencies(m,s,adapter)
    assert report['status']=='MEASURED'
    assert report['heads']['tool_logits']['layer0.fast']['connected']
    assert report['heads']['tool_logits']['layer0.fast_mass']['connected']
    assert report['heads']['tool_logits']['kv3.k']['connected']


def test_adapter_rejects_future_target_key():
    m,s=setup()
    s['target']=torch.zeros((2,))
    with pytest.raises(ValueError,match='ONLY'):
        adapter.validate(m,s)
