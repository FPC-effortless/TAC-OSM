"""Encoder/actor boundaries and precision for the staged semantic experiment."""
import copy
import json
from pathlib import Path
import sys
import pytest
import torch
from tac_osm.development.stable_semantic import StableSemanticAgent
from test_semantic_bottleneck import payload
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from data_stable_semantic_dev067 import legacy_training, prepare, take, corpus
from run_stable_semantic_dev067 import cached_posteriors, actor_loss, head_parameters
from run_learned_stack_dev055 import digest


def test_coordinate_transform_cannot_change_the_learned_encoder_belief():
    args=payload(); probability=StableSemanticAgent(seed=67).double()
    logarithmic=StableSemanticAgent(seed=67,coordinate='log_probability').double()
    a,b=probability(*args),logarithmic(*args)
    for name in ('posterior','latent','health_posterior','relation_posterior'):
        torch.testing.assert_close(a[name],b[name],atol=0,rtol=0)
    assert not torch.equal(a['logits'],b['logits'])


def test_log_coordinates_handle_exact_zero_public_probabilities_without_a_rule():
    args=payload(); model=StableSemanticAgent(seed=67,coordinate='log_probability').double()
    posterior=torch.zeros(2,12,dtype=torch.float64);posterior[0,0]=1;posterior[1,9]=1
    output=model.act_from_belief(posterior,args[2],args[3],args[4])
    assert all(torch.isfinite(v).all() for v in output.values())
    assert output['logits'].shape==(2,5)


def test_cached_actor_training_cannot_modify_or_differentiate_the_encoder():
    torch.set_num_threads(1);encoder=StableSemanticAgent(seed=8).double().eval()
    gate=copy.deepcopy(encoder.gate).float()
    records=corpus(16670901,4);data=prepare(records,gate)
    q=cached_posteriors(encoder,data);assert not q.requires_grad
    before={k:v.clone() for k,v in encoder.state_dict().items()}
    actor=StableSemanticAgent(seed=8,coordinate='log_probability').double()
    x=take(data,torch.arange(4),torch.float64)
    optimizer=torch.optim.AdamW(head_parameters(actor),lr=.003)
    loss,_=actor_loss(actor,x,q);optimizer.zero_grad();loss.backward();optimizer.step()
    assert all(p.grad is None for p in encoder.parameters())
    assert all(torch.equal(v,encoder.state_dict()[k]) for k,v in before.items())
    assert any(not torch.equal(before[k],v) for k,v in actor.state_dict().items() if k.startswith('policy.'))


def test_prepared_batches_preserve_float64_public_posterior_targets():
    records=corpus(16670902,4);m=StableSemanticAgent(seed=9)
    prepared=prepare(records,m.gate);x=take(prepared,torch.arange(4),torch.float32)
    assert x['events'].dtype==torch.float32 and x['posterior'].dtype==torch.float64
    torch.testing.assert_close(x['posterior'],torch.stack([r['posterior'] for r in records]),atol=0,rtol=0)


def test_legacy_import_rejects_a_heldout_world_even_with_valid_file_hash(tmp_path):
    folder=tmp_path/'group-96501';folder.mkdir()
    torch.save([dict(world_seed=6651003)],folder/'dagger.pt')
    (tmp_path/'summary.json').write_text(json.dumps(dict(status='COMPLETE')))
    (tmp_path/'manifest.json').write_text(json.dumps({'group-96501/dagger.pt':digest(folder/'dagger.pt')}))
    with pytest.raises(AssertionError,match='TRAINING'):
        legacy_training(tmp_path,1,1)
