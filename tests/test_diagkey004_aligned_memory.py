"""No fitting: independent reference, memory semantics and guarded registration."""
import copy
import json
from pathlib import Path
import pytest
import torch
from torch.nn import functional as F
from tac_osm.aligned_associative_tac import AlignedAssociativeTAC,gated_write
from scripts.diagkey004_aligned_recall import COUNTS,model_for,batch
from scripts.validate_research_parity import check_manifest
from tests.references.fla_gated_delta_reference import naive_recurrent_gated_delta_rule


def test_actual_whole_models_match_absolute_and_relative_budget_without_padding():
    for a,n in COUNTS.items():
        m=model_for(a)
        assert sum(p.numel() for p in m.parameters())==sum(p.numel() for p in m.parameters() if p.requires_grad)==n
    assert (max(COUNTS.values())-min(COUNTS.values()))/max(COUNTS.values())<.01


def test_cell_matches_pinned_independent_fla_reference_before_radius_projection():
    torch.manual_seed(1);b,t,d=3,7,8
    k=F.normalize(torch.randn(b,t,1,d),dim=-1);q=F.normalize(torch.randn_like(k),dim=-1)
    v=.1*torch.randn(b,t,1,d);beta=torch.rand(b,t,1);g=-.1*torch.rand(b,t,1)
    expected,final=naive_recurrent_gated_delta_rule(q,k,v,beta,g,scale=1.,output_final_state=True)
    state=torch.zeros(b,d,d);out=[]
    for j in range(t):
        state=gated_write(state,k[:,j,0],v[:,j,0],g[:,j].exp(),beta[:,j],torch.ones(b,1,dtype=torch.bool),radius=None)
        out.append(torch.bmm(state,q[:,j,0].unsqueeze(-1)).squeeze(-1))
    torch.testing.assert_close(torch.stack(out,1),expected[:,:,0],atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(state,final[:,0].transpose(-1,-2),atol=1e-6,rtol=1e-6)


def test_four_orthogonal_addresses_support_32_value_codes_in_8x8_oracle():
    torch.manual_seed(4);codes=F.normalize(torch.randn(32,8),dim=-1);keys=torch.eye(8)[:4]
    memory=torch.zeros(1,8,8);labels=[3,17,22,31]
    for key,label in zip(keys,labels):
        memory=gated_write(memory,key[None],codes[label:label+1],torch.ones(1,1),torch.ones(1,1),torch.ones(1,1,dtype=torch.bool))
    read=memory[0]@keys.T
    assert (read.T@codes.T).argmax(-1).tolist()==labels


def test_query_events_never_modify_or_decay_the_memory_even_when_norm_is_large():
    m=torch.randn(2,8,8)*10
    actual=gated_write(m,F.normalize(torch.randn(2,8),dim=-1),torch.randn(2,8),torch.zeros(2,1),torch.ones(2,1),torch.zeros(2,1,dtype=torch.bool))
    torch.testing.assert_close(actual,m,atol=0,rtol=0)


def test_shared_address_does_not_depend_on_value_and_all_learned_modules_receive_gradients():
    m=AlignedAssociativeTAC();t,k,v,y=batch(75001,0,'SOFTWARE_FIXTURE',4)
    logits,trace=m(t,k,v,return_trace=True)
    for i in range(len(t)):
        slot=(k[i,:4]==k[i,-1]).nonzero().item()
        torch.testing.assert_close(trace['addresses'][i,slot],trace['addresses'][i,-1],atol=0,rtol=0)
    torch.testing.assert_close(trace['memory'][:,-1],trace['memory'][:,-2],atol=0,rtol=0)
    F.cross_entropy(logits,y).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    assert torch.all(trace['memory'].square().sum((-1,-2)).sqrt()<=4.00001)


def test_narrow_registered_exception_cannot_admit_more_delay_budget_or_other_fits():
    m=json.loads(Path('research/mathematical-engineering/diagkey004/FAIR_COMPARISON.json').read_text())
    assert check_manifest(m)['real_world_capability_evidence'] is False
    for edit in ('count','budget','delay','owner','gate'):
        wrong=copy.deepcopy(m)
        if edit=='count':wrong['models'][0]['trainable_parameters']=40000
        if edit=='budget':wrong['training_budget']['steps']=5000
        if edit=='delay':wrong['training_budget']['trained_distractor_length']=1
        if edit=='owner':wrong['explicit_owner_requested_single_exception']=False
        if edit=='gate':wrong['all_five_seed_threshold']=.8
        with pytest.raises(ValueError):check_manifest(wrong)
    with pytest.raises(ValueError):check_manifest(m,'final')
