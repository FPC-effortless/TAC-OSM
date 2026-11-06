"""Pipeline correctness, never a neural capability benchmark or fit approval."""
import copy,json
from dataclasses import replace
import pytest
from tac_osm.csl125_training_data import (
    DataContractError,ToolBPE,BPEToolGrammar,api_catalogue,
    prepare_records,render_example,query_group,split_for,digest,collate,padded_tool_loss,propose_bpe_tools,
)

def api():
    return {'name':'weather','description':'Read weather for a city',
            'arguments':{'city':{'type':'str','required':True,'description':'City name'}},'returns':{},'examples':[]}

def row(query='Weather in Jos',city='Jos'):
    return {'query':query,'answers':[{'id':0,'name':'weather','arguments':{'city':city}}],
            'tools':[api()]}

def fixtures():
    tools,_=api_catalogue([api()])
    rs,_,_,_=prepare_records([row(),row('Weather in Lagos','Lagos'),row('Weather in Èkó 😀','Èkó 😀')],tools)
    return tools,[dict(r,split='FIT') for r in rs]

@pytest.fixture(scope='module')
def fitted():
    tools,rows=fixtures();return tools,rows,ToolBPE.fit(rows,tools,512)

def test_fixed_catalogue_not_selected_by_gold_or_row_tools():
    tools,_=api_catalogue([api()]);bad=row();bad['tools']=[{'name':'invented_decoy'}]
    a,_,_,_=prepare_records([row()],tools);b,_,_,_=prepare_records([bad],tools)
    assert a[0]['catalogue_sha256']==b[0]['catalogue_sha256']==digest(tools)
    assert render_example(a[0],tools)==render_example(b[0],tools)
    changed=dict(a[0],calls=[{'name':'weather','arguments':{'city':'Lagos'}}])
    assert render_example(changed,tools)[0]==render_example(a[0],tools)[0]

def test_duplicate_group_and_conflict_quarantine():
    tools,_=api_catalogue([api()])
    records,q,dups,a=prepare_records([row(),row('  WEATHER  in Jos  ')],tools)
    assert len(records)==1 and len(dups)==1 and not q
    records,q,dups,a=prepare_records([row(),row('  WEATHER in Jos  ','Lagos')],tools)
    assert not records and len(q)==2 and not dups
    assert query_group('  WEATHER in Jos ')==query_group('Weather in Jos')
    assert split_for(query_group('Weather in Jos'))==split_for(query_group('WEATHER IN JOS'))

@pytest.mark.parametrize('mutate,reason',[
    (lambda r:r['answers'][0]['arguments'].update(city='#0'),'unresolved_result_reference'),
    (lambda r:r['answers'][0]['arguments'].update(city=None),'arguments_violate_declared_runtime_schema'),
    (lambda r:r['answers'][0]['arguments'].update(extra='x'),'arguments_violate_declared_runtime_schema'),
    (lambda r:r['answers'][0].update(name='unregistered'),'tool_not_in_supported_catalogue'),
    (lambda r:r.update(answers=r['answers']*5),'call_count_outside_runtime_contract'),
    (lambda r:r.update(query='<|csl_eos|>'),'reserved_token_literal'),
])
def test_no_silent_label_repair(mutate,reason):
    tools,_=api_catalogue([api()]);sample=row();mutate(sample);before=copy.deepcopy(sample)
    records,q,_,_=prepare_records([sample],tools)
    assert not records and reason in q[0]['reasons'] and sample==before

def test_unsupported_dict_schema_excluded_and_empty_call_valid():
    bad=api();bad['arguments']['city']['type']='Dict[str, Any]'
    good=api();good['name']='city_weather'
    tools,excluded=api_catalogue([good,bad])
    assert len(tools)==1 and excluded[0]['name']=='weather'
    no=row();no['answers']=[]
    records,q,_,a=prepare_records([no],tools)
    assert not q and a['zero_call_examples']==1 and records[0]['calls']==[]

def test_BPE_fit_rejects_DEV_and_reserved_literals(fitted):
    tools,rows,codec=fitted
    with pytest.raises(DataContractError,match='FIT_only'):
        ToolBPE.fit([dict(rows[0],split='DEV')],tools,512)
    with pytest.raises(DataContractError,match='reserved'):
        codec.encode('prefix <|csl_eos|> suffix')
    for text in ('Èkó 😀','{"name":"weather"}','e\u0301','\n\t','漢字'):
        ids=codec.encode(text)
        assert codec.backend.decode(ids,skip_special_tokens=False)==text

def test_BPE_target_boundary_and_checkpoint_reload(fitted,tmp_path):
    tools,rows,codec=fitted
    p,y=render_example(rows[0],tools);e=codec.supervised(p,y)
    assert len(e.target_mask)==len(e.ids)-1
    assert e.ids[-1]==codec.eos_id
    assert not any(e.target_mask[:e.prompt_tokens-1])
    assert all(e.target_mask[e.prompt_tokens-1:])
    assert codec.backend.decode(list(e.ids[e.prompt_tokens:-1]),skip_special_tokens=False)==y
    with pytest.raises(DataContractError,match='no_truncation'):codec.supervised(p,y,max_tokens=3)
    path=tmp_path/'tok.json';codec.save(path);loaded=ToolBPE.load(path)
    assert loaded.encode(p)==codec.encode(p)

@pytest.mark.parametrize('city',['Jos','Èkó 😀','a"b','line\nbreak'])
def test_BPE_gold_schema_roundtrip(fitted,city):
    tools,rows,codec=fitted
    target='{"function_calls":[{"name":"weather","arguments":{"city":'+json.dumps(city,ensure_ascii=False)+'}}]}'
    grammar=BPEToolGrammar(codec,tools);session=grammar.session()
    for t in codec.encode(target):
        assert t in session.allowed().tolist()
        session.accept(t)
    assert session.complete and session.validate().function_calls[0]['arguments']['city']==city
    assert session.allowed().numel()==0

@pytest.mark.parametrize('target',[
    '{"function_calls":[{"name":"not_weather","arguments":{"city":"Jos"}}]}',
    '{"function_calls":[{"name":"weather","arguments":{"city":true}}]}',
    '{"function_calls":[{"name":"weather","arguments":{"city":"Jos","extra":"x"}}]}',
])
def test_BPE_grammar_rejects_bad_contracts(fitted,target):
    tools,_,codec=fitted;session=BPEToolGrammar(codec,tools).session()
    with pytest.raises(DataContractError):
        for t in codec.encode(target):session.accept(t)
        session.validate()

def test_BPE_control_tokens_rejected_and_matchers_isolated(fitted):
    tools,_,codec=fitted;grammar=BPEToolGrammar(codec,tools);a=grammar.session();b=grammar.session()
    for control in [codec.pad_id,codec.eos_id]:
        assert control not in a.allowed().tolist()
        with pytest.raises(DataContractError):a.accept(control)
    for t in codec.encode('{"function_calls":[]}'):a.accept(t)
    assert a.complete and not b.complete
    assert a.validate().status=='abstain'

def test_right_padding_loss_gradients_and_poison_invariance(fitted):
    import torch
    from tac_osm.csl125_r3 import CSL125R3Config
    from tac_osm.csl125_tool_model import CSL125R4
    from tac_osm.csl125_memory_precision import enable_precise_memory_
    tools,rows,codec=fitted
    first=codec.supervised(*render_example(rows[0],tools))
    longer=dict(rows[1],query=rows[1]['query']+' Please use the available weather tool for this request.')
    examples=[first,codec.supervised(*render_example(longer,tools))]
    tokens,mask,lengths=collate(examples,codec.pad_id)
    assert lengths.min()<lengths.max(), 'padding poison must not be a vacuous test'
    cfg=CSL125R3Config(vocab_size=codec.vocab_size,width=32,ff_width=64,layers=4,
        role_count=4,state_width=8,rungs=(2,4),attention_layers=(2,4),query_heads=4,kv_heads=2,
        attention_window=16,correction_rank=4)
    torch.manual_seed(125);model=enable_precise_memory_(CSL125R4(cfg)).eval()
    loss=padded_tool_loss(model,tokens,mask,lengths)
    assert torch.isfinite(loss);loss.backward()
    assert model.embedding.weight.grad is not None and torch.isfinite(model.embedding.weight.grad).all()
    poisoned=tokens.clone()
    for i,n in enumerate(lengths.tolist()):poisoned[i,n:]=17
    with torch.no_grad():again=padded_tool_loss(model,poisoned,mask,lengths)
    assert torch.allclose(loss.detach(),again,atol=1e-6,rtol=1e-6)
    bad=mask.clone();bad[0,0]=True
    with pytest.raises(DataContractError):padded_tool_loss(model,tokens,bad,lengths)
    session=BPEToolGrammar(codec,tools)
    result=propose_bpe_tools(model,'Weather in Jos',session,depth=2,max_new_tokens=1)
    assert result.status in ('incomplete','abstain','proposed')
    with pytest.raises(DataContractError,match='vocabulary'):
        model.config=replace(cfg,vocab_size=cfg.vocab_size+1)
        propose_bpe_tools(model,'Weather in Jos',session,max_new_tokens=1)
