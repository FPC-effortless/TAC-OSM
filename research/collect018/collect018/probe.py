"""Actual-source ResearchProbeAdapter for phase/context/GRU; no label input."""
import copy,dataclasses
import torch
from probe_vendor.research_probe import ResearchProbeAdapter,probe_entity_isolation,probe_dependencies
from .common import require
from .model import GRUState,tensors

def map_tree(value,fn):
    if isinstance(value,torch.Tensor):return fn(value)
    if dataclasses.is_dataclass(value):return dataclasses.replace(value,**{f.name:map_tree(getattr(value,f.name),fn) for f in dataclasses.fields(value)})
    if isinstance(value,tuple):return tuple(map_tree(x,fn) for x in value)
    return value
def views(state):return {'hidden':state.hidden.transpose(0,1)} if isinstance(state,GRUState) else {n:t for n,t in tensors(state).items() if t.numel()>0}
def adapter(model):
    def validate(m,s):
        require(set(s)=={'history','query','owners'},'labels/oracle fields not legal probe inputs')
        require(s['history'].ndim==s['query'].ndim==2 and s['history'].shape[0]==s['query'].shape[0]==len(s['owners']),'geometry differs')
    def start(m,s):
        if hasattr(m.core,'gru'):return GRUState(s['owners'],0,m.core.embedding.weight.new_zeros(1,len(s['owners']),125))
        return m.start(s['owners'])
    def advance(m,state,s,t):return m(s['history'][:,t:t+1],state=state,owners=s['owners'])[1]
    def forecast(m,state,s):return {'byte_logits':m(s['query'],state=state,owners=s['owners'])[0][:,-1]}
    def patch(state,name,new):
        if isinstance(state,GRUState):return dataclasses.replace(state,hidden=new.transpose(0,1))
        old=tensors(state)[name];return map_tree(state,lambda t:new if t is old else t.clone())
    def item(s,i):return dict(history=s['history'][i:i+1],query=s['query'][i:i+1],owners=(s['owners'][i],))
    def graph(m,state,s):
        if isinstance(state,GRUState):
            leaf=state.hidden.transpose(0,1).detach().clone().requires_grad_();new=dataclasses.replace(state,hidden=leaf.transpose(0,1));inputs={'hidden':leaf}
        else:
            original=tensors(state);inputs={n:t.detach().clone().requires_grad_() for n,t in original.items()};ids={id(t):inputs[n] for n,t in original.items()}
            new=map_tree(state,lambda t:ids.get(id(t),t.detach().clone()))
        return forecast(m,new,s),inputs
    return ResearchProbeAdapter('COLLECT018-'+('GRU' if hasattr(model.core,'gru') else getattr(model,'variant','phase')),validate,
        lambda s:s['history'].shape[1],start,advance,forecast,views,patch,lambda s:len(s['owners']),item,graph,None)

def run_source_probes(m):
    # Transient auxiliary routing traces are not persistent model state.
    if hasattr(m,"router_logits"):m.router_logits=[]
    sample=dict(history=torch.arange(12).reshape(2,6)%256,query=torch.tensor([[58,32],[58,32]]),owners=('probe-a','probe-b'))
    a=adapter(m);isolation=probe_entity_isolation(m,sample,a,tolerance=4e-5);dep=probe_dependencies(m,sample,a)
    require(isolation['status']=='PASS','actual adapter entity isolation failed')
    before={n:t.detach().clone() for n,t in m.state_dict().items()}
    with torch.no_grad():
        s=a.start(m,sample)
        for t in range(6):s=a.advance(m,s,sample,t)
        old={n:t.clone() for n,t in a.state_views(s).items()};a.forecast(m,s,sample)
        require(all(torch.equal(v,a.state_views(s)[n]) for n,v in old.items()),'forecast mutated original state')
    require(all(torch.equal(v,m.state_dict()[n]) for n,v in before.items()),'probe modified model')
    return dict(adapter=a.name,entity_isolation=isolation,dependencies=dep,forecast_state_purity='PASS',
        candidate_permutation='INCONCLUSIVE_NO_CANDIDATE_AXIS',precision_zero_patch='INCONCLUSIVE_INVALID_SPD_INTERVENTION' if not hasattr(m.core,'gru') else 'NOT_APPLICABLE',
        actual_source_forward_backward='COVERED_SEPARATELY_BY_FP32_FP64_REFERENCE_TESTS',isolation_tolerance_basis='Same 4e-5 bound used in initial source/reference and chunk tests before generic-adapter checks; tighter 1e-5 failure retained separately',independent_source_authentication='OPEN',optimizer_updates=0)
