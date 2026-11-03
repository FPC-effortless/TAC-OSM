"""Separate the phase address input from value input using existing native context."""
from dataclasses import dataclass
import hashlib,math,types
import torch
from torch import nn
from torch.nn import functional as F
from collect016.model import CarriedTAC,run_chunks
from collect011.memory import PhaseState,write_step
from tacopt109.reference.model import LayerState104
from tacopt109.reference.kernels import rotary_scan
from tacopt109.paired_banks import read_bank_pair

def phase_separated(m,value_input,address_input,state,mask,owners):
    m.validate(value_input,state,owners,mask)
    if address_input.shape!=value_input.shape:raise ValueError('address/value geometry differs')
    shape=(*value_input.shape[:2],m.heads,m.key_dim//2)
    theta=math.pi*(m.content(address_input).reshape(shape).tanh()+m.role(address_input).reshape(shape).tanh().roll(1,-1))
    key=torch.stack((theta.cos(),theta.sin()),-1).flatten(-2)/math.sqrt(m.key_dim//2)
    value=m.value(value_input).reshape(*value_input.shape[:2],m.heads,m.value_dim).tanh()
    probability=m.router(address_input).softmax(-1)
    hard=F.one_hot(probability.argmax(-1),m.heads).to(probability.dtype);route=hard+(probability-probability.detach())
    gain=m.admission(address_input).sigmoid()*mask[...,None]*route
    memory,p=state.memory,state.precision;reads=[]
    for k,v,g,r in zip(key.unbind(1),value.unbind(1),gain.unbind(1),route.unbind(1)):
        memory,p=write_step(memory,p,k,v,g,kind=m.kind)
        reads.append((memory@k.unsqueeze(-1)).squeeze(-1)*r[...,None])
    return torch.stack(reads,1),PhaseState(owners,state.clock+value_input.shape[1],memory.contiguous(),p.contiguous(),m.signature)

def context_block(self,x,state,phase,mask,mode,clock,*,owners):
    c=self.config;b,t,_=x.shape;z=self.memory_norm(x);projected=self._input_projections(z)
    inputs=[self._bank_inputs(getattr(self,n),n,z,getattr(state,n+'_memory'),getattr(state,n+'_mass'),mask,projected) for n in ('fast','slow')]
    fr,sr=read_bank_pair(*inputs,chunk_size=c.chunk_size,mode=mode)
    fast,fm,fc,fe=fr;slow,sm,sc,se=sr
    alpha=projected['rotary_retention'].sigmoid()[...,None]
    a=torch.where(mask[...,None,None],alpha,torch.ones_like(alpha))
    theta=math.pi*projected['rotary_angle'].tanh().reshape(b,t,c.n_heads,c.value_dim//2)
    theta=torch.where(mask[...,None,None],theta,torch.zeros_like(theta))
    offer=projected['rotary_offer'].tanh().reshape(b,t,c.n_heads,c.value_dim)
    field=rotary_scan(a,theta,offer,state.rotary,mode='parallel' if mode=='chunk' else 'recurrent')
    # Address sees only already-computed native pathways. No phase read is fed
    # back into its own address, and the current value projection keeps z.
    components=torch.cat((fast,slow,fast-slow,fast*slow,field,torch.zeros_like(field),fe[...,None],se[...,None]),-1).flatten(-2)
    context=x+self.residual_scale*self.perspective_out(components)
    address=self.memory_norm(context)
    recent,phase=phase_separated(self.phase_memory,z,address,phase,mask,owners)
    recent=recent.reshape(b,t,c.n_heads,c.value_dim)
    # Keep one final original mixer call, avoiding a changed accumulation order
    # in the residual/value branch; extra address-path work is measured.
    pieces=torch.cat((fast,slow,fast-slow,fast*slow,field,recent,fe[...,None],se[...,None]),-1).flatten(-2)
    x=x+self.residual_scale*self.perspective_out(pieces)
    zz=self.ffn_norm(x);v,g=F.linear(zz,torch.cat((self.ffn_value.weight,self.ffn_gate.weight))).split(c.ffn_dim,-1)
    x=x+self.residual_scale*self.ffn_out(v*F.silu(g))
    return x,LayerState104(fm,fc,sm,sc,field[:,-1].clone(),state.recent_key,state.recent_value),phase

class ContextPhase(CarriedTAC):
    def __init__(self):
        super().__init__('full_bptt');self.variant='collect018_context'
        self.collect_signature=hashlib.sha256((self.collect_signature+'C18context').encode()).hexdigest()
        for block in self.core.blocks:block.forward=types.MethodType(context_block,block)

@dataclass(frozen=True)
class GRUState:
    owners:tuple[str,...]
    clock:int
    hidden:torch.Tensor
    signature:str='collect018.gru125.v1'
    def detach(self):return GRUState(self.owners,self.clock,self.hidden.detach(),self.signature)
    def state_bytes(self):return self.hidden.numel()*self.hidden.element_size()

class GRUControl(nn.Module):
    def __init__(self):
        super().__init__();self.core=nn.Module();self.core.embedding=nn.Embedding(256,32)
        nn.init.normal_(self.core.embedding.weight,std=.02)
        self.core.gru=nn.GRU(32,125,batch_first=True);self.core.output_norm=nn.LayerNorm(125)
        self.core.output=nn.Linear(125,256)
    def forward(self,tokens,*,state=None,owners=None,write_mask=None,mode='chunk'):
        if tokens.dtype!=torch.long or tokens.ndim!=2 or tokens.numel()==0:raise ValueError('nonempty Long[B,T] required')
        if owners is None:
            if state is not None:raise ValueError('continuing owners required')
            owners=tuple(f'fresh-{i}' for i in range(tokens.shape[0]))
        owners=tuple(owners)
        if len(owners)!=tokens.shape[0] or len(set(owners))!=len(owners):raise ValueError('unique owner per row required')
        if write_mask is not None and not bool(write_mask.all()):raise ValueError('GRU observation freeze not implemented; INCONCLUSIVE rather than pretend freeze')
        if state is not None and (not isinstance(state,GRUState) or state.owners!=owners or state.hidden.shape!=(1,len(owners),125) or state.signature!='collect018.gru125.v1'):raise ValueError('complete compatible owner state required')
        h=None if state is None else state.hidden
        y,h=self.core.gru(self.core.embedding(tokens),h)
        z=self.core.output(self.core.output_norm(y))
        return z,GRUState(owners,(0 if state is None else state.clock)+tokens.shape[1],h)
    def balancing(self):return self.core.embedding.weight.new_zeros(()),self.core.embedding.weight.new_zeros(4,4)

def tensors(state):
    if isinstance(state,GRUState):return {'hidden':state.hidden}
    from collect011.model import named_state_tensors
    return named_state_tensors(state)
