"""LANGFAST095 locally optimized, no-fit replica of original LANGMEM086.

This is an isolated source-derived reference for profiled performance analysis.
Compared to archived original source blob 1e072193e6ba89a46d8bd33e65dd207843558402,
only alloc-free state schema validation and nonpersistent constant identity buffers
are changed. All learned weights and forward recurrence equations are intact.
Full source+state original checkpoint replay is required before use. Not promoted.
"""
"""LANGMEM086 compact TAC: causal learned query, clipped fast and weak slow memory.

New ~100k complete model, trained from scratch. No old checkpoint weights,
external history/address index or target feedback is accepted by inference.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F

@dataclass(frozen=True)
class StreamState086:
    owners:tuple[str,...]
    clock:int
    tensors:dict[str,torch.Tensor]
    def tensor_bytes(self):return sum(x.numel()*x.element_size() for x in self.tensors.values())
    def detached(self):return StreamState086(self.owners,self.clock,{k:v.detach() for k,v in self.tensors.items()})

class OptimizedCompactTAC095(nn.Module):
    def __init__(self,ffn_width=94):
        super().__init__();self.ffn_width=ffn_width
        self.embedding=nn.Embedding(256,32)
        self.position=nn.Linear(7,32,bias=False)
        self.context_lag=nn.Parameter(torch.full((3,32),.1))
        self.writer=nn.Linear(32,8)
        self.query=nn.Linear(64,8)
        self.value_residual=nn.Linear(32,8)
        self.fast_gate=nn.Linear(32,2)
        self.slow_gate=nn.Linear(32,1)
        self.slow_decay=nn.Parameter(torch.tensor(0.))
        self.read_mix=nn.Linear(64,2)
        self.aligned_value=nn.Linear(32,8)
        self.aligned_gate=nn.Linear(32,6)
        self.rotation=nn.Parameter(torch.stack((torch.eye(8),torch.roll(torch.eye(8),1,1))))
        self.wave_decay=nn.Linear(32,4)
        self.wave_value=nn.Linear(32,8)
        self.wave_angles=nn.Parameter(torch.linspace(.03,.3,4))
        self.basis=nn.Linear(32,48)
        self.concept_gate=nn.Linear(32,32)
        self.concept_value=nn.Linear(32,32)
        self.fast=nn.Linear(32+48+8+8,48)
        self.slow=nn.Linear(48+48+8,48)
        self.evidence=nn.Linear(32+48+48+8+8,32)
        self.concept_q=nn.Linear(8,8,bias=False)
        self.concept_k=nn.Linear(8,8,bias=False)
        self.norm=nn.LayerNorm(160)
        self.upper_input=nn.Linear(160,32)
        self.upper_forget=nn.Linear(32,32)
        self.upper_offer=nn.Linear(32,32)
        self.upper_read=nn.Linear(64,32)
        self.upper_out=nn.Linear(32,160)
        self.upper_gain=nn.Parameter(torch.tensor(-2.))
        self.ffn_up=nn.Linear(160,ffn_width)
        self.ffn_gate=nn.Linear(160,ffn_width)
        self.ffn_out=nn.Linear(ffn_width,256)
        self.concept_out=nn.Linear(8,256,bias=False)
        bits=((torch.arange(256)[:,None]>>torch.arange(8)[None,:])&1).float()
        self.register_buffer('byte_codes',(2*bits-1)/math.sqrt(8),persistent=True)
        # Reused exact identity matrices. Nonpersistent buffers: no trained weights,
        # optimizer state or legacy checkpoint names changed.
        self.register_buffer('eye4_langfast095',torch.eye(4),persistent=False)
        self.register_buffer('eye8_langfast095',torch.eye(8),persistent=False)

    def counts(self):
        return {'trainable':sum(p.numel() for p in self.parameters() if p.requires_grad),
                'deployed':sum(p.numel() for p in self.parameters()),'fixed_buffer_bytes':self.byte_codes.numel()*self.byte_codes.element_size()}

    def start(self,owners):
        if not owners or len(set(owners))!=len(owners) or any(type(o) is not str or not o for o in owners):raise ValueError('explicit unique owners')
        b=len(owners);w=self.embedding.weight
        shapes={'fast_memory':(8,8),'slow_memory':(8,8),'aligned':(2,8,8),
                'wave':(4,2),'basis':(48,),'concepts':(4,8),'lexical':(3,32),
                'previous_writer':(8,),'upper':(32,)}
        return StreamState086(tuple(owners),0,{k:w.new_zeros((b,*s)) for k,s in shapes.items()})

    @staticmethod
    def positions(clock,length,batch,dtype,device):
        t=torch.arange(clock,clock+length,device=device,dtype=dtype)
        z=torch.stack((torch.log1p(t)/8,torch.sin(t/16),torch.cos(t/16),torch.sin(t/64),torch.cos(t/64),torch.sin(t/256),torch.cos(t/256)),-1)
        return z.unsqueeze(0).expand(batch,-1,-1)

    def validate(self,tokens,state,owners):
        if tuple(owners)!=state.owners or state.clock<0:raise ValueError('source owner/clock')
        if tokens.dtype!=torch.long or tokens.ndim!=2 or tokens.shape[0]!=len(owners) or tokens.shape[1]<1:raise ValueError('observed [B,T] bytes')
        if tokens.min()<0 or tokens.max()>255:raise ValueError('byte vocabulary')
        # LANGFAST095: identical schema and finiteness checks, without allocating
        # nine all-zero reference tensors during every single-byte decode.
        expected_shapes={'fast_memory':(8,8),'slow_memory':(8,8),'aligned':(2,8,8),
            'wave':(4,2),'basis':(48,),'concepts':(4,8),'lexical':(3,32),
            'previous_writer':(8,),'upper':(32,)}
        if set(state.tensors)!=set(expected_shapes) or any(
            v.shape!=(len(owners),*expected_shapes[k]) or not torch.isfinite(v).all()
            for k,v in state.tensors.items()
        ):raise ValueError('source state shape/finite')

    def forward(self,tokens,*,state=None,owners=None,lesion=None,return_trace=False):
        if owners is None:owners=tuple(f'train-{i}' for i in range(tokens.shape[0]))
        if state is None:state=self.start(owners)
        self.validate(tokens,state,owners)
        if lesion not in (None,'fast_zero','slow_zero','both_zero','permuted_read'):raise ValueError('registered lesion')
        batch,length=tokens.shape;s=state.tensors
        embed=self.embedding(tokens);lex=embed.tanh()
        e=(embed+self.position(self.positions(state.clock,length,batch,embed.dtype,embed.device))).tanh()
        values=F.normalize(self.byte_codes[tokens]+.1*self.value_residual(e).tanh(),dim=-1,eps=1e-6)
        fg=self.fast_gate(e).sigmoid();sg=.01*self.slow_gate(e).sigmoid()
        basis=self.basis(e).tanh();cg=.05*self.concept_gate(e).sigmoid().view(batch,length,4,8)
        cv=self.concept_value(e).tanh().view(batch,length,4,8)
        ag=self.aligned_gate(e).sigmoid().view(batch,length,2,3);av=self.aligned_value(e).tanh()
        wd=.97+.029*self.wave_decay(e).sigmoid();wv=.025*self.wave_value(e).tanh().view(batch,length,4,2)
        angle=self.wave_angles;c=angle.cos();sn=angle.sin()
        rotation=torch.stack((torch.stack((c,-sn),-1),torch.stack((sn,c),-1)),-2)
        rho_s=.995+.004*self.slow_decay.sigmoid()
        zs=[];pools=[];traces=[]
        eye4=self.eye4_langfast095.to(dtype=e.dtype,device=e.device)
        eye8=self.eye8_langfast095.to(dtype=e.dtype,device=e.device)
        for j in range(length):
            t=state.clock+j
            local=lex[:,j]
            for lag in range(3):local=local+self.context_lag[lag].tanh()*s['lexical'][:,lag]
            local=local.tanh();writer=F.normalize(self.writer(local),dim=-1,eps=1e-6)
            query_input=torch.cat((local,s['upper']),-1)
            query=F.normalize(self.query(query_input),dim=-1,eps=1e-6)
            mix=self.read_mix(query_input).softmax(-1)
            old_writer=s['previous_writer'];v=values[:,j]
            Fm=s['fast_memory'];Sm=s['slow_memory']
            if t>0:
                base=(.97+.02*fg[:,j,0,None,None])*Fm
                error=v-(base@old_writer.unsqueeze(-1)).squeeze(-1)
                eta=torch.clamp(.35/(torch.linalg.vector_norm(error,dim=-1)+1e-9),max=1.)*fg[:,j,1]
                Fm=base+eta[:,None,None]*error[:,:,None]*old_writer[:,None,:]
                slow_base=rho_s*Sm
                slow_error=v-(slow_base@old_writer.unsqueeze(-1)).squeeze(-1)
                Sm=slow_base+sg[:,j,:,None]*slow_error[:,:,None]*old_writer[:,None,:]
            read_F=torch.zeros_like(Fm) if lesion in ('fast_zero','both_zero') else Fm
            read_S=torch.zeros_like(Sm) if lesion in ('slow_zero','both_zero') else Sm
            if lesion=='permuted_read':read_F=read_F.roll(1,dims=1);read_S=read_S.roll(1,dims=1)
            rF=(read_F@query.unsqueeze(-1)).squeeze(-1);rS=(read_S@query.unsqueeze(-1)).squeeze(-1)
            native=mix[:,0,None]*rF+mix[:,1,None]*rS
            # Original-style time-aligned two-head rectangular associative state.
            phase=torch.tensor([2*math.pi*(t%p)/p for p in (7,13,29,59)],dtype=e.dtype,device=e.device)
            ordinal=torch.stack((phase.cos(),phase.sin()),-1).flatten().expand(batch,-1)
            ak=F.normalize(torch.einsum('bi,hij->bhj',ordinal,self.rotation),dim=-1,eps=1e-6)
            ar=(s['aligned']@ak[:,:,:,None]).squeeze(-1).mean(1)
            p=ag[:,j,:,0,None,None]*(eye8-ag[:,j,:,1,None,None]*ak[:,:,:,None]*ak[:,:,None,:])
            aq=ag[:,j,:,2,None,None]*av[:,j,None,:,None]*ak[:,:,None,:]
            aligned=s['aligned']@p+aq
            oldwave=s['wave'].flatten(1);ctx=s['basis']/max(1,t)
            fast=self.fast(torch.cat((e[:,j],ctx,oldwave,native),-1)).tanh()
            slow=self.slow(torch.cat((fast,ctx,native),-1)).tanh()
            G=self.evidence(torch.cat((e[:,j],fast,slow,ar,oldwave),-1)).tanh().view(batch,4,8)
            q=self.concept_q(G);k=self.concept_k(G);weights=(q@k.transpose(-1,-2)/math.sqrt(8)).softmax(-1)
            adjacency=.5*(weights+weights.transpose(-1,-2))*(1-eye4)
            L=torch.diag_embed(adjacency.sum(-1))-adjacency
            C=torch.linalg.solve(1.25*eye4+.25*L,G+.25*s['concepts'])
            pooled=(C*G.mean(-1).softmax(-1)[:,:,None]).sum(1)
            z=self.norm(torch.cat((fast,slow,native,ar,oldwave,pooled,e[:,j]),-1))
            u=self.upper_input(z).tanh();gate=self.upper_read(torch.cat((u,s['upper']),-1)).sigmoid()
            refined=z+self.upper_gain.sigmoid()*self.upper_out((u+gate*s['upper']).tanh())
            alpha=.995*self.upper_forget(u).sigmoid()
            upper=alpha*s['upper']+(1-alpha)*self.upper_offer(u).tanh()
            wave=(s['wave'][:,:,None,:]@(wd[:,j,:,None,None]*rotation)).squeeze(-2)+wv[:,j]
            s={'fast_memory':Fm,'slow_memory':Sm,'aligned':aligned,'wave':wave,
               'basis':s['basis']+basis[:,j],'concepts':(1-cg[:,j])*s['concepts']+cg[:,j]*cv[:,j],
               'lexical':torch.stack((lex[:,j],s['lexical'][:,0],s['lexical'][:,1]),1),
               'previous_writer':writer,'upper':upper}
            zs.append(refined);pools.append(pooled)
            if return_trace:traces.append({'query':query,'writer':old_writer,'value':v,'read_fast':rF,'read_slow':rS,'mix':mix})
        z=torch.stack(zs,1);pooled=torch.stack(pools,1)
        logits=self.ffn_out(self.ffn_up(z)*F.silu(self.ffn_gate(z)))+self.concept_out(pooled)
        result=(logits,StreamState086(tuple(owners),state.clock+length,s))
        return (*result,traces) if return_trace else result
