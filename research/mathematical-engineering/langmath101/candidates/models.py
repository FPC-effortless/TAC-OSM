"""100K active-byte candidate family; no training or capability assertions.

Native098 is copied byte-for-byte into native098.py and checked at import.
New model geometry is frozen by the factory below, not fitted to DEV results.
"""
from dataclasses import dataclass, replace
import hashlib
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

try:
    from .native098 import ContextualAssociativeTAC098, affine_scan
except ImportError:
    from native098 import ContextualAssociativeTAC098, affine_scan

NATIVE_SHA = 'e59c0aec80a39d84a6f01a7ca662d0c8baf3e2c2d33901057f017cb23c999498'
assert hashlib.sha256(Path(__file__).with_name('native098.py').read_bytes()).hexdigest() == NATIVE_SHA
TARGET = 100370


@dataclass(frozen=True)
class State:
    owners: tuple[str, ...]
    clock: int
    tensors: dict[str, torch.Tensor]

    def tensor_bytes(self):
        return sum(v.numel()*v.element_size() for v in self.tensors.values())


def owners_for(tokens, owners):
    o = tuple(f'train-{i}' for i in range(tokens.shape[0])) if owners is None else tuple(owners)
    if (not o or len(o) != tokens.shape[0] or len(set(o)) != len(o)
            or any(type(x) is not str or not x for x in o)):
        raise ValueError('explicit unique stream ownership required')
    return o


def validate_input(model, tokens):
    if next(model.parameters()).dtype not in (torch.float32,torch.float64):
        raise ValueError('FP32/FP64 only until mixed precision is qualified')
    if (not isinstance(tokens, torch.Tensor) or tokens.dtype != torch.long
            or tokens.ndim != 2 or min(tokens.shape) < 1
            or bool((tokens < 0).any()) or bool((tokens > 255).any())
            or tokens.device != next(model.parameters()).device):
        raise ValueError('nonempty same-device Long[B,T] observed bytes required')


def validate_state(model, state, owners, shapes):
    if (state.owners != owners or type(state.clock) is not int or state.clock < 0
            or set(state.tensors) != set(shapes)):
        raise ValueError('state owner/clock/schema mismatch')
    dtype = next(model.parameters()).dtype
    for name, shape in shapes.items():
        v = state.tensors[name]
        if (v.shape != (len(owners), *shape) or v.device != next(model.parameters()).device
                or v.dtype != dtype or not bool(torch.isfinite(v).all())):
            raise ValueError('state shape/device/dtype/finite mismatch: '+name)


class Counted(nn.Module):
    def counts(self):
        return {'trainable': sum(p.numel() for p in self.parameters() if p.requires_grad),
                'deployed': sum(p.numel() for p in self.parameters()),
                'fixed_buffer_bytes': sum(p.numel()*p.element_size() for p in self.buffers())}

    def _state(self, owners, shapes):
        owners=tuple(owners)
        if not owners or len(set(owners)) != len(owners) or any(type(o) is not str or not o for o in owners):
            raise ValueError('explicit unique nonempty stream owners required')
        p=next(self.parameters())
        return State(owners,0,{n:p.new_zeros((len(owners),*s)) for n,s in shapes.items()})


class BudgetHead(nn.Module):
    """Real gated decoder capacity allocation; no unused or frozen padding."""
    def __init__(self, dimension, available):
        super().__init__()
        # head count = 2d + 2(d+1)w + 256(w+1)
        fixed=2*dimension+256; increment=2*(dimension+1)+256
        width=max(1,round((available-fixed)/increment))
        self.width=width
        self.norm=nn.LayerNorm(dimension)
        self.up=nn.Linear(dimension,width)
        self.gate=nn.Linear(dimension,width)
        self.out=nn.Linear(width,256)

    def forward(self,x):
        z=self.norm(x)
        return self.out(self.up(z)*F.silu(self.gate(z)))


class StatelessCandidate(Counted):
    """No mutable state across calls; bounded current-input window only."""
    def __init__(self, window=1, embed=32):
        super().__init__();self.window=window;self.embed=embed
        self.embedding=nn.Embedding(256,embed)
        base=self.counts()['trainable'];self.head=BudgetHead(window*embed,TARGET-base)

    def start(self,owners):return self._state(owners,{})

    def forward(self,tokens,*,state=None,owners=None,parallel=True,**kwargs):
        if kwargs:raise ValueError('unsupported input fields: '+str(sorted(kwargs)))
        validate_input(self,tokens);o=owners_for(tokens,owners)
        if state is not None:
            validate_state(self,state,o,{})
            if state.clock != 0:raise ValueError('stateless model cannot accept carried state')
        e=self.embedding(tokens).tanh();b,t,_=e.shape
        pad=e.new_zeros((b,self.window-1,self.embed));h=torch.cat((pad,e),1)
        x=torch.cat([h[:,self.window-1-j:self.window-1-j+t] for j in range(self.window)],-1)
        return self.head(x),State(o,0,{})


class AssociativeBlock(nn.Module):
    """Residual contextual normalized bank block, exact source-like bank law."""
    def __init__(self,d=32,c=16,k=4,v=8,level=None):
        super().__init__();self.d=d;self.c=c;self.k=k;self.v=v;self.level=level
        self.context_input=nn.Linear(d,c)
        horizons=torch.logspace(math.log10(8.),math.log10(4096.),c)
        if level is not None:horizons=horizons*(4.**level)
        self.raw_decay=nn.Parameter(torch.log(torch.expm1(1./horizons)))
        self.address_value=nn.Linear(d+c,4*k+2*v+2)
        self.compare=nn.Linear(d+c+3*v+2,v)
        self.norm=nn.LayerNorm(d+c+3*v+2)
        self.out=nn.Linear(d+c+3*v+2,d)
        self.update=nn.Linear(d,1) if level is not None else None
        if self.update is not None:
            nn.init.zeros_(self.update.weight);nn.init.constant_(self.update.bias,-level)

    def shapes(self):
        return {'context':(self.c,), 'key':(2,self.k),
                'matrix':(2,self.v,self.k),'mass':(2,self.k)}

    def forward(self,h,s,clock,parallel):
        b,t,_=h.shape;c,k,v=self.c,self.k,self.v
        rate=self.update(h).sigmoid() if self.update is not None else h.new_ones((b,t,1))
        decay=torch.sigmoid(-self.raw_decay)[None,None]
        a=1-rate*(1-decay)
        context=affine_scan(a,(1-a)*self.context_input(h).tanh(),s['context'],parallel=parallel)
        past=torch.cat((s['context'][:,None],context[:,:-1]),1)
        raw=self.address_value(torch.cat((h,past),-1))
        key=raw[...,:2*k].reshape(b,t,2,k).softmax(-1)
        query=raw[...,2*k:4*k].reshape(b,t,2,k).softmax(-1)
        value=raw[...,4*k:4*k+2*v].reshape(b,t,2,v).tanh()
        low=h.new_tensor([.97,.995]);high=h.new_tensor([.995,.99995])
        alpha=low+(high-low)*raw[...,-2:].sigmoid()
        alpha=1-rate*(1-alpha)
        if clock==0:alpha=torch.cat((torch.ones_like(alpha[:,:1]),alpha[:,1:]),1)
        previous=torch.cat((s['key'][:,None],key[:,:-1]),1)
        offer=torch.cat(((value[...,None]*previous[...,None,:]).flatten(-2),previous),-1)
        initial=torch.cat((s['matrix'].flatten(-2),s['mass']),-1)
        bank=affine_scan(alpha[...,None],(1-alpha[...,None])*offer,initial,parallel=parallel)
        matrix=bank[...,:v*k].reshape(b,t,2,v,k);mass=bank[...,v*k:]
        evidence=(mass*query).sum(-1)
        read=(matrix*query[...,None,:]).sum(-1)/(evidence[...,None]+1e-4)
        fast,slow=read.unbind(-2)
        gate=self.compare(torch.cat((h,past,fast,slow,(fast-slow).abs(),evidence),-1)).sigmoid()
        mixed=gate*fast+(1-gate)*slow
        z=torch.cat((h,past,fast,slow,mixed,evidence),-1)
        out=h+self.out(self.norm(z)).tanh()
        return out,{'context':context[:,-1],'key':key[:,-1],
                    'matrix':matrix[:,-1],'mass':mass[:,-1]}


class RecurrentBlock(nn.Module):
    """Non-associative temporal control with genuine gated FFN capacity."""
    def __init__(self,d=32,c=16,width=36,level=None):
        super().__init__();self.d=d;self.c=c;self.level=level
        self.context_input=nn.Linear(d,c)
        horizons=torch.logspace(math.log10(8.),math.log10(4096.),c)
        if level is not None:horizons=horizons*(4.**level)
        self.raw_decay=nn.Parameter(torch.log(torch.expm1(1./horizons)))
        self.norm=nn.LayerNorm(d+c);self.up=nn.Linear(d+c,width)
        self.gate=nn.Linear(d+c,width);self.out=nn.Linear(width,d)
        self.update=nn.Linear(d,1) if level is not None else None
        if self.update is not None:
            nn.init.zeros_(self.update.weight);nn.init.constant_(self.update.bias,-level)

    def shapes(self):return {'context':(self.c,)}

    def forward(self,h,s,clock,parallel):
        rate=self.update(h).sigmoid() if self.update is not None else h.new_ones((*h.shape[:2],1))
        decay=torch.sigmoid(-self.raw_decay)[None,None];a=1-rate*(1-decay)
        context=affine_scan(a,(1-a)*self.context_input(h).tanh(),s['context'],parallel=parallel)
        past=torch.cat((s['context'][:,None],context[:,:-1]),1)
        z=self.norm(torch.cat((h,past),-1))
        out=h+self.out(self.up(z)*F.silu(self.gate(z)))
        return out,{'context':context[:,-1]}


class StackedCandidate(Counted):
    def __init__(self,depth=1,associative=True,hierarchy=False,d=32):
        super().__init__();self.depth=depth;self.associative=associative;self.hierarchy=hierarchy;self.d=d
        self.embedding=nn.Embedding(256,d);self.local_lag=nn.Parameter(torch.full((3,d),.1))
        block=AssociativeBlock if associative else RecurrentBlock
        self.blocks=nn.ModuleList([block(d=d,level=i if hierarchy else None) for i in range(depth)])
        self.upward=nn.ModuleList([nn.Linear(d,d) for _ in range(depth-1)]) if hierarchy else None
        self.downward=nn.ModuleList([nn.Linear(d,d) for _ in range(depth-1)]) if hierarchy else None
        base=self.counts()['trainable'];self.head=BudgetHead(d,TARGET-base)
        self.shapes={'lexical':(3,d)}
        for i,b in enumerate(self.blocks):self.shapes.update({f'{i}.{k}':v for k,v in b.shapes().items()})

    def start(self,owners):return self._state(owners,self.shapes)

    def forward(self,tokens,*,state=None,owners=None,parallel=True,**kwargs):
        if kwargs:raise ValueError('unsupported input fields: '+str(sorted(kwargs)))
        validate_input(self,tokens);o=owners_for(tokens,owners)
        state=self.start(o) if state is None else state;validate_state(self,state,o,self.shapes)
        e=self.embedding(tokens).tanh();t=tokens.shape[1]
        history=torch.cat((state.tensors['lexical'],e),1);h=e
        for lag in range(3):h=h+self.local_lag[lag].tanh()*history[:,2-lag:2-lag+t]
        h=h.tanh();final={'lexical':history[:,-3:]};levels=[]
        for i,b in enumerate(self.blocks):
            if self.hierarchy and i:h=self.upward[i-1](h).tanh()
            ss={k:state.tensors[f'{i}.{k}'] for k in b.shapes()}
            h,next_s=b(h,ss,state.clock,parallel);levels.append(h)
            final.update({f'{i}.{k}':v for k,v in next_s.items()})
        if self.hierarchy:
            for i in range(self.depth-2,-1,-1):h=levels[i]+self.downward[i](h).tanh()
        return self.head(h),State(o,state.clock+t,final)


def intervene(state, groups=(), *, donor=None):
    """Explicit field/layer reset, plus coupled fast/slow matrix+mass surgery.

    Native098 uses memory/mass. New blocks use <layer>.matrix/<layer>.mass.
    Keys are not silently reset with banks: request previous_key or <layer>.key
    separately. This permits attribution of stored values versus write links.
    """
    selected=set(groups);values={};bank_requests={}
    for group in list(selected):
        if group in ('fast','slow','banks') and 'memory' in state.tensors:
            bank_requests.setdefault('',set()).update((0,) if group=='fast' else (1,) if group=='slow' else (0,1))
            selected.remove(group)
        elif '.' in group:
            layer,field=group.rsplit('.',1)
            if field in ('fast','slow','banks') and layer+'.matrix' in state.tensors:
                bank_requests.setdefault(layer+'.',set()).update((0,) if field=='fast' else (1,) if field=='slow' else (0,1))
                selected.remove(group)
    for group in selected:
        if not any(n==group or n.startswith(group+'.') for n in state.tensors):
            raise ValueError('unknown explicit state group: '+group)
    if donor is not None:
        if donor.clock!=state.clock or set(donor.tensors)!=set(state.tensors):
            raise ValueError('donor clock/schema mismatch')
        if any(donor.tensors[n].shape!=v.shape for n,v in state.tensors.items()):
            raise ValueError('donor shape mismatch')
    for name,value in state.tensors.items():
        chosen=name in selected or any(name.startswith(g+'.') for g in selected)
        values[name]=(torch.zeros_like(value) if donor is None else donor.tensors[name].clone()) if chosen else value.clone()
    for prefix,banks in bank_requests.items():
        matrix=prefix+('matrix' if prefix else 'memory');mass=prefix+'mass'
        for bank in banks:
            for name in (matrix,mass):
                values[name][:,bank]=0 if donor is None else donor.tensors[name][:,bank]
    return replace(state,tensors=values)


FAMILY_NAMES=['native098','factorized098','stateless_current','stateless_window4']
FAMILY_NAMES += [f'stack_{kind}_{depth}' for kind in ('assoc','recurrent') for depth in (1,2,3,4,8)]
FAMILY_NAMES += ['hierarchy_assoc_3','hierarchy_recurrent_3']

def make_model(name):
    if name in ('native098','factorized098'):m=ContextualAssociativeTAC098()
    elif name=='stateless_current':m=StatelessCandidate(1)
    elif name=='stateless_window4':m=StatelessCandidate(4)
    elif name.startswith('stack_'):
        _,kind,depth=name.split('_');m=StackedCandidate(int(depth),kind=='assoc')
    elif name.startswith('hierarchy_'):
        _,kind,depth=name.split('_');m=StackedCandidate(int(depth),kind=='assoc',True)
    else:
        try:
            from .conventional import make_control
        except ImportError:
            from conventional import make_control
        m=make_control(name)
    n=m.counts()['trainable']
    if not 100000<=n<=100650:raise ValueError(f'active parameter parity budget failed for {name}: {n}')
    m.arm=name
    return m

def catalog():
    names=list(FAMILY_NAMES)
    try:
        from .conventional import CONTROL_NAMES
    except ImportError:
        try:from conventional import CONTROL_NAMES
        except ImportError:CONTROL_NAMES=[]
    return names+list(CONTROL_NAMES)

