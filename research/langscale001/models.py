"""LANGSCALE001 count-frozen memory reallocation; historical graphs unchanged."""
import sys
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'research/langcap103'))
import preflight
NATIVE=preflight.load_native()

class ReallocatedTAC(NATIVE.StackedCandidate):
    """Same full forward as the pinned original; only constructor geometry changes."""
    def __init__(self, *, target, d, c, k, v, depth=8):
        NATIVE.Counted.__init__(self)
        self.depth=depth;self.associative=True;self.hierarchy=False;self.d=d
        self.embedding=nn.Embedding(256,d)
        self.local_lag=nn.Parameter(torch.full((3,d),.1))
        self.blocks=nn.ModuleList([NATIVE.AssociativeBlock(d=d,c=c,k=k,v=v) for _ in range(depth)])
        self.upward=None;self.downward=None
        base=sum(p.numel() for p in self.parameters())
        if target-base <= 2*d+256:raise ValueError('No budget for active output head')
        self.head=NATIVE.BudgetHead(d,target-base)
        self.shapes={'lexical':(3,d)}
        for i,b in enumerate(self.blocks):
            self.shapes.update({f'{i}.{name}':shape for name,shape in b.shapes().items()})
        count=sum(p.numel() for p in self.parameters())
        if abs(count-target)/target>.005:raise ValueError('Live count differs from target by >0.5%')


def configurations():
    """Frozen count-only rule: largest multiple-of-eight width with core <=90%."""
    original=__import__('json').loads((ROOT/'research/langcap105/OWNER_PROTOCOL.json').read_text())
    rows=[]
    for scale,target,c,k,v in [('small',1_000_000,32,8,16),('medium',10_000_000,64,16,32)]:
        originals=original['models'][scale]['configurations']
        orig=next(x for x in originals if x['arm']=='tac_associative_stack')
        with torch.device('meta'):
            widths=[]
            for d in range(64,orig['width']+1,8):
                block=NATIVE.AssociativeBlock(d=d,c=c,k=k,v=v)
                core=259*d+8*sum(p.numel() for p in block.parameters())
                if core<=.9*target:widths.append(d)
        d=max(widths)
        for arm,geom in [('tac_memory_scaled',dict(d=d,c=c,k=k,v=v)),
                         ('tac_head_to_state',dict(d=orig['width'],c=24 if scale=='small' else 32,k=6 if scale=='small' else 8,v=12 if scale=='small' else 16))]:
            rows.append(dict(scale=scale,target=target,arm=arm,depth=8,**geom))
        for row in originals:
            rows.append(dict(scale=scale,target=target,**{k:v for k,v in row.items() if k!='trainable_parameters'}))
    return rows


def build(row):
    if row['arm'].startswith(('tac_memory_scaled','tac_head_to_state')):
        return ReallocatedTAC(target=row['target'],**{k:row[k] for k in ('d','c','k','v','depth')})
    config={k:v for k,v in row.items() if k not in ('scale','target')}
    return preflight.build(config,row['target'],NATIVE,preflight.comparator_classes(torch))
