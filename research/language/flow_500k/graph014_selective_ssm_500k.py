"""GRAPH014 prospective 500K selective diagonal SSM, not recovered historical SSM.

Input-only selective diagonal recurrence, chronological affine parallel scan.
No attention or external retrieval. The original 499290-weight TAC is unchanged.
This is a NEW versioned comparator, not the earlier 498784-weight custom SSM.
"""
import torch
from torch import nn
from torch.nn import functional as F

class SelectiveDiagonalSSM500K(nn.Module):
    def __init__(self, vocab=128, width=160, ff_width=680):
        super().__init__()
        if (vocab,width,ff_width)!=(128,160,680):
            raise ValueError("registered model geometry is frozen")
        self.embedding=nn.Embedding(vocab,width)
        self.norm_in=nn.RMSNorm(width)
        self.x_proj=nn.Linear(width,3*width)
        self.state_proj=nn.Linear(width,width)
        self.skip_proj=nn.Linear(width,width)
        self.log_decay=nn.Parameter(torch.zeros(width))
        self.norm_out=nn.RMSNorm(width)
        self.ff_up=nn.Linear(width,2*ff_width)
        self.ff_down=nn.Linear(ff_width,width)
        self.lm_head=nn.Linear(width,vocab)
        if self.parameter_count()!=498288:
            raise RuntimeError("SSM parameter-count regression")

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @staticmethod
    def parallel_scan(a,b):
        """Inclusive associative scan of h[t]=a[t]*h[t-1]+b[t], h[-1]=0.

        Composition (a2,b2) o (a1,b1) = (a2*a1, b2+a2*b1).
        Complexity: O(T log T) total vector work, O(log T) graph depth,
        no flattening/reordering of chronological events. FP32 not bit exact
        to a sequential loop because multiplication grouping differs.
        """
        if a.shape!=b.shape or a.ndim!=3 or a.shape[1]<1:
            raise ValueError("nonempty matched [batch,time,hidden] gates needed")
        aa,bb=a,b
        distance=1
        while distance<a.shape[1]:
            new_aa=torch.cat((aa[:,:distance],aa[:,distance:]*aa[:,:-distance]),dim=1)
            new_bb=torch.cat((bb[:,:distance],bb[:,distance:]+aa[:,distance:]*bb[:,:-distance]),dim=1)
            aa,bb=new_aa,new_bb
            distance*=2
        return bb

    @staticmethod
    def serial_scan(a,b):
        if a.shape!=b.shape or a.ndim!=3 or a.shape[1]<1:
            raise ValueError("nonempty matched gates required")
        h=torch.zeros_like(b[:,0])
        values=[]
        for t in range(a.shape[1]):
            h=a[:,t]*h+b[:,t]
            values.append(h)
        return torch.stack(values,1)

    def recurrence(self, tokens, *, use_serial=False):
        if tokens.ndim!=2 or tokens.dtype!=torch.long or tokens.shape[1]<1:
            raise ValueError("nonempty int64 [batch,time] input required")
        if not bool(((tokens>=0)&(tokens<128)).all()):
            raise ValueError("out of vocabulary source byte")
        z=self.norm_in(self.embedding(tokens))
        delta,candidate,read=self.x_proj(z).chunk(3,-1)
        # Strictly positive decay and elapsed-time factors: 0<a<=1.
        a=torch.exp(-F.softplus(delta)*F.softplus(self.log_decay)[None,None,:])
        b=(1-a)*torch.tanh(candidate)
        states=(self.serial_scan if use_serial else self.parallel_scan)(a,b)
        return z,states,read

    def forward(self, tokens, *, use_serial=False):
        z,states,read=self.recurrence(tokens,use_serial=use_serial)
        h=self.skip_proj(z)+self.state_proj(states*torch.sigmoid(read))
        h=self.norm_out(h)
        gate,value=self.ff_up(h).chunk(2,-1)
        h=h+self.ff_down(F.silu(gate)*value)
        return self.lm_head(h)
