"""Independent functional reference: copied parameter tensors, no model calls.

Manual GRU equations and explicitly reconstructed sparse/straight-through
routes. Exact-real and source-gradient parity are numerical checks only;
whole IEEE/AD formal refinement remains OPEN.
"""
from dataclasses import dataclass
import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class RefState:
    owners: tuple
    clock: int
    tensors: dict
    pending_bytes: tuple
    pending_modes: tuple
    cache_tokens: tuple=()


def forward(p,mode,tokens,*,owners=None,state=None,merges=(),force=None,code_override=None,execution='surrogate'):
    # This module deliberately does not import the candidate implementation.
    b,t=tokens.shape;hsize=p['gru.weight_hh'].shape[1]
    owners=tuple(owners) if owners is not None else tuple('train-'+str(i) for i in range(b))
    if state is None:
        z=p['embedding.weight']
        state=RefState(owners,0,{'hidden':z.new_zeros((b,hsize)),'pending':z.new_zeros((b,4,24))},tuple(b'' for _ in owners),tuple(1 for _ in owners))

    def linear(x,name):return F.linear(x,p[name+'.weight'],p[name+'.bias'])
    def embed(i):return F.embedding(i,p['embedding.weight'])
    def gru(x,h):
        ir,iz,inn=F.linear(x,p['gru.weight_ih'],p['gru.bias_ih']).chunk(3,-1)
        hr,hz,hn=F.linear(h,p['gru.weight_hh'],p['gru.bias_hh']).chunk(3,-1)
        r=(ir+hr).sigmoid();z=(iz+hz).sigmoid();n=(inn+r*hn).tanh()
        return (1-z)*n+z*h
    def head(h):return linear(h,'head')
    def pool(xs):
        full=xs+[torch.zeros_like(xs[0]) for _ in range(4-len(xs))]
        return linear(torch.cat(full,-1),'pool').tanh()
    def rep(xs,kind):
        if len(xs)==1:return xs[0]
        z=pool(xs)
        if kind==2:return z
        scores=-(z[:,None]-p['codes.weight'][None]).square().sum(-1)
        q=scores.softmax(-1)
        index=q.detach().argmax(-1) if code_override is None else torch.tensor([code_override],dtype=torch.long,device=z.device)
        one=F.one_hot(index,16).to(z.dtype);w=one+(q-q.detach())
        return w@p['codes.weight']
    def bpe(raw):
        seq=list(raw)
        for rank,pair in enumerate(merges):
            out=[];i=0
            while i<len(seq):
                if i+1<len(seq) and (seq[i],seq[i+1])==tuple(pair):out.append(256+rank);i+=2
                else:out.append(seq[i]);i+=1
            seq=out
        return seq

    outputs=[];hs=[];pads=[];raws=[];kinds=[];cache_rows=[];cache_ids=[]
    for row in range(b):
        raw=state.pending_bytes[row];h=state.tensors['hidden'][row:row+1]
        xs=[state.tensors['pending'][row:row+1,j] for j in range(len(raw))] if mode!='a5_fixed_bpe' else []
        kind=state.pending_modes[row];out=[]
        for idx in tokens[row]:
            if mode=='a5_fixed_bpe':
                raw+=bytes([int(idx.item())]);h=p['embedding.weight'].new_zeros((1,hsize))
                final_ids=[288]+bpe(raw);final_cached=[]
                for token in final_ids:
                    h=gru(embed(torch.tensor([token],device=tokens.device)),h);final_cached.append(h)
                out.append(head(h));continue
            current=embed(idx.reshape(1));combined=xs+[current]
            if mode=='a5_byte':route=0;q=None
            elif mode in ('a5_discrete','a5_latent'):route=1 if mode=='a5_discrete' else 2;q=None
            else:
                z=pool(combined);lp=head(h).log_softmax(-1)
                entropy=-(lp.exp()*lp).sum(-1,keepdim=True)/5.545177444479562
                scores=linear(torch.cat((h,current,z,entropy),-1),'controller');q=scores.softmax(-1)
                route=int(scores.detach().argmax(-1).item()) if force is None else ('byte','discrete','latent').index(force)
            finish=len(combined)==4
            if mode=='a5_adaptive' and execution=='surrogate':
                before=gru(rep(xs,kind),h) if xs else h
                candidate=[gru(current,before)]+[gru(rep(combined,k),h) if finish else h for k in (1,2)]
                if force is None:
                    one=F.one_hot(torch.tensor([route],device=tokens.device),3).to(h.dtype);w=one+(q-q.detach())
                    h=sum(w[:,j:j+1]*candidate[j] for j in range(3))
                else:h=candidate[route]
            if route==0:
                if mode!='a5_adaptive' or execution=='inference':
                    if xs:h=gru(rep(xs,kind),h)
                    h=gru(current,h)
                xs=[];raw=b''
            else:
                xs=combined;raw+=bytes([int(idx.item())]);kind=route
                if finish:
                    if mode!='a5_adaptive' or execution=='inference':h=gru(rep(xs,route),h)
                    xs=[];raw=b''
            out.append(head(h))
        outputs.append(torch.cat(out,0));hs.append(h);raws.append(raw);kinds.append(kind)
        zeros=[h.new_zeros((1,24)) for _ in range(4-len(xs))]
        pads.append(torch.stack(xs+zeros,1))
        if mode=='a5_fixed_bpe':cache_rows.append(final_cached);cache_ids.append(tuple(final_ids))
    tensors={'hidden':torch.cat(hs,0),'pending':torch.cat(pads,0)}
    if mode=='a5_fixed_bpe':
        size=max(map(len,cache_rows));packed=[]
        for cached in cache_rows:packed.append(torch.stack(cached+[hs[0].new_zeros((1,hsize)) for _ in range(size-len(cached))],1))
        tensors['bpe_cache']=torch.cat(packed,0)
    final=RefState(owners,state.clock+t,tensors,tuple(raws),tuple(kinds),tuple(cache_ids))
    return torch.stack(outputs),final
