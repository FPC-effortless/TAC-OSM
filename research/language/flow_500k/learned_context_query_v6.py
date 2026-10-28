"""GRAPH010: source-causal learned *approximate-context* retrieval on exact GRAPH009 misses.

No claim of semantic understanding: this is a tiny learned metric over previous
3-character source contexts, trained on genuine FIT next-character outcomes.

Memory stores at most capacity dated observed (context,continuation) events
per owner and a 4-character carry. The neural 499,290-weight backbone
remains FROZEN. Every comparator must receive exactly the same retrieval
state, memory, external storage and query FLOP accounting.

The cache includes an event (h_j, x_(j+1)) only if j+1 < t. Its current
query uses x_0..x_t and predicts x_(t+1); it never inspects x_(t+1).
"""
from __future__ import annotations
from collections import deque
import math
import torch
from torch import nn
from torch.nn import functional as F

class LearnedContextQuery(nn.Module):
    def __init__(self,prior,*,capacity=512,width=16,order=3,
                 topk=8,temperature=.25,tau=.05,min_similarity=-1.):
        super().__init__()
        if (type(order) is not int or order !=3 or type(capacity) is not int or capacity<2
            or type(width) is not int or width<1 or type(topk) is not int or topk<1
            or not (math.isfinite(temperature) and temperature>0)
            or not (math.isfinite(tau) and tau>0)
            or not (math.isfinite(min_similarity) and -1<=min_similarity<=1)):
            raise ValueError("invalid predeclared query geometry")
        if (not isinstance(prior,torch.Tensor) or prior.shape!=(128,)
            or not bool(torch.isfinite(prior).all()) or not bool((prior>0).all())
            or abs(float(prior.double().sum())-1)>1e-6):
            raise ValueError("strict positive normalized FIT-only prior required")
        self.register_buffer("fit_prior",prior.detach().clone())
        self.embedding=nn.Embedding(128,width)
        self.project=nn.Linear(order*width,width,bias=False)
        self.capacity=capacity;self.order=order;self.width=width
        self.topk=topk;self.temperature=temperature;self.tau=tau
        self.min_similarity=min_similarity

    def learned_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def encode(self,contexts):
        if contexts.shape[-1]!=self.order or contexts.dtype!=torch.long:
            raise ValueError("context must have three integer bytes")
        enc=self.project(self.embedding(contexts).flatten(-2))
        return F.normalize(enc,dim=-1,eps=1e-8)

    def predict_window(self,observed):
        """Differentiable teacher-forced source prefix computation for FIT.
        Returns [B,T,128] distribution on queries observed through time t.
        Final position has no *local* subsequent training label.
        """
        if observed.ndim!=2 or observed.dtype!=torch.long or observed.shape[1]<self.order+2:
            raise ValueError("window must be int64[B,T] with T>=5")
        if not bool(((observed>=0)&(observed<128)).all()):
            raise ValueError("source byte outside vocabulary")
        B,T=observed.shape;k=self.order
        contexts=observed.unfold(1,k,1).contiguous()  # [B,N,k]
        N=contexts.shape[1]
        encoded=self.encode(contexts)
        score=encoded@encoded.transpose(-1,-2)
        ix=torch.arange(N,device=observed.device)
        query_ix=ix[:,None];source_ix=ix[None,:]
        strictly_observed=(source_ix+1 < query_ix)  # continuation time < now
        in_window=(query_ix-source_ix-1<=self.capacity)
        legal=strictly_observed & in_window
        same=(contexts[:,:,None,:]==contexts[:,None,:,:]).all(-1)
        exact_hit=(same & legal[None]).any(-1)
        allowable=legal[None] & ~same & ~exact_hit[:,:,None]
        masked=score.masked_fill(~allowable,float("-inf"))
        K=min(self.topk,N)
        vals,inds=masked.topk(K,-1)
        good=torch.isfinite(vals) & (vals>=self.min_similarity)
        has=good.any(-1)
        # -1e9 sentinel, then null all invalid positions and renormalize.
        w=torch.softmax(torch.where(good,vals/self.temperature,torch.full_like(vals,-1e9)),-1)
        w=w*good
        w=w/w.sum(-1,keepdim=True).clamp_min(1e-30)
        # Source context index j has observed continuation byte x[k+j].
        # The last context cannot be an eligible memory entry.
        labels=torch.cat((observed[:,k:],observed.new_zeros((B,1))),1)
        picked=torch.gather(labels[:,None,:].expand(-1,N,-1),-1,inds)
        counts=torch.zeros((B,N,128),device=observed.device,dtype=encoded.dtype)
        counts.scatter_add_(-1,picked,w)
        prior=self.fit_prior.to(device=observed.device,dtype=encoded.dtype)
        posterior=(counts+self.tau*prior[None,None,:])/(w.sum(-1,keepdim=True)+self.tau)
        posterior=torch.where(has[:,:,None],posterior,prior[None,None,:])
        before=prior.view(1,1,128).expand(B,k-1,128)
        probs=torch.cat((before,posterior),dim=1)
        hit=torch.cat((torch.zeros((B,k-1),device=observed.device,dtype=torch.bool),has),dim=1)
        exact=torch.cat((torch.zeros((B,k-1),device=observed.device,dtype=torch.bool),exact_hit),dim=1)
        return {"posterior":probs,"hit":hit,"exact_hit":exact,
                "max_cosine":torch.cat((torch.full((B,k-1),-1.,device=observed.device,dtype=score.dtype),
                        vals.masked_fill(~good,-1).max(-1).values),1)}

    def contrastive_window_loss(self,observed):
        """FIT-only supervised source-continuation retrieval objective.

        Train using later observed labels in LOSS ONLY. Query embeddings are
        functions only of context x_0..x_t; no labels/future bytes enter
        predict_window or scan. Positives are earlier non-identical contexts
        with the same observed NEXT byte. If no positives exist, skip rather
        than pretending any arbitrary neighbor can recover the target.
        """
        if observed.ndim!=2 or observed.dtype!=torch.long or observed.shape[1]<self.order+2:
            raise ValueError("long enough int64 source training window required")
        if not bool(((observed>=0)&(observed<128)).all()):
            raise ValueError("source byte outside vocabulary")
        k=self.order;B,T=observed.shape
        ctx=observed.unfold(1,k,1).contiguous()
        N=ctx.shape[1]
        z=self.encode(ctx)
        scores=(z@z.transpose(-1,-2))/self.temperature
        ix=torch.arange(N,device=observed.device)
        j=ix[None,:];i=ix[:,None]
        legal=(j+1<i)&(i-j-1<=self.capacity)
        same=(ctx[:,:,None,:]==ctx[:,None,:,:]).all(-1)
        exact=(same&legal[None]).any(-1)
        possible=legal[None] & ~same & ~exact[:,:,None]
        # The last source context has no next-byte target inside this
        # training window, and is always excluded from training labels.
        target=observed[:,k:]  # [B,N-1]: next byte after each context
        label=torch.cat((target,observed.new_zeros((B,1))),1)
        positives=possible&(label[:,:,None]==label[:,None,:])
        has_positive=positives.any(-1)
        eligible=has_positive.clone()
        eligible[:,-1]=False
        if not bool(eligible.any()):
            return scores.sum()*0,{
                "training_queries":int(B*(N-1)),
                "nonexact_miss_queries":int((~exact[:,:-1]).sum()),
                "positive_evidence_queries":0}
        # Avoid NaN backward on unsupported rows: logsumexp(all -inf)
        # has undefined derivatives, even when those rows are not used in
        # the final loss. A finite sentinel plus explicit eligible masking
        # makes positive-evidence training numerically fail-closed.
        all_score=torch.logsumexp(scores.masked_fill(~possible,-1e9),dim=-1)
        pos_score=torch.logsumexp(scores.masked_fill(~positives,-1e9),dim=-1)
        loss=(all_score-pos_score)[eligible].mean()
        if not bool(torch.isfinite(loss)):
            raise ValueError("nonfinite positive-evidence ranking loss")
        return loss,{
            "training_queries":int(B*(N-1)),
            "nonexact_miss_queries":int((~exact[:,:-1]).sum()),
            "positive_evidence_queries":int(eligible.sum())}

    def empty(self,owners):
        if type(owners) is not tuple or not owners or any(type(x) is not str or not x for x in owners):
            raise ValueError("nonempty unique owner identifiers required")
        if len(set(owners))!=len(owners):
            raise ValueError("each batch row must have distinct owner")
        return dict(count=0,source_ids=owners,history=tuple(() for _ in owners),
                    events=tuple(() for _ in owners))

    def scan(self,observed,*,start=0,source_ids=None,initial_state=None):
        """Exact same causal event rule as GRAPH009 with owner-bound state.
        Differentiable query scoring; Python event loop is NOT a fast GPU kernel.
        """
        if observed.ndim!=2 or observed.dtype!=torch.long or observed.shape[1]<1:
            raise ValueError("nonempty int64[B,T] source required")
        B,T=observed.shape
        if type(start) is not int or start<0 or type(source_ids) is not tuple or len(source_ids)!=B:
            raise ValueError("invalid start/owner")
        if not bool(((observed>=0)&(observed<128)).all()):
            raise ValueError("source out of 128-character vocabulary")
        state=self.empty(source_ids) if initial_state is None else initial_state
        if (not isinstance(state,dict) or state.get("count")!=start
            or state.get("source_ids")!=source_ids
            or len(state.get("events",()))!=B or len(state.get("history",()))!=B):
            raise ValueError("carried owner/ordinal mismatch")
        outputs=[];allhit=[];all_exact=[];newqueues=[];newhist=[]
        prior=self.fit_prior.to(device=observed.device,dtype=self.embedding.weight.dtype)
        for row in range(B):
            history=deque(state["history"][row],maxlen=self.order+1)
            if len(history)>min(start,self.order+1) or any(type(x) is not int or x<0 or x>=128 for x in history):
                raise ValueError("invalid carried context history")
            events=deque(state["events"][row]);last=-1
            if len(events)>self.capacity+1:raise ValueError("unbounded carried event count")
            for ordinal,key,value in events:
                if (type(ordinal) is not int or ordinal<=last or ordinal<max(0,start-self.capacity-1)
                    or ordinal>start-2 or type(key) is not tuple or len(key)!=self.order
                    or any(type(c) is not int or not 0<=c<128 for c in key)
                    or type(value) is not int or not 0<=value<128):
                    raise ValueError("invalid carried causal event")
                last=ordinal
            rowp=[];rowh=[];rowe=[]
            for delta,byte in enumerate(observed[row].tolist()):
                now=start+delta
                while events and events[0][0]<now-self.capacity:
                    events.popleft()
                if len(history)>=self.order+1:
                    # Source follow-up x_(now-1) has been observed strictly
                    # before this current prediction at now.
                    events.append((now-1,tuple(list(history)[-self.order-1:-1]),history[-1]))
                history.append(byte)
                if len(history)<self.order:
                    rowp.append(prior);rowh.append(False);rowe.append(False);continue
                query=tuple(list(history)[-self.order:])
                exact=any(c==query for _,c,_ in events)
                candidate=[(c,v) for _,c,v in events if c!=query]
                if exact or not candidate:
                    rowp.append(prior);rowh.append(False);rowe.append(exact);continue
                ctx=torch.tensor([c for c,_ in candidate],dtype=torch.long,device=observed.device)
                q=torch.tensor(query,dtype=torch.long,device=observed.device)[None,:]
                scores=self.encode(ctx)@self.encode(q)[0]
                kk=min(self.topk,len(candidate))
                vals,inds=scores.topk(kk)
                valid=vals>=self.min_similarity
                if not bool(valid.any()):
                    rowp.append(prior);rowh.append(False);rowe.append(False);continue
                vals=vals[valid];inds=inds[valid]
                weights=torch.softmax(vals/self.temperature,0)
                counts=torch.zeros(128,device=observed.device,dtype=weights.dtype)
                labels=torch.tensor([candidate[int(i)][1] for i in inds],device=observed.device,dtype=torch.long)
                counts.scatter_add_(0,labels,weights)
                post=(counts+self.tau*prior)/(weights.sum()+self.tau)
                rowp.append(post);rowh.append(True);rowe.append(False)
            outputs.append(torch.stack(rowp))
            allhit.append(rowh);all_exact.append(rowe)
            newqueues.append(tuple(events));newhist.append(tuple(history))
        return {"posterior":torch.stack(outputs),
                "hit":torch.tensor(allhit,device=observed.device,dtype=torch.bool),
                "exact_hit":torch.tensor(all_exact,device=observed.device,dtype=torch.bool)}, \
            {"count":start+T,"source_ids":source_ids,"history":tuple(newhist),
             "events":tuple(newqueues)}
