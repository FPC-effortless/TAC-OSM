"""LANGMATH101 optional A5 candidates, no optimizer or data access.

Unlike original SENSE-TOK005, the byte decoder receives hidden state ONLY.
Once a multi-byte patch is committed as a discrete ID, its raw byte vectors do
not have an additional readout path. Incomplete patches deliberately leave the
prediction unchanged until commitment: valid causal byte likelihood, not a
promise of quality or speed. New source/PNDS refinement remains OPEN.
"""
from dataclasses import dataclass
from collections import Counter
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

MODES=('a5_byte','a5_fixed_bpe','a5_discrete','a5_latent','a5_adaptive')
WIDTH=24
SPAN=4


@dataclass(frozen=True)
class AdaptiveState:
    owners: tuple[str,...]
    clock: int
    tensors: dict[str,torch.Tensor]
    pending_bytes: tuple[bytes,...]
    pending_modes: tuple[int,...]
    cache_tokens: tuple[tuple[int,...],...]=()
    cache_guard: tuple=()

    def tensor_bytes(self):
        return sum(t.numel()*t.element_size() for t in self.tensors.values())


class FrozenBPE:
    """Conventional ranked byte-pair merges fitted solely to supplied FIT files.

    Prefix encodings can revise earlier token boundaries, so this reference
    RETOKENIZES the complete observed prefix, reuses an unchanged token-prefix
    hidden cache, and recomputes the changed suffix. It does not pretend that
    growing raw history/hidden cache is constant-memory or Transformer KV state.
    No future bytes, EOS or DEV/TEST records enter tokenization.
    """
    def __init__(self,merges,source_hashes,*,partition='FIT'):
        if partition!='FIT' or len(merges)!=32:
            raise ValueError('exactly32 FIT-only ranked merges required')
        self.merges=tuple(tuple(map(int,p)) for p in merges)
        self.source_hashes=tuple(source_hashes)
        for rank,(a,b) in enumerate(self.merges):
            if not (0<=a<256+rank and 0<=b<256+rank):
                raise ValueError('merge refers to unavailable token')
        if not self.source_hashes or any(len(h)!=64 for h in self.source_hashes):
            raise ValueError('complete original FIT content hashes required')

    @classmethod
    def fit(cls,sources,*,partition):
        if partition!='FIT' or not sources or any(not isinstance(s,bytes) or not s for s in sources):
            raise ValueError('explicit nonempty original FIT byte sources required')
        sequences=[list(s) for s in sources]
        merges=[]
        for rank in range(32):
            counts=Counter(pair for seq in sequences for pair in zip(seq,seq[1:]))
            if not counts:raise ValueError('FIT cohort cannot define32 nonempty BPE merges')
            pair=min(counts,key=lambda p:(-counts[p],p))
            merges.append(pair);token=256+rank
            sequences=[cls._merge(seq,pair,token) for seq in sequences]
        return cls(merges,[hashlib.sha256(s).hexdigest() for s in sources])

    @staticmethod
    def _merge(seq,pair,token):
        out=[];i=0
        while i<len(seq):
            if i+1<len(seq) and (seq[i],seq[i+1])==pair:
                out.append(token);i+=2
            else:out.append(seq[i]);i+=1
        return out

    def encode(self,raw):
        seq=list(raw)
        for rank,pair in enumerate(self.merges):seq=self._merge(seq,pair,256+rank)
        return seq

    def decode(self,ids):
        table={i:bytes([i]) for i in range(256)}
        for rank,(a,b) in enumerate(self.merges):table[256+rank]=table[a]+table[b]
        return b''.join(table[int(i)] for i in ids)

    def receipt(self):
        body={'partition':'FIT','merges':self.merges,'source_sha256':self.source_hashes}
        body['sha256']=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()
        return body

    @classmethod
    def from_receipt(cls,receipt):
        result=cls(receipt['merges'],receipt['source_sha256'],partition=receipt['partition'])
        if result.receipt()['sha256']!=receipt['sha256']:raise ValueError('frozen tokenizer receipt changed')
        return result


class AdaptiveRepresentationModel(nn.Module):
    arm='a5'
    def __init__(self,mode='a5_adaptive',*,tokenizer=None):
        super().__init__()
        if mode not in MODES:raise ValueError('registered A5 mode required')
        self.mode=mode;self.tokenizer=tokenizer
        self.hidden_size={'a5_byte':130,'a5_fixed_bpe':129,'a5_discrete':127,'a5_latent':128,'a5_adaptive':127}[mode]
        if mode=='a5_fixed_bpe' and not isinstance(tokenizer,FrozenBPE):
            raise ValueError('FIT-only frozen BPE tokenizer required')
        vocab=289 if mode=='a5_fixed_bpe' else 256
        self.embedding=nn.Embedding(vocab,WIDTH)
        self.gru=nn.GRUCell(WIDTH,self.hidden_size)
        self.head=nn.Linear(self.hidden_size,256)
        if mode in ('a5_discrete','a5_latent','a5_adaptive'):
            self.pool=nn.Linear(SPAN*WIDTH,WIDTH)
        if mode in ('a5_discrete','a5_adaptive'):
            self.codes=nn.Embedding(16,WIDTH)
        if mode=='a5_adaptive':
            self.controller=nn.Linear(self.hidden_size+2*WIDTH+1,3)
        self.last_receipt=None

    def counts(self):
        return {'trainable':sum(p.numel() for p in self.parameters() if p.requires_grad),
                'deployed':sum(p.numel() for p in self.parameters())}

    def configuration(self):
        return {'mode':self.mode,'hidden_size':self.hidden_size,'embedding_width':WIDTH,
            'patch_span':SPAN,'code_count':16 if hasattr(self,'codes') else None,
            'tokenizer':self.tokenizer.receipt() if self.mode=='a5_fixed_bpe' else None,
            'raw_byte_readout_bypass':False,'whole_IEEE_AD':'OPEN'}

    def start(self,owners):
        owners=tuple(owners)
        if not owners or len(set(owners))!=len(owners) or any(type(o)is not str or not o for o in owners):
            raise ValueError('unique nonempty explicit owners required')
        b=len(owners);w=self.embedding.weight
        tensors={'hidden':w.new_zeros((b,self.hidden_size)),'pending':w.new_zeros((b,SPAN,WIDTH))}
        if self.mode=='a5_fixed_bpe':tensors['bpe_cache']=w.new_zeros((b,0,self.hidden_size))
        return AdaptiveState(owners,0,tensors,tuple(b'' for _ in owners),tuple(1 for _ in owners),
            tuple(() for _ in owners) if self.mode=='a5_fixed_bpe' else (),())

    def _validate(self,tokens,state,owners):
        if (not isinstance(tokens,torch.Tensor) or tokens.dtype!=torch.long or tokens.ndim!=2
             or min(tokens.shape)<1 or bool(((tokens<0)|(tokens>255)).any())):
            raise ValueError('observed byte Long[B,T] required')
        if (state.owners!=owners or len(owners)!=tokens.shape[0] or len(set(owners))!=len(owners)
              or any(type(o)is not str or not o for o in owners)
              or type(state.clock)is not int or state.clock<0):raise ValueError('owner/clock mismatch')
        if self.embedding.weight.dtype not in (torch.float32,torch.float64) or tokens.device!=self.embedding.weight.device:
            raise ValueError('qualified FP32/FP64 model and matching device required')
        shapes={'hidden':(len(owners),self.hidden_size),'pending':(len(owners),SPAN,WIDTH)}
        if self.mode=='a5_fixed_bpe':
            if len(state.cache_tokens)!=len(owners) or any(any(type(k)is not int or not 0<=k<=288 for k in ids) for ids in state.cache_tokens):
                raise ValueError('explicit cached token IDs required')
            shapes['bpe_cache']=(len(owners),max(map(len,state.cache_tokens),default=0),self.hidden_size)
        if set(state.tensors)!=set(shapes):raise ValueError('complete state required')
        for name,v in state.tensors.items():
            if v.shape!=shapes[name] or v.dtype!=self.embedding.weight.dtype or v.device!=tokens.device or not bool(torch.isfinite(v).all()):
                raise ValueError('state geometry/precision/device/finite mismatch')
        limit=state.clock if self.mode=='a5_fixed_bpe' else SPAN-1
        if len(state.pending_bytes)!=len(owners) or any(not isinstance(x,bytes) or len(x)>limit for x in state.pending_bytes):
            raise ValueError('pending source history mismatch')
        if self.mode=='a5_fixed_bpe' and any(len(x)!=state.clock for x in state.pending_bytes):raise ValueError('complete prefix state required')
        if len(state.pending_modes)!=len(owners) or any(k not in (1,2) for k in state.pending_modes):raise ValueError('pending type mismatch')

    def _pool(self,items):
        if not 1<=len(items)<=SPAN:raise ValueError('observed patch length required')
        pad=[torch.zeros_like(items[0]) for _ in range(SPAN-len(items))]
        return self.pool(torch.cat(items+pad,-1)).tanh()

    def _discrete(self,items,override=None):
        pooled=self._pool(items)
        distances=(pooled[:,None]-self.codes.weight[None]).square().sum(-1)
        soft=(-distances).softmax(-1)
        index=soft.detach().argmax(-1) if override is None else torch.full((1,),int(override),dtype=torch.long,device=pooled.device)
        if bool(((index<0)|(index>=16)).any()):raise ValueError('code override outside vocabulary')
        hard=F.one_hot(index,16).to(pooled.dtype)
        weights=hard+(soft-soft.detach())
        return weights@self.codes.weight,int(index.item())

    def _represent(self,items,kind,override=None):
        if len(items)==1:return items[0],None
        if kind==1:return self._discrete(items,override)
        return self._pool(items),None

    def forward(self,tokens,*,state=None,owners=None,parallel=True,force=None,code_override=None,execution='surrogate'):
        if force not in (None,'byte','discrete','latent'):raise ValueError('invalid evaluator-only route intervention')
        if force is not None and self.mode!='a5_adaptive':raise ValueError('route intervention requires adaptive controller')
        if execution not in ('surrogate','inference'):raise ValueError('execution must distinguish training surrogate from sparse inference')
        if code_override is not None and self.mode not in ('a5_discrete','a5_adaptive'):raise ValueError('code intervention on noncode arm')
        if code_override is not None and (type(code_override)is not int or not 0<=code_override<16):raise ValueError('integer code intervention0..15 required')
        if not isinstance(tokens,torch.Tensor) or tokens.ndim!=2:raise ValueError('observed byte tensor required')
        owners=tuple(owners) if owners is not None else tuple('train-'+str(i) for i in range(tokens.shape[0]))
        state=self.start(owners) if state is None else state;self._validate(tokens,state,owners)
        if self.mode=='a5_fixed_bpe':return self._bpe(tokens,state,owners)
        logits=[];hidden=[];buffers=[];raws=[];kinds=[];events=[];updates=0;candidate_calls=0
        for row in range(tokens.shape[0]):
            h=state.tensors['hidden'][row:row+1]
            raw=state.pending_bytes[row];items=[state.tensors['pending'][row:row+1,j] for j in range(len(raw))]
            pending_kind=state.pending_modes[row];outs=[]
            for offset,idx in enumerate(tokens[row]):
                current=self.embedding(idx.reshape(1));byte=int(idx.item());all_items=items+[current]
                if self.mode=='a5_byte':route=0;probs=None
                elif self.mode in ('a5_discrete','a5_latent'):route=1 if self.mode=='a5_discrete' else 2;probs=None
                else:
                    p=self._pool(all_items)
                    prev=self.head(h).log_softmax(-1);entropy=-(prev.exp()*prev).sum(-1,keepdim=True)/5.545177444479562
                    gate=self.controller(torch.cat((h,current,p,entropy),-1));probs=gate.softmax(-1)
                    route=int(gate.detach().argmax(-1).item()) if force is None else ('byte','discrete','latent').index(force)
                finish=len(all_items)==SPAN
                if self.mode=='a5_adaptive' and execution=='surrogate':
                    # All candidate calls charged to training. Forward uses only
                    # selected candidate; the categorical gate gets ST gradients.
                    preh=h
                    if items:
                        r,_=self._represent(items,pending_kind,code_override);preh=self.gru(r,h);candidate_calls+=1
                    candidates=[self.gru(current,preh)];candidate_calls+=1
                    for kind in (1,2):
                        if finish:
                            r,_=self._represent(all_items,kind,code_override);candidates.append(self.gru(r,h));candidate_calls+=1
                        else:candidates.append(h)
                    if force is None:
                        hard=F.one_hot(torch.tensor([route],device=tokens.device),3).to(h.dtype)
                        weights=hard+(probs-probs.detach());h=sum(weights[:,j:j+1]*candidates[j] for j in range(3))
                    else:h=candidates[route]
                if route==0:
                    if items:
                        rep,code=self._represent(items,pending_kind,code_override)
                        if self.mode!='a5_adaptive' or execution=='inference':h=self.gru(rep,h);candidate_calls+=1
                        updates+=1;events.append({'row':row,'end':state.clock+offset,'bytes':len(items),'code':code,'kind':pending_kind})
                    if self.mode!='a5_adaptive' or execution=='inference':h=self.gru(current,h);candidate_calls+=1
                    updates+=1;events.append({'row':row,'end':state.clock+offset+1,'bytes':1,'code':None,'kind':0})
                    items=[];raw=b''
                else:
                    items=all_items;raw+=bytes([byte]);pending_kind=route
                    if finish:
                        rep,code=self._represent(items,route,code_override)
                        if self.mode!='a5_adaptive' or execution=='inference':h=self.gru(rep,h);candidate_calls+=1
                        updates+=1;events.append({'row':row,'end':state.clock+offset+1,'bytes':len(items),'code':code,'kind':route})
                        items=[];raw=b''
                # NO byte/current/pool/token-ID concatenation enters this head.
                outs.append(self.head(h))
            logits.append(torch.cat(outs,0));hidden.append(h);raws.append(raw);kinds.append(pending_kind)
            zeros=[h.new_zeros((1,WIDTH)) for _ in range(SPAN-len(items))]
            buffers.append(torch.stack(items+zeros,1))
        final=AdaptiveState(owners,state.clock+tokens.shape[1],{'hidden':torch.cat(hidden,0),'pending':torch.cat(buffers,0)},tuple(raws),tuple(kinds))
        self.last_receipt={'source_bytes':int(tokens.numel()),'recurrent_updates':updates,'candidate_recurrent_calls':candidate_calls,
            'events':events,'pending_source_bytes':sum(map(len,raws)),'byte_readouts':int(tokens.numel()),'execution':execution,
            'raw_byte_readout_bypass':False,'compute_savings_established':False}
        return torch.stack(logits),final

    def _bpe(self,tokens,state,owners):
        outputs=[];histories=[];hs=[];steps=0;cached_steps=0;all_cached=[];all_ids=[]
        # Guard ordinary object/version/dtype/device changes, not adversarial
        # raw-storage tamper. A different model reload rebuilds from raw bytes.
        versions=[];tracked=True
        for name,p in self.named_parameters():
            try:version=p._version
            except RuntimeError:version=None;tracked=False
            versions.append((name,id(p),version,str(p.dtype),str(p.device)))
        guard=(('tokenizer',self.tokenizer.receipt()['sha256']),)+tuple(versions)
        valid_guard=tracked and state.cache_guard==guard
        for row in range(tokens.shape[0]):
            history=state.pending_bytes[row];out=[]
            old_ids=list(state.cache_tokens[row]) if valid_guard else []
            # Cloning restores the source kernel's aligned single-state input
            # layout instead of feeding offset slices from a packed cache.
            cached=[state.tensors['bpe_cache'][row:row+1,j].clone() for j in range(len(old_ids))]
            # State metadata must agree with the actual observed source prefix;
            # source-derived retokenization is authoritative on invalid caches.
            expected=[288]+self.tokenizer.encode(history) if history else []
            if old_ids!=expected:old_ids=[];cached=[]
            for idx in tokens[row]:
                history+=bytes([int(idx.item())]);ids=[288]+self.tokenizer.encode(history)
                common=0
                while common<min(len(old_ids),len(ids)) and old_ids[common]==ids[common]:common+=1
                cached_steps+=common
                h=cached[common-1] if common else self.embedding.weight.new_zeros((1,self.hidden_size))
                new_cached=cached[:common]
                for token in ids[common:]:
                    h=self.gru(self.embedding(torch.tensor([token],device=tokens.device)),h);steps+=1
                    new_cached.append(h)
                old_ids=ids;cached=new_cached
                out.append(self.head(h))
            outputs.append(torch.cat(out,0));histories.append(history);hs.append(h);all_cached.append(cached);all_ids.append(tuple(old_ids))
        maxlen=max(map(len,all_cached));packed=[]
        for cached in all_cached:
            packed.append(torch.stack(cached+[self.embedding.weight.new_zeros((1,self.hidden_size)) for _ in range(maxlen-len(cached))],1))
        final=AdaptiveState(owners,state.clock+tokens.shape[1],{'hidden':torch.cat(hs,0),
            'pending':self.embedding.weight.new_zeros((len(owners),SPAN,WIDTH)),
            'bpe_cache':torch.cat(packed,0)},tuple(histories),tuple(1 for _ in owners),tuple(all_ids),guard)
        self.last_receipt={'source_bytes':int(tokens.numel()),'recurrent_updates':steps,'candidate_recurrent_calls':steps,
            'byte_readouts':int(tokens.numel()),'raw_history_bytes':sum(map(len,histories)),
            'reused_prefix_hidden_states':cached_steps,'cache_guard_valid_on_entry':valid_guard,
            'cached_token_ids':sum(map(len,all_ids)),'cache_tensor_bytes':final.tensors['bpe_cache'].numel()*final.tensors['bpe_cache'].element_size(),
            'tokenizer_sha256':self.tokenizer.receipt()['sha256'],'prefix_retokenization':True,
            'prefix_hidden_reuse':True,'raw_storage_tamper_proof':'OPEN',
            'parameter_versions_trackable':tracked,
            'raw_byte_readout_bypass':False,'compute_savings_established':False}
        return torch.stack(outputs),final


def make_model(arm,*,tokenizer=None):return AdaptiveRepresentationModel(arm,tokenizer=tokenizer)
