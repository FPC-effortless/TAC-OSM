"""DIAGKEY001: owner-requested synthetic *mechanism-only* keyed recall.

This is not real-world performance, physical forecasting, or architecture
superiority evidence. A synthetic result CANNOT pass the TAC real-data gate.
All four models: 95k-105k active and deployed learned weights; <1% parameter
gap, same synthetic examples/schedules and fixed optimizer updates.
Profiled FLOP mismatch is explicitly reported and blocks a FAIR HEAD-TO-HEAD.
Original ME073 models/evidence and actual real holdouts are untouched.
"""
from __future__ import annotations
import argparse, hashlib, json, math, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

STUDY="DIAGKEY001_SYNTHETIC_ASSOCIATIVE_RECALL_MECHANISM_ONLY"
SEEDS=(75001,75002,75003,75004,75005)
LENGTHS=(16,64,256)
STEPS=640
BATCH=16
NKEY=64
NVAL=32
CHANCE=1.0/NVAL
RESULT_DIR=Path("research/replay-bundles/DIAGKEY001")
ARMS=("tac","tac_noassoc","gru","transformer")
PER_ARM_PARAMETERS={"tac":101554,"tac_noassoc":101560,"gru":101500,"transformer":101616}

class Native(nn.Module):
    """Direct recurrence lifted from native ME069: nonlinear fast/slow,
    normalized rank-one 8x8 key-value error writing, and norm-bounded state.
    Original workflow source uses 64D; diagnostic width=88 to match 100k.
    This is an isolated TAC-core diagnostic, NOT the identical ME073 model.
    """
    def __init__(self,width=88,decoder=120,association=True):
        super().__init__()
        self.width=width;self.association=association
        self.kind_embed=nn.Embedding(3,48)
        self.key=nn.Embedding(65,48)
        self.value=nn.Embedding(33,48)
        self.encoder=nn.Linear(48,48);self.query=nn.Linear(48,8) if association else None
        self.fast_map=nn.Linear(48+width+(8 if association else 0),2*width)
        self.slow_map=nn.Linear(2*width+(8 if association else 0),2*width)
        if association:self.write_map=nn.Linear(48+2*width,18)
        self.readout=nn.Sequential(nn.Linear(2*width+(8 if association else 0)+48,decoder),nn.SiLU(),
                                   nn.Linear(decoder,NVAL))
    def forward(self,t,k,v):
        x=torch.tanh(self.encoder(self.kind_embed(t)+self.key(k)+self.value(v)))
        f=x.new_zeros(len(x),self.width);s=f.clone();m=x.new_zeros(len(x),8,8)
        for j in range(x.shape[1]):
            e=x[:,j]
            query=F.normalize(f[:,:8]+s[:,:8],dim=-1,eps=1e-6)
            recall=torch.bmm(m,query.unsqueeze(-1)).squeeze(-1)
            value,gate=self.fast_map(torch.cat((e,f,recall),-1) if self.association else torch.cat((e,f),-1)).chunk(2,-1)
            f=torch.lerp(f,value.tanh(),gate.sigmoid())
            delta,gate=self.slow_map(torch.cat((f,s,recall),-1) if self.association else torch.cat((f,s),-1)).chunk(2,-1)
            s=torch.lerp(s,(s+delta.tanh()).tanh(),.25*gate.sigmoid())
            if self.association:
                kk,vv,beta,relevance=self.write_map(torch.cat((e,f,s),-1)).split((8,8,1,1),-1)
                kk=F.normalize(kk,dim=-1,eps=1e-6)
                error=vv.tanh()-torch.bmm(m,kk.unsqueeze(-1)).squeeze(-1)
                m=m+((beta.sigmoid()*relevance.sigmoid())*error).unsqueeze(-1)*kk.unsqueeze(1)
                m=m/m.square().sum((1,2),keepdim=True).sqrt().clamp_min(1.)
        if self.association:
            memory_read=torch.bmm(m,F.normalize(self.query(x[:,-1]),dim=-1).unsqueeze(-1)).squeeze(-1)
            return self.readout(torch.cat((f,s,memory_read,x[:,-1]),-1))
        return self.readout(torch.cat((f,s,x[:,-1]),-1))

class GRU(nn.Module):
    def __init__(self,width=128,decoder=124):
        super().__init__()
        self.kind_embed=nn.Embedding(3,48);self.key=nn.Embedding(65,48)
        self.value=nn.Embedding(33,48)
        self.encoder=nn.Linear(48,48)
        self.cell=nn.GRUCell(48,width)
        self.readout=nn.Sequential(nn.Linear(width+48,decoder),nn.SiLU(),
                                   nn.Linear(decoder,NVAL))
    def forward(self,t,k,v):
        x=torch.tanh(self.encoder(self.kind_embed(t)+self.key(k)+self.value(v)))
        h=x.new_zeros(len(x),self.cell.hidden_size)
        for j in range(x.shape[1]):h=self.cell(x[:,j],h)
        return self.readout(torch.cat((h,x[:,-1]),-1))

class CausalTransformer(nn.Module):
    def __init__(self,width=64,ff=144):
        super().__init__()
        self.kind_embed=nn.Embedding(3,48);self.key=nn.Embedding(65,48)
        self.value=nn.Embedding(33,48)
        self.encoder=nn.Linear(48,width);self.position=nn.Embedding(320,width)
        self.layers=nn.ModuleList([nn.TransformerEncoderLayer(width,4,ff,dropout=0.0,
                        batch_first=True,norm_first=True) for _ in range(2)])
        self.head=nn.Linear(width,NVAL)
    def forward(self,t,k,v):
        x=self.encoder(self.kind_embed(t)+self.key(k)+self.value(v))
        x=x+self.position.weight[:t.shape[1]][None]
        mask=torch.ones((len(x[0]),len(x[0])),device=x.device,dtype=torch.bool).triu(1)
        for layer in self.layers:x=layer(x,src_mask=mask)
        return self.head(x[:,-1])

def model_for(arm):
    return {
        "tac":lambda:Native(88,120,True),
        "tac_noassoc":lambda:Native(88,152,False),
        "gru":lambda:GRU(128,124),
        "transformer":lambda:CausalTransformer(64,144),
    }[arm]()

def model_counts():
    observed={}
    for name in ARMS:
        model=model_for(name)
        trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
        deployed=sum(p.numel() for p in model.parameters())
        if trainable!=PER_ARM_PARAMETERS[name] or deployed!=trainable:
            raise ValueError(f"{name}: original actual model count differs: {trainable} {deployed}")
        if not 95000<=trainable<=105000:raise ValueError("absolute 100k budget violated")
        observed[name]=trainable
    if (max(observed.values())-min(observed.values()))/max(observed.values())>0.01:
        raise ValueError("between-arm 1% parameter parity fails")
    return observed

def batch_for(seed,step,batch_size=BATCH,length=None,split="FIT"):
    if length is None:length=LENGTHS[step%3]
    assert length in LENGTHS
    raw=f"{STUDY}:{split}:{seed}:{step}:{length}:4_pairs:32_values"
    rng=np.random.default_rng(int(hashlib.sha256(raw.encode()).hexdigest()[:16],16))
    n=length+5
    t=np.ones((batch_size,n),dtype=np.int64)
    keys=np.zeros_like(t);values=np.zeros_like(t)
    labels=np.zeros(batch_size,dtype=np.int64)
    t[:,:4]=0;t[:,-1]=2
    for i in range(batch_size):
        src_keys=rng.choice(16,4,replace=False)
        src_values=rng.choice(NVAL,4,replace=False)
        target_slot=int(rng.integers(4))
        keys[i,:4]=src_keys+1;values[i,:4]=src_values+1
        keys[i,4:-1]=rng.integers(17,65,size=length)
        values[i,4:-1]=rng.integers(1,33,size=length)
        keys[i,-1]=src_keys[target_slot]+1
        labels[i]=src_values[target_slot]
        if (keys[i,4:-1]==keys[i,-1]).any():raise ValueError("target key appears as distractor")
        if values[i,-1]!=0:raise ValueError("query leaks answer value")
    return tuple(torch.from_numpy(a) for a in (t,keys,values,labels))

def op_profiler(arm,length):
    """Partial PyTorch-supported operator FLOPs; NOT total hardware compute."""
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                   with_flops=True) as prof:
            model=model_for(arm)
            t,k,v,y=batch_for(76000,0,batch_size=2,length=length,split="PROFILE")
            loss=F.cross_entropy(model(t,k,v),y)
            loss.backward()
        return int(sum(int(e.flops or 0) for e in prof.key_averages()))
    except Exception as err:
        return {"unavailable":str(err)[:250]}

def eval_model(model,seed):
    model.eval();scores={}
    with torch.no_grad():
        for length in LENGTHS:
            correct=0;total=0;negative_log_loss=0.
            # Prespecified 512 independent fresh synthetic sequences per delay.
            for i in range(16):
                t,k,v,y=batch_for(seed+20000,200000+length*37+i,batch_size=32,
                                   length=length,split="TEST")
                logits=model(t,k,v)
                correct+=int((logits.argmax(-1)==y).sum())
                negative_log_loss+=float(F.cross_entropy(logits,y,reduction="sum"))
                total+=len(y)
            scores[str(length)]={"exact_recall_correct":correct,
                "query_count":total,"accuracy":correct/total,
                "NLL":negative_log_loss/total}
    return scores

def one_fit(seed,arm,steps=STEPS):
    if arm not in ARMS or seed not in SEEDS:raise ValueError("not a prespecified original seed/arm")
    torch.set_num_threads(1)
    counts=model_counts()
    torch.manual_seed(seed)
    random.seed(seed);np.random.seed(seed)
    m=model_for(arm);m.train()
    optimizer=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=1e-4)
    outcomes=[];start=time.perf_counter()
    for step in range(steps):
        t,k,v,y=batch_for(seed,step)
        loss=F.cross_entropy(m(t,k,v),y)
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.0)
        optimizer.step()
        if step in (0,steps//2,steps-1):outcomes.append({"update":step+1,"loss":float(loss.detach())})
    score=eval_model(m,seed)
    output={
        "study":STUDY,"arm":arm,"seed":seed,"trained_real_world_capability":False,
        "synthetic_only_mechanism_diagnostic":True,"data_generator":"generated 4 independent stored keys with 16/64/256 unrelated write distractors then answer-free query",
        "trainable_parameters":counts[arm],"deployed_learned_parameters":counts[arm],
        "steps":steps,"examples_per_step":BATCH,"identical_fit_seed_across_arms":seed,
        "synthetic_seen_examples":steps*BATCH,"synthetic_per_length_512_sealed_test":True,
        "train_loss_samples":outcomes,"scores":score,
        "partial_supported_operator_flops_at_batch2":{str(d):op_profiler(arm,d) for d in LENGTHS},
        "wall_seconds":time.perf_counter()-start,
        "limits":["NO real data; no real-world intelligence claim","No matched full hardware FLOPs guarantee",
            "A negative diagnostic does NOT logically disprove every associative memory architecture.",
            "A positive diagnostic cannot establish real-world forecasting superiority."],
    }
    RESULT_DIR.mkdir(parents=True,exist_ok=True)
    folder=RESULT_DIR/f"seed-{seed}";folder.mkdir(parents=True,exist_ok=True)
    destination=folder/(arm+".pt")
    if destination.exists():raise RuntimeError("original synthetic weights already committed; refuse retraining")
    torch.save({"original_synthetic_train_only":True,
               "weights":m.state_dict(),"optimizer":optimizer.state_dict(),
               "rng":torch.get_rng_state(),"spec":output},destination)
    digest=hashlib.sha256(destination.read_bytes()).hexdigest()
    output["original_checkpoint_sha256"]=digest
    (folder/(arm+".json")).write_text(json.dumps(output,indent=2,sort_keys=True)+"\n")
    print("SYNTHETIC_DIAGNOSTIC_ONLY="+json.dumps({
        "arm":arm,"seed":seed,"test":score,"checkpoint_sha256":digest}),flush=True)
    return output

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,default=75001)
    p.add_argument("--arm",choices=ARMS,default="tac")
    p.add_argument("--mode",choices=("smoke","fit"),default="smoke")
    args=p.parse_args()
    if args.mode=="smoke":
        print("ACTUAL_100K_COUNTS="+json.dumps(model_counts(),sort_keys=True))
        for arm in ARMS:
            m=model_for(arm)
            t,k,v,y=batch_for(args.seed,0,batch_size=2,length=16)
            pred=m(t,k,v);loss=F.cross_entropy(pred,y);loss.backward()
            if not bool(torch.isfinite(pred).all()):raise ValueError("nonfinite diagnostic outputs")
            if any(p.grad is None for p in m.parameters() if p.requires_grad):
                raise ValueError("inactive trained parameter in diagnostic model")
            print(f"PASS_DIAG_SMOKE_{arm}_NO_FIT")
        return
    one_fit(args.seed,args.arm)
if __name__=="__main__":main()
