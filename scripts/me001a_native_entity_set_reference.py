"""ME-001A compact standalone native TAC mathematical-reference control.
Software diagnostic only; not a trained world model, Mamba implementation,
or evidence that entities were semantically discovered. PyTorch CPU-compatible.
"""
import json
import math
import torch
from torch import nn
from torch.nn import functional as F


class NativeEntitySet(nn.Module):
    def __init__(self, dim=4, width=12, slots=4, mode="learned", tau=1.0):
        super().__init__()
        if mode not in ("learned", "known_id") or tau <= 0:
            raise ValueError("invalid routing configuration")
        self.dim, self.width, self.slots, self.mode, self.tau = dim,width,slots,mode,tau
        self.query=nn.Linear(width*2,width,bias=False)
        self.key=nn.Linear(dim*2,width,bias=False)
        self.null=nn.Linear(dim*2,1)
        self.sensory=nn.Linear(dim,width)
        self.recurrent=nn.Linear(width,width,bias=False)
        self.write=nn.Linear(dim+width,1)
        self.update=nn.Linear(width,1)
        self.readout=nn.Linear(width,dim)
        self.raw_decay=nn.Parameter(torch.full((width,),-3.0))

    def initial(self,batch,anchors=None):
        if anchors is None:
            anchors=F.normalize(torch.randn(batch,self.slots,self.width),dim=-1)
        return {"h":torch.zeros_like(anchors),"anchors":anchors}

    def step(self,tokens,feature_mask,valid,dt,state,ids=None):
        # tokens [B,J,D]; h [B,N,H]; causal information only
        b,j,d=tokens.shape
        if d!=self.dim or feature_mask.shape!=tokens.shape or valid.shape!=(b,j) or dt.shape!=(b,):
            raise ValueError("shape mismatch")
        if bool((dt<0).any()): raise ValueError("negative dt")
        visible=feature_mask.bool() & valid.bool().unsqueeze(-1)
        value=torch.where(visible,tokens,torch.zeros_like(tokens))
        if self.mode=="known_id":
            if ids is None or ids.shape!=(b,j): raise ValueError("ids required for control")
            if ((ids < -1)|(ids>=self.slots)).any(): raise ValueError("out-of-range id")
            assn=F.one_hot(ids.clamp(min=0),self.slots).permute(0,2,1).to(value.dtype)
            assn=assn*((ids>=0)&valid.bool()).unsqueeze(1)
            null_mass=valid.float()-assn.sum(1)
        else:
            feature=torch.cat([value,visible.float()],dim=-1)
            queries=self.query(torch.cat([state["h"],state["anchors"]],dim=-1))
            keys=self.key(feature)
            score=torch.einsum("bnh,bjh->bnj",queries,keys)/math.sqrt(self.width)
            score=torch.cat([score,self.null(feature).transpose(1,2)],dim=1)
            probs=F.softmax(score/self.tau,dim=1)*valid[:,None,:]
            assn,null_mass=probs[:,:self.slots],probs[:,self.slots]
        mass=assn.sum(-1)
        inp=torch.einsum("bnj,bjd->bnd",assn,value)/mass.unsqueeze(-1).clamp_min(1e-8)
        obs=(torch.einsum("bnj,bjd->bnd",assn,visible.float())>1e-8)
        decay=torch.exp(-F.softplus(self.raw_decay)[None,None,:]*dt[:,None,None])
        held=decay*state["h"]
        gate=torch.sigmoid(self.update(held))
        write_gate=torch.sigmoid(self.write(torch.cat([inp,held],dim=-1)))
        write_gate=write_gate*obs.any(-1,keepdim=True)
        proposal=torch.tanh(self.recurrent(held)+write_gate*self.sensory(inp))
        next_h=(1-gate)*held+gate*proposal
        assert bool((next_h.abs()<=1.000001).all())
        return self.readout(next_h),{"h":next_h,"anchors":state["anchors"]},{
            "assignment":assn,"null_mass":null_mass,"write":write_gate,
        }


def verify(seed=20261019):
    torch.manual_seed(seed)
    b,n,j,d,w=3,4,5,4,12
    tokens=torch.randn(b,j,d)
    feature_mask=(torch.rand(b,j,d)>0.2).float()
    valid=torch.tensor([[1,1,1,1,0],[1,1,0,0,0],[1,1,1,1,1]],dtype=torch.bool)
    dt=torch.tensor([0.1,0.7,2.0])
    ids=torch.tensor([[2,0,3,1,-1],[1,0,-1,-1,-1],[3,0,1,2,2]])
    outputs={}
    for mode in ("learned","known_id"):
        model=NativeEntitySet(d,w,n,mode)
        state=model.initial(b)
        y,ns,a=model.step(tokens,feature_mask,valid,dt,state,ids)
        tperm=torch.tensor([2,4,0,1,3])
        yy,ss,aa=model.step(tokens[:,tperm],feature_mask[:,tperm],valid[:,tperm],dt,state,ids[:,tperm])
        token_error=max((y-yy).abs().max().item(),(ns["h"]-ss["h"]).abs().max().item(),
                        (a["assignment"]-aa["assignment"][:,:,torch.argsort(tperm)]).abs().max().item())
        sperm=torch.tensor([2,0,3,1])
        newids=torch.where(ids<0,-1,torch.argsort(sperm)[ids.clamp_min(0)])
        rotated={"h":state["h"][:,sperm],"anchors":state["anchors"][:,sperm]}
        yr,sr,ar=model.step(tokens,feature_mask,valid,dt,rotated,newids)
        slot_error=max((ns["h"][:,sperm]-sr["h"]).abs().max().item(),
                       (y[:,sperm]-yr).abs().max().item(),
                       (a["assignment"][:,sperm]-ar["assignment"]).abs().max().item())
        corrupted=tokens.clone()
        corrupted[~feature_mask.bool()]+=1e5
        corrupted[~valid]+=1e5
        ybad,sbad,abad=model.step(corrupted,feature_mask,valid,dt,state,ids)
        leak=max((y-ybad).abs().max().item(),(ns["h"]-sbad["h"]).abs().max().item(),
                 (a["assignment"]-abad["assignment"]).abs().max().item())
        _,_,empty=model.step(tokens,feature_mask,torch.zeros_like(valid),dt,state,ids)
        zero={"h":torch.zeros_like(state["h"]),"anchors":state["anchors"]}
        ycarry,_,_=model.step(tokens,feature_mask,valid,dt,ns,ids)
        yreset,_,_=model.step(tokens,feature_mask,valid,dt,zero,ids)
        effect=(ycarry-yreset).abs().max().item()
        (y.square().mean()+0.1*ns["h"].square().mean()).backward()
        grad=all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
        assert token_error<1e-5 and slot_error<1e-5 and leak==0 and grad and effect>1e-8
        assert empty["write"].abs().max().item()==0
        outputs[mode]={"token_error":token_error,"slot_error":slot_error,
                       "leak":leak,"state_intervention":effect,"grads_finite":grad}
        if mode=="learned":
            assert model.query.weight.grad.abs().max().item()>0
            impostor_ids=torch.full_like(ids,-1)
            ya,_,_=model.step(tokens,feature_mask,valid,dt,state,impostor_ids)
            assert (ya-y).abs().max().item()==0
            same=model.initial(b,anchors=torch.zeros_like(state["anchors"]))
            _,collapsed,_=model.step(tokens,feature_mask,valid,dt,same,ids)
            assert (collapsed["h"]-collapsed["h"][:,:1]).abs().max().item()==0
            assert (ns["h"]-ns["h"][:,:1]).abs().max().item()>0
    return outputs


if __name__=="__main__":
    print(json.dumps({"status":"engineering diagnostics only",
                      "checks":{str(seed):verify(seed) for seed in range(20261019,20261029)}},
                     indent=2,sort_keys=True))
