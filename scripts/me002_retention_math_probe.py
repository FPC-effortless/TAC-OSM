#!/usr/bin/env python3
"""ME-002: native TAC retention-path numerical tests.
Unit diagnostics with random inputs ONLY; not forecasting, entity discovery, or SSM.
"""
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

SEEDS=(20261019,20261020,20261021,20261022,20261023)
LENGTHS=(8,32,128,512)
WIDTH=12

class RetentionCandidate(nn.Module):
    def __init__(self,mode):
        super().__init__()
        if mode not in ("dissipative_base","identity_first","two_timescale"):
            raise ValueError(mode)
        self.mode=mode
        self.U=nn.Linear(WIDTH,WIDTH,bias=False)
        self.V=nn.Linear(WIDTH,WIDTH,bias=False)
        self.write=nn.Linear(2*WIDTH,1)
        nn.init.constant_(self.write.bias,0.0 if mode=="dissipative_base" else -4.0)

    def forward(self,h,x):
        g=torch.sigmoid(self.write(torch.cat((x,h),dim=-1)))
        proposal=torch.tanh(self.U(h)+self.V(x))
        if self.mode=="dissipative_base":
            return (1-g)*(.95*h)+g*proposal
        if self.mode=="identity_first":
            return (1-g)*h+g*proposal
        middle=WIDTH//2
        gf=g
        gs=.015*torch.ones_like(g)  # predeclared constant, not learned skill
        fast=(1-gf)*h[:,:middle]+gf*proposal[:,:middle]
        slow=(1-gs)*h[:,middle:]+gs*proposal[:,middle:]
        return torch.cat((fast,slow),dim=-1)

def run():
    torch.set_num_threads(1)
    records=[]
    for seed in SEEDS:
        torch.manual_seed(seed)
        inputs=torch.randn(max(LENGTHS),1,WIDTH)*.15
        initial=torch.randn(1,WIDTH)*.15
        direction=F.normalize(torch.randn(1,WIDTH),dim=-1)
        for mode in ("dissipative_base","identity_first","two_timescale"):
            # Reseed ensures U,V and all weights other than declared biases match.
            torch.manual_seed(seed+10000)
            cell=RetentionCandidate(mode)
            for length in LENGTHS:
                h0=initial.clone().detach().requires_grad_(True)
                h=h0
                for x in inputs[:length]:
                    h=cell(h,x)
                    if not torch.isfinite(h).all() or h.abs().max()>1.000001:
                        raise AssertionError("nonfinite or out-of-bounds state")
                grad,=torch.autograd.grad((h*direction).sum(),h0)
                if not torch.isfinite(grad).all():
                    raise AssertionError("nonfinite gradient")
                records.append({"seed":seed,"mode":mode,"length":length,
                                "initial_state_gradient_l2":float(grad.norm()),
                                "final_state_max_abs":float(h.abs().max()),
                                "valid":True})
    if len(records)!=60:
        raise AssertionError("incomplete seed/length/mode matrix")
    for L in LENGTHS:
        base=[r["initial_state_gradient_l2"] for r in records
              if r["mode"]=="dissipative_base" and r["length"]==L]
        for variant in ("identity_first","two_timescale"):
            alt=[r["initial_state_gradient_l2"] for r in records
                 if r["mode"]==variant and r["length"]==L]
            if not all(a>b for a,b in zip(alt,base)):
                raise AssertionError(f"gradient ordering failed: {variant}@{L}")
    return {"classification":"mathematical/software diagnostics only",
            "seed_count":5,"records":records,
            "not_trained":True,"real_world_performance_claim":False}

if __name__=="__main__":
    result=run()
    p=Path("results/me002")
    p.mkdir(parents=True,exist_ok=True)
    (p/"gradients.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    for name in ("dissipative_base","identity_first","two_timescale"):
        print(name,[(L, sum(r["initial_state_gradient_l2"] for r in result["records"]
                              if r["mode"]==name and r["length"]==L)/5)
                    for L in LENGTHS])
    print("ME002 PASS numerical gradients; not a trained-world-model result")
