"""Post-null non-neural frequency checks on the exact observed TRAIN targets.

These descriptive count controls are not matched neural architecture rivals.
They add no extra source examples, optimizer steps, or protected evaluation.
"""
import json
from collections import Counter
import torch
from .common import ROOT,SEEDS,verify_data
from .data import pools,training_batch
from collect014.train import batch
def main():
    torch.set_num_threads(1);data,_=verify_data();pool=pools(data['train']);rows=[]
    g=torch.Generator().manual_seed(120099);targets=[]
    for _ in range(16):
        _,y,_=batch(data['dev'],g,4,128,'cpu');targets.append(y.flatten())
    targets=torch.cat(targets)
    for seed in SEEDS:
        natural=torch.zeros(256,dtype=torch.float64);weighted=natural.clone()
        for step in range(1,513):
            _,y,mask,_=training_batch(data['train'],pool,seed=seed,step=step)
            a=torch.bincount(y[0],minlength=256).double();b=torch.bincount(y[1],minlength=256).double();c=torch.bincount(y[mask],minlength=256).double()
            natural+=a;weighted+=a+.25*b+96*c
        for name,counts in [('natural_targets_only',natural),('joint_objective_weighted',weighted)]:
            probability=(counts+1)/(counts.sum()+256)
            rows.append(dict(seed=seed,control=name,ordinary_dev_nll=float(-probability[targets].log().mean()),
                argmax_byte=int(probability.argmax()),argmax_character=chr(int(probability.argmax())),counts=counts.tolist(),
                dev_targets=len(targets),smoothing='one pseudocount per byte, fixed without a sweep'))
    out=dict(status='COMPLETE_POST_NULL_CLASSICAL_DIAGNOSTIC',rows=rows,optimizer_updates=0,neural_models_fit=0,
        source='Exact already-presented C16/C17 TRAIN target schedule; same exposed 8192 DEV targets',
        weighted_definition='Natural target count + .25 task target count + 96 answer-target count (scalar objective multiplied by384)',
        caution='Unmatched classical practical baselines; neither source authentication nor matched neural-architecture superiority. Post-null analysis, no model selection.')
    (ROOT/'results/MARGINAL_CONTROLS.json').write_text(json.dumps(out,indent=2));print(json.dumps([{k:v for k,v in r.items() if k!='counts'} for r in rows],indent=2))
if __name__=='__main__':main()
