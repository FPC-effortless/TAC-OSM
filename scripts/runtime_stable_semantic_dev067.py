"""Torch-only public beliefs and learned action scores; no simulator imports."""
import json
import hashlib
from pathlib import Path
import sys
import torch
from agent_stable import StableSemanticAgent as BeliefAgent

torch.set_num_threads(1)
root=Path(__file__).resolve().parent;meta=json.loads((root/'metadata.json').read_text());m=BeliefAgent(meta['spec']['backend'],meta['spec']['width'],meta['spec']['aux'],clock_mode=meta['spec']['clock_mode'],bypass=meta['spec']['bypass'],coordinate=meta['spec']['coordinate']).double().eval()
assert hashlib.file_digest((root/'agent.pt').open('rb'),'sha256').hexdigest()==meta['checkpoint_sha256'],'Checkpoint integrity mismatch'
m.load_state_dict(torch.load(root/'agent.pt',weights_only=True,map_location='cpu'));state=None

def tensor(value,shape):
    x=torch.tensor(value,dtype=torch.float64)
    if list(x.shape)!=shape or not torch.isfinite(x).all():raise ValueError(f'Finite public tensor {shape} required')
    return x[None]

with torch.inference_mode():
    for line in sys.stdin:
        r=json.loads(line)
        if r.get('reset') or state is None:state=m.initial(tensor(r['keys'],[2,16]))
        for e in r['events']:state=m.observe(state,tensor(e,[46]))
        o=m.decide(state,tensor(r['candidates'],[5,19]),tensor(r['prices'],[2]));p=o['posterior'][0];health=p.reshape(4,3).sum(-1);relation=p.reshape(4,3).sum(-2);index=int(m.choice(o)[0]);print(json.dumps({'index':index,'scores':o['logits'][0].tolist(),'joint_health_relation_probabilities':p.tolist(),'world_healthy_probability':float(health[0]),'entity_fault_probabilities':[float(health[2]+health[3]),float(health[1]+health[3])],'relation_probabilities':relation.tolist(),'action_outcome_probabilities':o['future'][0].softmax(-1).tolist(),'normalized_cost_scores':o['cost'][0].tolist()},allow_nan=False),flush=True)
