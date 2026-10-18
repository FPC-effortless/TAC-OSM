"""Torch-only JSON-lines public interface for a DEV061.5 research checkpoint."""
import json
import hashlib
from pathlib import Path
import sys
import torch
from agent_model import IdentityAgent,choose

torch.set_num_threads(1)

root=Path(__file__).resolve().parent
meta=json.loads((root/'metadata.json').read_text())
model=IdentityAgent(meta['arm'],meta['spec']['width'],meta['spec']['aux_width']).double().eval()
assert hashlib.file_digest((root/'agent.pt').open('rb'),'sha256').hexdigest()==meta['checkpoint_sha256'],'Checkpoint integrity mismatch'
model.load_state_dict(torch.load(root/'agent.pt',weights_only=True,map_location='cpu'),strict=True)
state=None

def tensor(value,shape,dtype=torch.float64):
    x=torch.tensor(value,dtype=dtype)
    if list(x.shape)!=shape:raise ValueError(f'Expected shape {shape}, received {list(x.shape)}')
    if x.is_floating_point() and not torch.isfinite(x).all():raise ValueError('Nonfinite public input')
    return x[None]

with torch.inference_mode():
    for line in sys.stdin:
        request=json.loads(line)
        if request.get('reset') or state is None:
            keys=tensor(request['keys'],[6,16]);mask=tensor(request['entity_valid'],[6],torch.bool)
            if not mask.any():raise ValueError('Empty public entity manifest')
            state=model.initial(keys,mask)
        for record in request['events']:state,_=model.observe(state,tensor(record,[38]))
        actions=tensor(request['candidates'],[13,19]);valid=tensor(request['valid'],[13],torch.bool)
        if not valid.any():raise ValueError('Empty choice set')
        out=model.decide(state,actions,valid);choice=int(choose(out['logits'],valid)[0])
        print(json.dumps({'index':choice,'scores':out['logits'][0].tolist(),'completion_probability':float(out['completion'][0].sigmoid())},allow_nan=False),flush=True)
