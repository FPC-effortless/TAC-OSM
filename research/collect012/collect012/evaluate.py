"""Frozen postfit diagnostics; no optimization, selection or sealed TEST."""
import argparse,hashlib,json
from pathlib import Path
import torch
from torch.nn import functional as F
from .common import ROOT,build,verify_data,require,digest
from .train import batch,fingerprint

def score(model,data,mode):
    hooks=[];g=torch.Generator().manual_seed(120099);loss=0.;targets=0
    if mode!='normal' and model.arm!='attention':
        def intervene(module,inputs,output):
            read,state=output
            return (torch.zeros_like(read) if mode=='memory_zero' else read.roll(1,0)),state
        hooks=[b.phase_memory.register_forward_hook(intervene) for b in model.core.blocks]
    try:
        with torch.inference_mode():
            for _ in range(16):
                x,y,_=batch(data,g,4,128,'cpu');logits,_=model(x)
                loss+=float(F.cross_entropy(logits.flatten(0,1),y.flatten(),reduction='sum'));targets+=y.numel()
        return dict(nll=loss/targets,targets=targets,status='MEASURED' if mode=='normal' or hooks else 'NOT_APPLICABLE')
    finally:
        for h in hooks:h.remove()

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(1);receipt=json.loads((a.run/'FIT_RECEIPT.json').read_text())
    require(receipt['status']=='FULL_FIT_COMPLETE_INTERNAL_UNQUALIFIED','full fixed budget is incomplete')
    checkpoint=a.run/receipt['checkpoint'];require(digest(checkpoint)==receipt['checkpoint_sha256'],'checkpoint bytes changed')
    saved=torch.load(checkpoint,weights_only=True,map_location='cpu')
    model=build(receipt['arm'],receipt['seed']);model.load_state_dict(saved['model'],strict=True);model.eval()
    replay=torch.load(a.run/'REPLAY.pt',weights_only=True,map_location='cpu')
    with torch.inference_mode():actual,_=model(replay['tokens'])
    require(torch.equal(actual,replay['logits']),'fresh-process CPU replay differs')
    data,_=verify_data();before=fingerprint(model)
    metrics={mode:score(model,data['dev'],mode) for mode in ('normal','memory_zero','memory_batch_shuffle')}
    initial=torch.load(a.run/'checkpoint_0000.pt',weights_only=True,map_location='cpu')
    initial_model=build(receipt['arm'],receipt['seed']);initial_model.load_state_dict(initial['model']);initial_model.eval()
    initial_dev=score(initial_model,data['dev'],'normal')
    generated=[]
    for temp in (.8,1.2):
        rng=torch.Generator().manual_seed(120100);tokens=[];maxlogit=0.
        with torch.inference_mode():
            logits,state=model(torch.tensor(list(data['train'][20000:20064]))[None],owners=('generation',))
            sizes=[]
            for i in range(512):
                require(bool(torch.isfinite(logits).all()),'nonfinite trained generation')
                maxlogit=max(maxlogit,float(logits.abs().max()))
                nxt=torch.multinomial((logits[:,-1]/temp).softmax(-1),1,generator=rng)
                tokens.append(int(nxt));logits,state=model.step(nxt,state=state,owners=('generation',))
                if i in (255,511):sizes.append(state.state_bytes())
        raw=bytes(tokens);file=a.run/f'generation_temperature_{temp}.bin';file.write_bytes(raw)
        (a.run/f'generation_temperature_{temp}.txt').write_text(raw.decode('utf-8',errors='replace'))
        generated.append(dict(temperature=temp,bytes=512,sha256=hashlib.sha256(raw).hexdigest(),maximum_abs_logit=maxlogit,state_tensor_bytes=sizes))
    require(before==fingerprint(model),'evaluation changed weights')
    result=dict(arm=receipt['arm'],seed=receipt['seed'],checkpoint_sha256=receipt['checkpoint_sha256'],
        initial_dev=initial_dev,final_dev=metrics,generation=generated,cold_replay_exact=True,weights_unchanged=True,
        status='INTERNALLY_MEASURED_EXPOSED_DEV_NOT_INDEPENDENTLY_QUALIFIED',optimizer_updates=0,
        limitations='No compute-matched architecture ranking; phase interventions are postfit causal diagnostics, not real alternative outcomes; no long-memory or semantic-binding guarantee.')
    (a.run/'EVALUATION.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
