"""Read-only identical-runtime scores for reused controls and new checkpoints.

Original remote evaluation stays untouched. Cross-runtime replay tolerances
are explicit, and all comparison cohorts are recomputed by the same process.
"""
import argparse,json
from pathlib import Path
import torch
from .common import ROOT,build,digest,require,verify_data
from .assays import retrieval,continuation
from collect012.train import fingerprint
from collect012.evaluate import score

def evaluate(folder,output,control):
    torch.set_num_threads(1);r=json.loads((folder/'FIT_RECEIPT.json').read_text());ck=folder/r['checkpoint']
    require(r['optimizer_updates']==1024 and digest(ck)==r['checkpoint_sha256'],'checkpoint differs')
    if control:
        from collect012.common import build as old
        m=old('hybrid_hard_p',r['seed'])
    else:m=build(r['arm'],r['seed'])
    m.load_state_dict(torch.load(ck,weights_only=True,map_location='cpu')['model']);m.eval();before=fingerprint(m)
    replay=torch.load(folder/'REPLAY.pt',weights_only=True,map_location='cpu')
    with torch.inference_mode():actual,_=m(replay['tokens'])
    error=float((actual-replay['logits']).abs().max());torch.testing.assert_close(actual,replay['logits'],atol=2e-5,rtol=2e-5)
    data,_=verify_data();counts={};handles=[]
    for i,b in enumerate(m.core.blocks):
        def record(module,args,out,i=i):
            c=torch.bincount(out.argmax(-1).flatten(),minlength=4).cpu();counts[i]=counts.get(i,torch.zeros_like(c))+c
        handles.append(b.phase_memory.router.register_forward_hook(record))
    metrics={'normal':score(m,data['dev'],'normal')}
    for h in handles:h.remove()
    for mode in ('memory_zero','memory_batch_shuffle'):metrics[mode]=score(m,data['dev'],mode)
    with torch.inference_mode():structured=retrieval(m);carry=continuation(m)
    output.mkdir(exist_ok=True,parents=True)
    (output/'COMMON_RETRIEVAL.json').write_text(json.dumps(structured,indent=2));(output/'COMMON_CONTINUATION.json').write_text(json.dumps(carry,indent=2))
    require(before==fingerprint(m),'common evaluation changed weights')
    result=dict(variant='control' if control else r['arm'],seed=r['seed'],torch=torch.__version__,
        checkpoint_sha256=digest(ck),final_dev=metrics,weights_unchanged=True,optimizer_updates=0,
        routing=[dict(layer=i,counts=c.tolist(),total=int(c.sum()),largest_fraction=float(c.max()/c.sum())) for i,c in sorted(counts.items())],
        cross_runtime_replay_maximum_error=error,cross_runtime_replay_tolerance=2e-5,
        status='COMMON_RUNTIME_READ_ONLY_EXPOSED_DEV_DIAGNOSTICS',generation='original registered generation retained; not repeated here')
    (output/'COMMON_EVALUATION.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--control',action='store_true');a=p.parse_args();evaluate(a.run,a.output,a.control)

if __name__=='__main__':main()
