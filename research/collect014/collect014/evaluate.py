"""Fixed final-checkpoint diagnostics, with immutable reused controls."""
import argparse,hashlib,json
from pathlib import Path
import torch
from .common import ROOT,build,verify_data,require,digest
from .assays import retrieval,continuation
from collect012.train import fingerprint,batch
from collect012.evaluate import score

def evaluate(folder,output,control=False):
    torch.set_num_threads(1);receipt=json.loads((folder/'FIT_RECEIPT.json').read_text())
    require(receipt['status']=='FULL_FIT_COMPLETE_INTERNAL_UNQUALIFIED' and receipt['optimizer_updates']==1024,'incomplete fit')
    checkpoint=folder/receipt['checkpoint'];require(digest(checkpoint)==receipt['checkpoint_sha256'],'checkpoint identity changed')
    saved=torch.load(checkpoint,weights_only=True,map_location='cpu');variant='control' if control else receipt['arm']
    if control:
        from collect012.common import build as old_build
        model=old_build('hybrid_hard_p',receipt['seed'])
    else:model=build(variant,receipt['seed'])
    model.load_state_dict(saved['model'],strict=True);model.eval();before=fingerprint(model)
    replay=torch.load(folder/'REPLAY.pt',weights_only=True,map_location='cpu')
    with torch.inference_mode():logits,_=model(replay['tokens'])
    replay_error=float((logits-replay['logits']).abs().max())
    # New remote fits must replay bitwise; old remote controls may be audited in
    # a newer local runtime, with that difference explicit and tolerant.
    if control:torch.testing.assert_close(logits,replay['logits'],atol=2e-5,rtol=2e-5)
    else:require(torch.equal(logits,replay['logits']),'fresh-process replay differs')
    data,_=verify_data();output.mkdir(parents=True,exist_ok=True)
    counts={};hooks=[]
    for i,b in enumerate(model.core.blocks):
        def record(module,args,out,i=i):
            c=torch.bincount(out.argmax(-1).flatten(),minlength=4).cpu();counts[i]=counts.get(i,torch.zeros_like(c))+c
        hooks.append(b.phase_memory.router.register_forward_hook(record))
    metrics={'normal':score(model,data['dev'],'normal')}
    for h in hooks:h.remove()
    for mode in ('memory_zero','memory_batch_shuffle'):metrics[mode]=score(model,data['dev'],mode)
    routing=[dict(layer=i,counts=c.tolist(),total=int(c.sum()),largest_fraction=float(c.max()/c.sum()),active_banks=int((c>0).sum())) for i,c in sorted(counts.items())]
    with torch.inference_mode():
        structure=retrieval(model);carry=continuation(model)
    (output/'DELAYED_RETRIEVAL.json').write_text(json.dumps(structure,indent=2))
    (output/'CONTINUATION.json').write_text(json.dumps(carry,indent=2))
    generations=[]
    for temp in (.8,1.2):
        rng=torch.Generator().manual_seed(140100);tokens=[];sizes=[]
        with torch.inference_mode():
            y,state=model(torch.tensor(list(data['train'][20000:20064]))[None],owners=('generation',))
            for i in range(512):
                require(bool(torch.isfinite(y).all()),'nonfinite generation')
                nxt=torch.multinomial((y[:,-1]/temp).softmax(-1),1,generator=rng)
                tokens.append(int(nxt));y,state=model.step(nxt,state=state,owners=('generation',))
                if i in (255,511):sizes.append(state.state_bytes())
        raw=bytes(tokens);(output/f'generation_temperature_{temp}.bin').write_bytes(raw)
        (output/f'generation_temperature_{temp}.txt').write_text(raw.decode('utf-8',errors='replace'))
        generations.append(dict(temperature=temp,bytes=512,sha256=hashlib.sha256(raw).hexdigest(),state_tensor_bytes=sizes))
    require(before==fingerprint(model),'evaluation changed trained weights')
    result=dict(variant=variant,seed=receipt['seed'],checkpoint_sha256=digest(checkpoint),
        torch=torch.__version__,reused_control=control,replay_maximum_error=replay_error,replay_exact=replay_error==0,
        final_dev=metrics,routing=routing,generation=generations,weights_unchanged=True,optimizer_updates=0,
        status='INTERNALLY_MEASURED_FINAL_CHECKPOINT_NOT_COMPUTE_QUALIFIED',
        structured_task='zero-shot generated diagnostic, not trained retrieval utility',
        dev_cohort='same 64 C12 windows, 8192 shared byte targets; continuation is a separate 8-window cohort')
    (output/'EVALUATION.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--control',action='store_true');a=p.parse_args()
    evaluate(a.run,a.output,a.control)

if __name__=='__main__':main()
