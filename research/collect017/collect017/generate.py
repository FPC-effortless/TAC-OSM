"""Frozen-model 512-byte continuations, two temperatures, all seeds retained."""
import argparse,json
from pathlib import Path
import torch
from .common import build,verify_data,digest,require,fingerprint
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1)
    r=json.loads((a.run/'FIT_RECEIPT.json').read_text());ck=a.run/r['checkpoint'];require(digest(ck)==r['checkpoint_sha256'],'checkpoint differs')
    m=build(r['arm'],r['seed']);m.load_state_dict(torch.load(ck,weights_only=True,map_location='cpu')['model']);m.eval();initial=fingerprint(m);data,_=verify_data();prompt=data['train'][20000:20064];rows=[]
    for temperature in (.8,1.2):
        generator=torch.Generator().manual_seed(160100);tokens=[];payload=[];error=None
        try:
            with torch.inference_mode():
                z,state=m(torch.tensor([list(prompt)]),owners=('generation',))
                for i in range(512):
                    require(torch.isfinite(z).all(),'nonfinite generation logits')
                    token=torch.multinomial((z[0,-1]/temperature).softmax(-1),1,generator=generator);tokens.append(int(token))
                    z,state=m(token.reshape(1,1),state=state,owners=('generation',))
                    if i+1 in (256,512):payload.append(dict(generated_bytes=i+1,state_tensor_bytes=state.state_bytes()))
        except Exception as exc:error=type(exc).__name__+': '+str(exc)
        raw=bytes(tokens);rows.append(dict(temperature=temperature,rng=160100,generated_bytes=len(raw),hex=raw.hex(),text=raw.decode('utf8','replace'),
            payload=payload,status='COMPLETE' if error is None else 'FAILED_RETAINED',error=error))
    require(initial==fingerprint(m),'generation changed weights')
    a.output.write_text(json.dumps(dict(arm=r['arm'],seed=r['seed'],torch=torch.__version__,checkpoint_sha256=digest(ck),prompt_hex=prompt.hex(),rows=rows,weights_unchanged=True,optimizer_updates=0),indent=2))
    print(json.dumps(dict(arm=r['arm'],seed=r['seed'],statuses=[x['status'] for x in rows])))
if __name__=='__main__':main()
