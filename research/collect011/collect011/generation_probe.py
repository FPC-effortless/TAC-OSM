"""Long autoregressive execution stability on FRESH UNTRAINED weights.

Does not measure text quality, language loss or retrieval learning.
"""
import argparse,hashlib,json,time
from pathlib import Path
import torch
from .model import CollectiveTAC,FFN_DIM as FFN

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arms',nargs='+',choices=CollectiveTAC.ARMS,default=list(CollectiveTAC.ARMS))
    parser.add_argument('--output',default='GENERATION_STABILITY');args=parser.parse_args()
    torch.set_num_threads(1);root=Path(__file__).resolve().parents[1]
    prompt=torch.tensor(list((root/'data/train.bin').read_bytes()[20000:20064]),dtype=torch.long)[None]
    results=[]
    for arm in args.arms:
        for temp in (.8,1.2):
            torch.manual_seed(11001);m=CollectiveTAC(arm,ffn_dim=FFN[arm]).eval()
            rng=torch.Generator().manual_seed(11002);generated=[];state_bytes=[]
            maximum=0.;start=time.perf_counter();minimum=float('inf')
            with torch.inference_mode():
                logits,state=m(prompt,owners=('generation',))
                for t in range(512):
                    assert bool(torch.isfinite(logits).all())
                    maximum=max(maximum,float(logits.abs().max()))
                    token=torch.multinomial((logits[:,-1]/temp).softmax(-1),1,generator=rng)
                    generated.append(int(token));logits,state=m.step(token,state=state,owners=('generation',))
                    if t in (255,511):state_bytes.append(dict(generated=t+1,bytes=state.state_bytes()))
                for p in state.phase:
                    if p.precision is not None:minimum=min(minimum,float(torch.linalg.eigvalsh(p.precision).min()))
                assert all(bool(torch.isfinite(t).all()) for s in state.phase for t in (s.memory,s.precision) if t is not None)
            row=dict(arm=arm,temperature=temp,generated_tokens=512,seconds=time.perf_counter()-start,
                maximum_abs_logit=maximum,minimum_precision_eigenvalue=None if minimum==float('inf') else minimum,
                state_bytes=state_bytes,generated_sha256=hashlib.sha256(bytes(generated)).hexdigest(),
                status='PASS_EXECUTION_STABILITY_UNTRAINED',quality='NOT_MEASURED',optimizer_updates=0)
            results.append(row);print(json.dumps(row),flush=True)
            (root/f'results/{args.output}.json').write_text(json.dumps(dict(qualification='untrained execution stability only',rows=results),indent=2))

if __name__=='__main__':main()
