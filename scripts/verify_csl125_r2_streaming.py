"""Full-width no-fit CSL-125-R2 causal/streaming equivalence receipt."""
from __future__ import annotations
import hashlib,json,resource,time
from pathlib import Path
import torch
from tac_osm.csl125_r2 import CSL125R2
ROOT=Path(__file__).resolve().parents[1]

def run():
    torch.set_num_threads(1);torch.manual_seed(125209)
    t=time.perf_counter();model=CSL125R2().eval()
    a=torch.randint(0,32768,(1,24));b=a.clone();b[:,16:]=torch.randint(0,32768,(1,8))
    output={'stage':'CSL125R2_FULL_WIDTH_NO_FIT_STREAMING','model_source_sha256':hashlib.sha256((ROOT/'src/tac_osm/csl125_r2.py').read_bytes()).hexdigest(),
        'model_parameters':model.parameter_count(),'device':'cpu','precision':'FP32','threads':1,
        'input_length':24,'chunks':[7,6,11],'future_mutation_start':16,'optimizer_steps':0,'rungs':{},
        'claims':'causal source-local numerical checks only, not global information-flow proof or trained recall'}
    with torch.no_grad():
        for depth in (2,8,20):
            full,state=model(a,depth=depth,stream_ids=('owner',))
            future,_=model(b,depth=depth,stream_ids=('owner',))
            chunks=[];current=None
            for x,y in ((0,7),(7,13),(13,24)):
                chunk,current=model(a[:,x:y],depth=depth,state=current,stream_ids=('owner',))
                chunks.append(chunk[depth]['token_logits'])
            glued=torch.cat(chunks,1)
            max_diff=(full[depth]['token_logits']-glued).abs().max().item()
            future_diff=(full[depth]['token_logits'][:,:16]-future[depth]['token_logits'][:,:16]).abs().max().item()
            assert max_diff<3e-4 and future_diff==0,(depth,max_diff,future_diff)
            output['rungs'][str(depth)]={'chunk_vs_full_max_logit_diff':max_diff,
                'future_token_poison_max_previous_logit_diff':future_diff,
                'status':'PASS'}
    output['status']='PASS_ALL_EXITS'
    output['elapsed_seconds']=round(time.perf_counter()-t,4)
    output['rss_peak_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    path=ROOT/'research/mathematical-engineering/csl125_r2/FULL_STREAMING_NO_FIT.json'
    path.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({d:row['chunk_vs_full_max_logit_diff'] for d,row in output['rungs'].items()},indent=2))
    return output
if __name__=='__main__':run()
