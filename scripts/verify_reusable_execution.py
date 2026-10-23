"""No-fit independent end-to-end real-input verification, with no retiming."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from scripts.me080_train_world import integrity,frozen,load_original,evaluation_origins,batch,OUT,SEEDS,sha
from tac_osm.variants.streaming_mixture_world import StreamingMixtureWorld,quantile
from tac_osm.variants.reusable_execution import BatchedMixtureWorld,mixture_quantiles


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cache',type=Path,default=Path('/tmp/me077-original-source'));parser.add_argument('--output',type=Path);args=parser.parse_args()
    torch.set_num_threads(1);integrity();frozen()
    _,_,report,identified,evidence=load_original(args.cache);queries,_=evaluation_origins(report,identified)
    models=[]
    with torch.no_grad():
        for seed in SEEDS:
            path=OUT/f'seed-{seed}'/'final.pt';weights=torch.load(path,map_location='cpu',weights_only=True)['weights']
            reference=StreamingMixtureWorld(identified).eval();optimized=BatchedMixtureWorld(identified).eval()
            reference.load_state_dict(weights,strict=True);optimized.load_state_dict(weights,strict=True)
            fields={};bad={};count=0
            for i in range(0,len(queries),24):
                x,a,_,_=batch(report,queries[i:i+24]);old=reference(x,a);allnew=optimized(x,a);last=optimized.forecast_last(x,a)
                values={**{'all/'+k:(old[k],allnew[k]) for k in old},
                        **{'last/'+k:(old[k][:,-1],last[k]) for k in old}}
                original=torch.stack([quantile({k:v[:,-1] for k,v in old.items()},p) for p in (.05,.5,.95)])
                values['final_quantiles']=(original,mixture_quantiles(last))
                for label,(expected,actual) in values.items():
                    fields[label]=max(fields.get(label,0.),float((expected-actual).abs().max()))
                    if not torch.allclose(expected,actual,rtol=1e-5,atol=1e-6):bad[label]=bad.get(label,0)+1
                count+=len(x)
            models.append({'seed':seed,'original_checkpoint_sha256':sha(path),'actual_query_windows':count,
                           'maximum_absolute_difference':fields,'failed_batches':bad})
    result={'status':'FAIL' if any(m['failed_batches'] for m in models) else 'PASS','no_fit':True,'no_retiming':True,
        'source_sha256':evidence['original_160_episode_source_sha256'],'models':models,'rtol':1e-5,'atol':1e-6,
        'scope':'same-host optimized-vs-frozen-reference execution, NOT remediation of original ME080 cross-platform replay exception'}
    if args.output:args.output.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print('TAC_REUSE_VERIFICATION='+json.dumps(result,sort_keys=True))
    if result['status']!='PASS':raise RuntimeError('Execution contract failed; original outputs/tolerance unchanged')

if __name__=='__main__':main()
