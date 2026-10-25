"""Read-only full external numerical census; NEVER changes strict gate/results."""
import argparse,json,platform
from pathlib import Path
import numpy as np
import torch
import scripts.me087_train_world as R
from scripts.me087_train_world import integrity,frozen,load_original,evaluation_origins,sha,score_one,long_queries
from tac_osm.variants.bounded_value_world import BoundedValueWorld


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',choices=tuple(R.CONFIG),default='me087');parser.add_argument('--cache',type=Path,default=Path('/tmp/me077-original-source'));args=parser.parse_args();R.configure(args.study);OUT,SEEDS=R.OUT,R.SEEDS
    torch.set_num_threads(1);integrity();frozen()
    _,_,report,identified,evidence=load_original(args.cache);queries,_=evaluation_origins(report,identified)
    records=[];totals={};maximum={};totalrows=0
    for history in (8,32):
        for seed in SEEDS:
            folder=OUT/f'seed-{seed}';model=BoundedValueWorld(identified)
            model.load_state_dict(torch.load(folder/'final.pt',map_location='cpu',weights_only=True)['weights'],strict=True);model.double()
            computed=score_one(model,report,identified,queries if history==8 else long_queries(report,queries),history);original=json.loads((folder/('ORIGINAL_PREDICTIONS.json' if history==8 else 'ORIGINAL_PREDICTIONS_H32.json')).read_text())
            if len(computed)!=len(original):raise RuntimeError('Denominator differs')
            failures={};maxerrors={};examples=[]
            for old,new in zip(original,computed):
                for key in old:
                    if isinstance(old[key],(float,list)):
                        aa=np.asarray(old[key]);bb=np.asarray(new[key]);error=float(np.max(np.abs(aa-bb)))
                        maxerrors[key]=max(maxerrors.get(key,0.),error)
                        if not np.allclose(aa,bb,rtol=1e-5,atol=1e-6):
                            failures[key]=failures.get(key,0)+1
                            if len(examples)<8:examples.append({'task':old['task'],'episode':old['episode'],'frame':old['frame'],'horizon':old['horizon'],'field':key,'maximum_abs_difference':error})
                    elif old[key]!=new[key]:raise RuntimeError('Identity/label differs')
            for k,v in failures.items():totals[k]=totals.get(k,0)+v
            for k,v in maxerrors.items():maximum[k]=max(maximum.get(k,0.),v)
            totalrows+=len(original);records.append({'seed':seed,'observed_history':history,'checkpoint_sha256':sha(folder/'final.pt'),'rows':len(original),'outside_original_tolerance_per_field':failures,'maximum_absolute_difference_per_field':maxerrors,'examples':examples})
    result={'study':R.STUDY,'audit_completed':True,'strict_original_contract':'FAIL' if totals else 'PASS','no_fit':True,
       'original_models':5,'model_history_conditions':len(records),'original_forecasts':totalrows,'rtol':1e-5,'atol':1e-6,
       'outside_original_tolerance_per_field':totals,'maximum_absolute_difference_per_field':maximum,'models':records,
       'source_sha256':evidence['original_160_episode_source_sha256'],
       'environment':{'python':platform.python_version(),'torch':torch.__version__,'threads':torch.get_num_threads()},
       'no_original_outputs_weights_or_tolerance_changed':True,
       'scope':'Full original field audit, not a replacement PASS for the failed strict replay. Rows may fail more than one field; no unique-failed-row inference from summed fields.'}
    print('BOUNDED_VALUE_EXTERNAL_FULL_AUDIT='+json.dumps(result,sort_keys=True),flush=True)

if __name__=='__main__':main()
