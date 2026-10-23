"""Read-only complete replay drift audit; never changes the frozen strict gate.

Recompute every original forecast and compare every scalar/vector field under the
ORIGINAL rtol1e-5/atol1e-6. This audit may finish even when strict replay fails;
a successful audit process is NOT a passing numerical replay contract.
"""
import argparse,importlib,json
from pathlib import Path
import numpy as np
import torch


def main(study,cache):
    module=importlib.import_module(f'scripts.{study}_train_world');module.integrity();module.frozen()
    torch.set_num_threads(1)
    _,_,report,identified,evidence=module.load_original(cache);queries,_=module.evaluation_origins(report,identified)
    models=[];overall={}
    for arm in module.ARMS:
        for seed in module.SEEDS:
            folder=module.OUT/arm/f'seed-{seed}';fit=json.loads((folder/'FIT_RECEIPT.json').read_text())
            if module.sha(folder/'final.pt')!=fit['checkpoint_sha256']:raise RuntimeError('original model SHA differs')
            m=module.model(arm,identified);m.load_state_dict(torch.load(folder/'final.pt',map_location='cpu',weights_only=True)['weights'],strict=True)
            recomputed=module.score_one(m,report,identified,queries);original=json.loads((folder/'ORIGINAL_PREDICTIONS.json').read_text())
            if len(original)!=len(recomputed):raise RuntimeError('real origin denominator differs')
            fields={}
            for old,new in zip(original,recomputed):
                for key in old:
                    if isinstance(old[key],(float,list)):
                        a=np.asarray(old[key]);b=np.asarray(new[key]);d=np.abs(a-b);bad=d>(1e-6+1e-5*np.abs(b))
                        f=fields.setdefault(key,{'elements':0,'outside_original_tolerance':0,'max_abs_error':0.})
                        f['elements']+=int(a.size);f['outside_original_tolerance']+=int(bad.sum());f['max_abs_error']=max(f['max_abs_error'],float(d.max()))
                    elif old[key]!=new[key]:raise RuntimeError('case identity/label mismatch')
            for key,f in fields.items():
                g=overall.setdefault(key,{'elements':0,'outside_original_tolerance':0,'max_abs_error':0.})
                g['elements']+=f['elements'];g['outside_original_tolerance']+=f['outside_original_tolerance'];g['max_abs_error']=max(g['max_abs_error'],f['max_abs_error'])
            delta={str(h):float(np.mean([n['mae']-o['mae'] for o,n in zip(original,recomputed) if o['horizon']==h])) for h in module.HORIZONS}
            models.append({'architecture':arm,'seed':seed,'original_checkpoint_sha256':fit['checkpoint_sha256'],'raw_forecasts':len(original),'fields':fields,'horizon_mean_MAE_drift':delta})
    bad=sum(f['outside_original_tolerance'] for f in overall.values())
    receipt={'study':study.upper(),'strict_original_contract_passed':bad==0,'original_rtol':1e-5,'original_atol':1e-6,
      'all_original_models':len(models),'total_elements_outside_original_contract':bad,'aggregate_fields':overall,'models':models,
      'no_refit':True,'no_tolerance_change':True,'source_sha256':evidence['original_160_episode_source_sha256']}
    module.write(module.OUT/'CROSS_CPU_REPLAY_AUDIT.json',receipt)
    print(study.upper()+'_CROSS_CPU_AUDIT='+json.dumps(receipt,sort_keys=True),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',choices=['me078','me079'],required=True);p.add_argument('--cache',type=Path,required=True);a=p.parse_args();main(a.study,a.cache)
