#!/usr/bin/env python3
"""Optional, registered exact execution optimization of original native models."""
import argparse,copy,json,time
from pathlib import Path
import numpy as np
import torch
from scripts.me044_train_multiscale import DOMAINS,period,predict,save,sha
from tac_osm.me044_multiscale_decoder import build_model

torch.set_num_threads(1)

@torch.no_grad()
def run(out):
    out=Path(out);report=json.loads((out/'results.json').read_text());assert report['completed']
    rows=[];values=0
    for domain,data in report['domains'].items():
        mod=DOMAINS[domain];dest=out/domain;z,_=mod.get_source(dest)
        for row in data['fits']:
            if row['arm']!='pyramid_native':continue
            cp=dest/row['checkpoint'];assert sha(cp)==row['checkpoint_sha256']
            eager=build_model(row['arm'],row['horizon'],row['seed'],period(domain))
            eager.load_state_dict(torch.load(cp,weights_only=True)['state_dict']);eager.eval()
            fast=copy.deepcopy(eager);t=time.process_time()
            fast.branch=torch.jit.script(fast.branch)
            compile_cpu=time.process_time()-t
            export=cp.parent/'scripted-native-branch.pt';torch.jit.save(fast.branch,str(export))
            fast.branch=torch.jit.load(str(export));fast.eval()
            with np.load(cp.parent/'DEV.npz') as saved:
                ix=saved['origin'];ps=[];cs=[];qs=[]
                for j in range(0,len(ix),16):
                    x,_=mod.samples(z,ix[j:j+16],row['horizon'])
                    p,q,c=predict(fast,x);ps.append(p.numpy());qs.append(q.numpy());cs.append(c.numpy())
                actual=np.concatenate(ps)
                if not np.array_equal(actual,saved['pred']):raise ArithmeticError('compiled native full forecasts changed')
                assert np.array_equal(np.concatenate(cs),saved['cheap'])
                assert np.array_equal(np.concatenate(qs),saved['q'])
                values+=actual.size
            x,_=mod.samples(z,[mod.TR],row['horizon'])
            for net in (eager,fast):
                for _ in range(5):net(x)
            times={'eager':[],'scripted':[]}
            for repeat in range(5):
                order=(('eager',eager),('scripted',fast)) if repeat%2==0 else (('scripted',fast),('eager',eager))
                for label,net in order:
                    start=time.process_time()
                    for _ in range(20):net(x)
                    times[label].append((time.process_time()-start)/20)
            speed=float(np.median(times['eager'])/np.median(times['scripted']))
            rows.append({'domain':domain,'horizon':row['horizon'],'seed':row['seed'],'lane':row['lane'],
                'compile_CPU':compile_cpu,'times':times,'speedup':speed,'exact_replay':True,
                'export':str(export.relative_to(out)),'sha256':sha(export)})
    assert len(rows)==16
    speed=float(np.exp(np.mean(np.log([r['speedup'] for r in rows]))))
    result={'status':'EXACT_REPLAY_PASS','rows':rows,'models':16,'forecast_scalars_exact':values,
        'geometric_mean_full_forecast_speedup':speed,'speed_target_1_25_pass':bool(speed>=1.25),
        'new_fits':0,'default_backend_changed':False}
    save(out/'inference.json',result);print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();run(a.out)
