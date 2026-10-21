#!/usr/bin/env python3
"""Numerical portability audit only: no refit, outcome selection or tolerance change."""
import argparse,json,os,subprocess,sys
from pathlib import Path
import numpy as np
import torch
from scripts.me044_train_multiscale import DOMAINS,build_model,predict

torch.set_num_threads(1)

def child(out):
    root=Path(out);mod=DOMAINS['etth1'];z,meta=mod.get_source(root/'etth1')
    folder=root/'etth1'/'exposure-h96-s44001-pyramid_native'
    checkpoint=torch.load(folder/'final.pt',weights_only=True)
    net=build_model('pyramid_native',96,44001,24);net.load_state_dict(checkpoint['state_dict']);net.eval()
    reference=np.load('research/mathematical-engineering/me044/reference-replay.npz')
    with np.load(folder/'DEV.npz') as a:ix=a['origin'][:16]
    fresh,_=mod.samples(z,ix,96)
    result={'CPU_dispatch':torch.backends.cpu.get_cpu_capability(),'torch':torch.__version__,
            'numpy':np.__version__,'original_input_equal':bool(np.array_equal(fresh.numpy(),reference['x'])),
            'from_original_inputs':{},'from_recomputed_inputs':{}}
    with torch.no_grad():
        for name,x in [('from_recomputed_inputs',fresh),('from_original_inputs',torch.from_numpy(reference['x'].copy()))]:
            features=net.features(x);base,tokens,ctx,latent,window,mean,scale=features
            state=net.branch(tokens,latent);inputs=torch.cat((ctx,latent,state),-1)
            coef=net.coefficients(inputs);trust=torch.sigmoid(net.trust(inputs))
            cheap,_=net.cheap_from(features);residual,_,_=net.innovation_from(features)
            p,q,c=predict(net,x)
            arrays=dict(x=x,base=base,tokens=tokens,context=ctx,latent=latent,mean=mean,scale=scale,
                        state=state,coef=coef,trust=trust,cheap=cheap,residual=residual,pred=p)
            for key,v in arrays.items():
                actual=v.numpy();old=reference[key]
                result[name][key]={'equal':bool(np.array_equal(actual,old)),
                    'different':int(np.count_nonzero(actual!=old)),
                    'max_absolute_error':float(np.max(np.abs(actual.astype('float64')-old.astype('float64'))))}
    print(json.dumps(result),flush=True)


def profiles(out):
    choices={'default':{},'mkl_avx2':{'MKL_ENABLE_INSTRUCTIONS':'AVX2'},
        'aten_avx2':{'ATEN_CPU_CAPABILITY':'avx2'},
        'both_avx2':{'MKL_ENABLE_INSTRUCTIONS':'AVX2','ATEN_CPU_CAPABILITY':'avx2'},
        'cbwr_avx2':{'MKL_CBWR':'AVX2','ATEN_CPU_CAPABILITY':'avx2'},
        'compatible':{'MKL_CBWR':'COMPATIBLE','ATEN_CPU_CAPABILITY':'default'}}
    if 'avx512f' in Path('/proc/cpuinfo').read_text():
        choices['mkl_avx512']={'MKL_ENABLE_INSTRUCTIONS':'AVX512'}
    rows=[]
    for name,settings in choices.items():
        p=subprocess.run([sys.executable,__file__,'--out',out,'--child'],env=dict(os.environ,**settings),capture_output=True,text=True)
        row={'profile':name,'settings':settings,'returncode':p.returncode}
        if p.returncode==0:row['comparison']=json.loads(p.stdout)
        else:row['failure']=p.stderr[-3000:]
        rows.append(row);print(json.dumps(row),flush=True)
    (Path(out)/'portability.json').write_text(json.dumps({'purpose':'exact replay backend compatibility, not predictive tuning','profiles':rows},indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--child',action='store_true');a=p.parse_args()
    if a.child:child(a.out)
    else:profiles(a.out)
