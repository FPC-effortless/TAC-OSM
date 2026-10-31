"""Actual authorized FIT forward/backward readiness, exactly zero optimizer steps."""
import argparse
from pathlib import Path
import sys
import time
import torch
import importlib.util
_spec=importlib.util.spec_from_file_location('langtrain099_common',Path(__file__).with_name('common.py'))
C=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=C;_spec.loader.exec_module(C)

def run(args):
    reg=C.read(C.HERE/'REGISTER.json');source=C.source_hashes(args.source_commit)
    if args.output.exists():raise FileExistsError('retain original profiles')
    args.output.mkdir(parents=True);corpus,windows,episodes,manifest=C.load_fit(args.data,reg);rows=[]
    for seed in C.SEEDS:
        C.setup(seed);model=C.make(seed);before=C.state_digest(model);times={};diagnostics={}
        plan=C.schedule(seed,4,windows,episodes)
        for mode,number in [('language',0),('episodic',3)]:
            measured=[]
            for repeat in range(3):
                model.zero_grad(set_to_none=True);values=C.batch(plan[number],corpus,windows,episodes)
                started=time.perf_counter();loss=C.loss(model,values);loss.backward();diag=C.gradients(model);elapsed=time.perf_counter()-started
                if not bool(torch.isfinite(loss)) or not all(v>0 for v in diag.values()):raise ValueError('prefit gradient readiness failed')
                measured.append(elapsed);diagnostics[mode]={'loss':float(loss.detach()),'parameter_gradient_l2':diag,'exposure':values[-1]}
            times[mode]=max(measured[1:])
        if C.state_digest(model)!=before:raise ValueError('no-fit profile changed original weights')
        model.zero_grad(set_to_none=True);p=args.output/f'INIT_{seed}.pt'
        init=C.checkpoint(p,model,None,{'seed':seed,'optimizer_steps':0,'counts':model.counts(),'model_sha256':before})
        estimated=reg['training']['steps']*(.75*times['language']+.25*times['episodic'])*1.75+120
        row={'seed':seed,'counts':model.counts(),'model_sha256':before,'initial_archive_sha256':init,
             'initial_archive_file':p.name,'forward_backward_seconds':times,'diagnostics':diagnostics,
             'estimated_training_seconds_including75pct_margin_and120s':estimated,'optimizer_steps':0,'source_hashes':source,'hardware':C.hardware()}
        rows.append(row);C.save(args.output/f'PROFILE_{seed}.json',row)
        del model
    result={'source_commit':args.source_commit,'profiles':rows,'optimizer_steps':0,
            'fits_budget':all(r['estimated_training_seconds_including75pct_margin_and120s']<=reg['training']['max_seconds'] for r in rows)}
    C.save(args.output/'PROFILE_INDEX.json',result)
    print({'profiles':3,'optimizer_steps':0,'fits_budget':result['fits_budget']})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--source-commit',required=True);run(p.parse_args())
