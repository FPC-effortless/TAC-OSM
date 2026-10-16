#!/usr/bin/env python3
"""Exercise pinned PR146 in an isolated checkout; never merge its modules.

This is an integration inventory check, not a new capability experiment.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

PIN='5923ed0bf1aa2b9ae125c98d60dab25ec3cf07aa'


def check(checkout):
    root=Path(checkout).resolve()
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    if sha!=PIN or subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip():
        raise ValueError('requires clean pinned PR146 checkout')
    sys.path.insert(0,str(root/'src'))
    from tac_osm.fusion.builder import build_fused_model
    from tac_osm.fusion.ablation import full_config
    built=build_fused_model(full_config(seed=0),n_steps=24)
    model=built.model;calls={}
    boundaries={'memory':('context','observe'),'regime':('context','update'),
        'coordinator':('route','update'),'operators':('execute_modules','select_candidate','update'),
        'verifier':('verify','commit_allowed'),'repair':('propose',),
        'environment':('transition',),'state':('read','write')}
    for component,methods in boundaries.items():
        for method in methods:
            obj=getattr(model,component);original=getattr(obj,method);key=component+'.'+method
            def wrapped(*args,_original=original,_key=key,**kwargs):
                calls[_key]=calls.get(_key,0)+1
                return _original(*args,**kwargs)
            setattr(obj,method,wrapped)
    steps=model.run()
    assert len(steps)==24 and calls['environment.transition']==24
    assert calls['operators.select_candidate']==24 and calls['verifier.verify']==24
    return dict(experiment='DEV-029 pinned fusion boundary inventory',source=sha,
        branch_merged=False,seed=0,steps=len(steps),calls=calls,
        physical_environment_exercised=True,
        environment_scope='legacy synthetic relational/state-lookup/replay bit tasks; not maintenance prediction benchmark',
        prospective_osm_exercised=False,repair_proposed=True,repair_executed=False,
        original_cdl_teacher_used=False,component_advantage_validated=False,
        capability_validated=False,manifest=built.manifest,
        state_inspection=model.inspect(),
        limitation='Single bounded existing-branch smoke rollout; no new trained capability or component ablation claim')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkout',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();result=check(a.checkout)
    Path(a.output).write_text(json.dumps(result,indent=2,default=str)+'\n')
    print(json.dumps({k:result[k] for k in ('source','steps','calls','prospective_osm_exercised','repair_executed','capability_validated')}))
