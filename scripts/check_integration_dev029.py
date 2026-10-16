#!/usr/bin/env python3
"""Execute preserved learned integration reference; no capability assertion.

Gradient dependence and observed calls test graph wiring, not usefulness.
Repair is proposed by the reference but not executed as a second environment
interaction. CDL here is a learned router, not the original NLL teacher.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from tac_osm.development.full_plm_reference import FullLearnedPLM, FullPLMConfig


def checkpoint():
    torch.set_num_threads(1); torch.manual_seed(29029)
    model=FullLearnedPLM(FullPLMConfig(d_model=24, num_heads=4))
    calls={}
    handles=[]
    for name in ('representation','mtsk','cdl','casm','verifier','repair'):
        def hook(module, args, output, name=name):
            calls[name]=calls.get(name,0)+1
        handles.append(getattr(model,name).register_forward_hook(hook))
    # MTSK.forward_sequence and OSM.predict are methods, not Module.forward.
    for name, method in (('mtsk','forward_sequence'),('osm','predict')):
        original=getattr(getattr(model,name),method)
        def wrapped(*args,_original=original,_name=name,**kwargs):
            calls[_name]=calls.get(_name,0)+1
            return _original(*args,**kwargs)
        setattr(getattr(model,name),method,wrapped)
    output=model(torch.randint(32,(2,4,8)),torch.randn(2,4,1,16,16),torch.randn(2,4,1,64))
    score=(output['action_logits']*torch.arange(1,6)).sum()
    casm_grad=torch.autograd.grad(score,output['casm_state'],retain_graph=True)[0]
    osm_params=list(model.osm.parameters())
    osm_grads=torch.autograd.grad(score,osm_params,retain_graph=True,allow_unused=True)
    step=model.verify_and_update(output, action=output['action_logits'].argmax(-1),outcome=torch.tensor([0.,1.]))
    for handle in handles: handle.remove()
    finite=all(torch.isfinite(t).all().item() for t in
               (output['action_logits'],step.next_memory_state,step.repair_logits))
    result=dict(experiment='DEV-029 integration wiring checkpoint',
        source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        implementation='preserved FullLearnedPLM reference; random weights; synthetic tensors',
        calls=calls, finite=finite,
        casm_action_gradient_nonzero=bool(casm_grad.abs().sum()>0),
        osm_action_gradient_nonzero=any(g is not None and bool(g.abs().sum()>0) for g in osm_grads),
        memory_updated=not torch.equal(output['memory_state'],step.next_memory_state),
        repair_proposed=True, repair_executed=False, physical_environment_exercised=False,
        original_cdl_teacher_used=False, learned_router_used=True,
        fusion_branch_integrated=False, capability_validated=False,
        component_advantage_validated=False)
    if not finite or not all(calls.get(n,0)>0 for n in ('representation','mtsk','cdl','casm','osm','verifier','repair')):
        raise AssertionError('integration boundary not exercised')
    if not result['casm_action_gradient_nonzero'] or not result['osm_action_gradient_nonzero']:
        raise AssertionError('action graph missing required dependency')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    result=checkpoint();Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
