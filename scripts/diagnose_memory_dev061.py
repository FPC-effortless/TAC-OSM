"""Frozen public-read interventions and public/privileged references; no fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from run_structured_memory_dev061 import ROOT,PROTOCOL,ResearchWorld,evaluate
from run_learned_stack_dev055 import write,digest
from audit_evidence_gaps_dev059 import public_reference
from tac_osm.development.frozen_read_diagnostic import FrozenReadDiagnostic
from tac_osm.development.learned_agent import AgentConfig
from tac_osm.development.learned_agent_world import teacher


def reference(seed,n,h,count,family,private=False):
    rows=[]
    for i in range(count):
        w=ResearchWorld(seed,i,n,h,family)
        for _ in range(2*n+4):
            d,v,_,_,commands=w.candidates()
            choice=int(teacher(w.bits,w.parents,commands).argmax()) if private else public_reference(w.events,d,v)
            w.act(commands[choice])
            if w.stopped:break
        r=w.finish(0);initial=r.pop('initial_events');r.pop('keys');r.pop('parents')
        r.update(family=family,economic_cost=r['cost'],physical_repair=r['final_faults']==0,
                 public_initial_sha256=hashlib.sha256(np.asarray(initial,np.float32).tobytes()).hexdigest())
        rows.append(r)
    return {'seed':seed,'entities':n,'history':h,'family':family,'worlds':count,'rows':rows}


def run(output):
    torch.set_num_threads(1);p=json.loads(PROTOCOL.read_text());result={'protocol_sha256':digest(PROTOCOL),'references':{},'frozen':{},'limitations':'Forced reads are public retained-bank interventions. Latest-minus-oldest holds forced relation reads constant, changes status age conditional on this forced format. Public reference is task-specific; private policy not information-matched under noise.'}
    for private in (False,True):
        name='privileged' if private else 'public'
        result['references'][name]=[reference(p['data']['test_seed'],n,h,32,'clean',private) for n in (2,4,6) for h in (32,128,512,1024)]
        result['references'][name]+=[reference(p['data']['test_seed'],n,96,32,family,private) for family in ('unseen_ids','noisy_initial','noisy_receipts') for n in (2,4,6)]
        result['references'][name+'_validation']=[reference(p['data']['validation_seed'],n,h,16,'clean',private) for n in (2,4,6) for h in (32,96)]
    for seed in (7581,7582):
        path=ROOT/f'results/dev058-main/r0-{seed}';report=json.loads((path/'report.json').read_text())
        models={}
        for mode in ('learned','latest','oldest'):
            m=FrozenReadDiagnostic(AgentConfig(**report['config']),seed,mode).eval();m.load_state_dict(torch.load(path/'agent.pt',weights_only=True),strict=True)
            models[mode]=[evaluate(m,p['data']['test_seed'],n,h,32) for n in (2,4,6) for h in (32,128,512,1024)]
        result['frozen'][str(seed)]={'checkpoint_sha256':digest(path/'agent.pt'),'modes':models}
    write(output,result)


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--output',required=True);args=a.parse_args();run(Path(args.output))
