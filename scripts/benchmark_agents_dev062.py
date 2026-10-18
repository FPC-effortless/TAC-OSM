"""Measured CPU inference costs; synthetic intervention units are not currency."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from run_learned_stack_dev055 import write
from run_identity_dev0615 import make as identity_make
from data_identity_dev0615 import IdentityWorld
from run_belief_dev062 import make as belief_make
from data_belief_dev062 import BeliefWorld


@torch.no_grad()
def measure(root,family):
    result=[];paths=sorted(Path(root).glob('group-*/**/report.json'));first=min(json.loads(p.read_text())['seed'] for p in paths);torch.set_num_threads(1)
    for path in paths:
        r=json.loads(path.read_text())
        if r['seed']!=first:continue
        weights=torch.load(path.parent/'agent.pt',weights_only=True)
        if family=='identity':model=identity_make(r['arm'],r['spec'],r['seed']).double();model.load_state_dict(weights)
        else:
            gate={k.removeprefix('gate.'):v.float() for k,v in weights.items() if k.startswith('gate.')};model=belief_make(r['arm'],r['spec'],r['seed'],gate).double();model.load_state_dict(weights)
        for dtype in (torch.float32,torch.float64):
            m=copy.deepcopy(model).to(dtype).eval()
            for batchsize in (1,16):
                for history in (32,128,512):
                    if family=='identity':
                        w=IdentityWorld(9962301,0,4,history,'fresh');d,v,k,q,_=w.public_candidates();state=m.initial(torch.tensor(k)[None].to(dtype).expand(batchsize,-1,-1),torch.tensor(q)[None].expand(batchsize,-1));raw=w.public()
                        def decision():return m.decide(state,torch.tensor(d)[None].to(dtype).expand(batchsize,-1,-1),torch.tensor(v)[None].expand(batchsize,-1))
                    else:
                        w=BeliefWorld(9962301,0,history);d,_=w.candidates();state=m.initial(torch.tensor(w.identities)[None].to(dtype).expand(batchsize,-1,-1));raw=np.array(w.events)
                        def decision():return m.decide(state,torch.tensor(d)[None].to(dtype).expand(batchsize,-1,-1),torch.tensor([[w.error,w.inspection]],dtype=dtype).expand(batchsize,-1))
                    start=time.process_time_ns()
                    for e in raw:
                        if family=='identity':state,_=m.observe(state,torch.tensor(e)[None].to(dtype).expand(batchsize,-1))
                        else:state=m.observe(state,torch.tensor(e)[None].to(dtype).expand(batchsize,-1))
                    observation_cpu=(time.process_time_ns()-start)/1e6
                    for _ in range(3):decision()
                    cpu=[];wall=[]
                    for _ in range(20):
                        a=time.process_time_ns();b=time.perf_counter_ns();decision();wall.append((time.perf_counter_ns()-b)/1e6);cpu.append((time.process_time_ns()-a)/1e6)
                    result.append({'arm':r['arm'],'seed':first,'dtype':str(dtype),'batch':batchsize,'history':history,'observation_cpu_ms':observation_cpu,'decision_cpu_ms_median':float(np.median(cpu)),'decision_wall_ms_median':float(np.median(wall)),'parameter_bytes':sum(p.numel()*p.element_size() for p in m.parameters()),'retained_tensor_bytes':sum(v.numel()*v.element_size() for v in state.values()),'logical_acceptance':float(state['present'].sum())/batchsize/history})
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--family',choices=['identity','belief'],required=True);p.add_argument('--output',required=True);a=p.parse_args();write(a.output,{'scope':'Single-host CPU process and wall time; no isolated production/SLA/DRAM/FLOPs claim. Includes public tensor conversion. Batched latency is per batch, not per item. Physical memory traffic UNKNOWN without counters.','host':{'python':sys.version,'torch':str(torch.__version__)},'measurements':measure(a.directory,a.family)})
