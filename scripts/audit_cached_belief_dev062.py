"""Registered same-weight cached-state conformance and inclusive CPU measurements."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import subprocess
import resource
import time
import numpy as np
import torch
from run_belief_dev062 import ROOT,make,rollout
from data_belief_dev062 import BeliefWorld
from tac_osm.development.cached_belief_agent import CachedBeliefAgent
from audit_belief_dev062 import export,standalone
from audit_identity_dev0615 import compare
from run_learned_stack_dev055 import write,digest


def cpu_used():
    child=resource.getrusage(resource.RUSAGE_CHILDREN)
    return time.process_time()+child.ru_utime+child.ru_stime


def restore(folder,cached=False):
    r=json.loads((folder/'report.json').read_text());weights=torch.load(folder/'agent.pt',weights_only=True);gate={k.removeprefix('gate.'):v.float() for k,v in weights.items() if k.startswith('gate.')}
    m=CachedBeliefAgent(r['arm'],r['spec']['width'],r['spec']['aux'],r['seed']) if cached else make(r['arm'],r['spec'],r['seed'],gate);m=m.double().eval();m.load_state_dict(weights);return r,m


@torch.no_grad()
def requests(model,seed,index,history):
    w=BeliefWorld(seed,index,history);state=model.initial(torch.tensor(w.identities)[None].double());pending=w.events.copy();records=[]
    for _ in range(10):
        d,permutation=w.candidates();prices=torch.tensor([[w.error,w.inspection]],dtype=torch.float64);events=[torch.tensor(e)[None].double() for e in pending]
        for e in events:state=model.observe(state,e)
        candidates=torch.tensor(d)[None].double();out=model.decide(state,candidates,prices);choice=int(model.choice(out)[0]);records.append((torch.tensor(w.identities)[None].double(),events,candidates,prices,choice));start=len(w.events);w.act(int(permutation[choice]));pending=w.events[start:]
        if w.stopped:break
    return records


@torch.no_grad()
def timed(model,records):
    dtype=next(model.parameters()).dtype;state=model.initial(records[0][0].to(dtype));observe=decide=0.;matches=0
    for _,events,candidates,prices,expected in records:
        for e in events:
            e=e.to(dtype);start=time.process_time_ns();state=model.observe(state,e);observe+=(time.process_time_ns()-start)/1e6
        candidates=candidates.to(dtype);prices=prices.to(dtype);start=time.process_time_ns();choice=int(model.choice(model.decide(state,candidates,prices))[0]);decide+=(time.process_time_ns()-start)/1e6;matches+=choice==expected
    return {'observe_cpu_ms':observe,'decision_cpu_ms':decide,'total_model_cpu_ms':observe+decide,'choices':len(records),'matched_float64_trace_choices':matches,'retained_tensor_bytes':sum(v.numel()*v.element_size() for v in state.values())}


@torch.no_grad()
def run(directory,output,exports):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);root=Path(directory);assert json.loads((root/'summary.json').read_text())['status']=='COMPLETE';start=cpu_used();result={'status':'RUNNING','parity':{'values':0,'max_abs':0.},'checkpoints':0,'trajectories':0,'cells':0,'measurements':[],'failures':[],'scope':'Same weights, exact semantic choices; forced fixed-trace CPU timing excludes simulation/filter, JSON parsing and input tensor creation. No memory-traffic, C5 or commercial claim.'}
    try:
        for path in sorted(root.glob('group-*/tac/report.json'))+sorted(p for p in (root/'group-92601').glob('*/report.json') if p.parent.name!='tac'):
            r,m=restore(path.parent,True);allcells=json.loads((path.parent/'test.json').read_text())['cells'];cells=r['validation']+allcells if r['seed']==92601 else allcells[:2]
            for c in cells:
                if cpu_used()-start>600:raise TimeoutError('Registered cache audit CPU cap')
                actual,_=rollout(m,c['seed'],c['history'],c['worlds'],c['horizon'],False,c['error'],c['inspection']);compare(actual,c,result['parity']);result['cells']+=1;result['trajectories']+=c['worlds']
            result['checkpoints']+=1;write(output,result)
        for path in sorted((root/'group-92601').glob('*/report.json')):
            r,base=restore(path.parent);_,cached=restore(path.parent,True)
            for history in (32,128,512):
                for index in range(8):
                    trace=requests(base,6263001,index,history)
                    for dtype in (torch.float32,torch.float64):
                        for label,model in [('base',copy.deepcopy(base).to(dtype)),('cache',copy.deepcopy(cached).to(dtype))]:
                            if cpu_used()-start>600:raise TimeoutError('Registered cache audit CPU cap')
                            result['measurements'].append(dict(arm=r['arm'],history=history,index=index,dtype=str(dtype),variant=label,**timed(model,trace)))
            dest=Path(exports)/r['arm'];standalone(path.parent,r,cached,dest);export(path.parent,r,dest);shutil.copy(ROOT/'src/tac_osm/development/cached_belief_agent.py',dest/'cached_model.py');runtime=(ROOT/'scripts/runtime_belief_dev062.py').read_text().replace('from agent_model import BeliefAgent','from cached_model import CachedBeliefAgent as BeliefAgent');(dest/'runtime.py').write_text(runtime)
            # Export the exact cached implementation; separate from original artifact.
            meta=json.loads((dest/'metadata.json').read_text());meta['runtime']='same-weight incremental learned state';write(dest/'metadata.json',meta)
            guard="""import sys,runpy,builtins
sys.path.insert(0,'.')
import torch
original=builtins.__import__
def checked(name,globals=None,locals=None,fromlist=(),level=0):
    if level==0 and name.split('.')[0] in ('numpy','tac_osm','investigation_world'):raise ImportError('research dependency')
    return original(name,globals,locals,fromlist,level)
builtins.__import__=checked
runpy.run_path('runtime.py',run_name='__main__')
"""
            response=subprocess.run([sys.executable,'-I','-B','-c',guard],cwd=dest,input=(dest/'demo-requests.jsonl').read_text(),text=True,capture_output=True);assert response.returncode==0,response.stderr;compare([json.loads(v) for v in response.stdout.splitlines()],[json.loads(v) for v in (dest/'demo-expected.jsonl').read_text().splitlines()],result['parity'])
        b=[m['total_model_cpu_ms'] for m in result['measurements'] if m['arm']=='tac' and m['dtype']=='torch.float64' and m['variant']=='base'];c=[m['total_model_cpu_ms'] for m in result['measurements'] if m['arm']=='tac' and m['dtype']=='torch.float64' and m['variant']=='cache'];result['TAC_float64_total_CPU_ratio']=float(np.mean(c)/np.mean(b));result['qualifies']=result['TAC_float64_total_CPU_ratio']<=.9;result['status']='COMPLETE'
    except Exception:
        import traceback
        result['status']='FAILED';result['qualifies']=False;result['failures'].append(traceback.format_exc())
    result['audit_cpu_seconds']=cpu_used()-start;result['protocol_sha256']=digest(ROOT/'research/dev062-cache/PROTOCOL.md');write(output,result);print('CACHE_AUDIT',json.dumps({k:v for k,v in result.items() if k not in ('measurements','failures')}),flush=True)
    if result['failures']:raise RuntimeError(result['failures'][0])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--output',required=True);p.add_argument('--exports',required=True);a=p.parse_args();run(a.directory,a.output,a.exports)
