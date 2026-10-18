"""Same-weight packet conformance and fair batched versus cached CPU profiles."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import torch
from run_belief_dev062 import ROOT,rollout
from tac_osm.development.batched_belief_agent import BatchedBeliefAgent,BatchedCachedBeliefAgent
from audit_cached_belief_dev062 import restore,requests,timed,cpu_used
from audit_belief_dev062 import standalone
from audit_identity_dev0615 import compare
from run_learned_stack_dev055 import write,digest


def model_for(folder,variant):
    r,m=restore(folder)
    if variant!='base':
        cls=BatchedBeliefAgent if variant=='batch' else BatchedCachedBeliefAgent;n=cls(r['arm'],r['spec']['width'],r['spec']['aux'],r['seed']).double().eval();n.load_state_dict(m.state_dict());m=n
    return r,m


@torch.no_grad()
def timed_packet(model,records):
    import time
    dtype=next(model.parameters()).dtype;state=model.initial(records[0][0].to(dtype));observe=decide=0.;matches=choices=0
    for _,events,candidates,prices,expected in records:
        if events:
            packet=torch.stack([e.to(dtype) for e in events],1);start=__import__('time').process_time_ns();state=model.observe_many(state,packet);observe+=(time.process_time_ns()-start)/1e6
        if candidates is None:continue
        candidates=candidates.to(dtype);prices=prices.to(dtype);start=time.process_time_ns();choice=int(model.choice(model.decide(state,candidates,prices))[0]);decide+=(time.process_time_ns()-start)/1e6;matches+=choice==expected;choices+=1
    return {'observe_cpu_ms':observe,'decision_cpu_ms':decide,'total_model_cpu_ms':observe+decide,'choices':choices,'matched_float64_trace_choices':matches,'retained_tensor_bytes':sum(v.numel()*v.element_size() for v in state.values() if isinstance(v,torch.Tensor))}


@torch.no_grad()
def portable(folder,r,model,destination,variant):
    standalone(folder,r,model,destination,demo_seed=6261003 if r['seed']==92601 else 6281003)
    for source,name in [('cached_belief_agent.py','cached_model.py'),('batched_belief_agent.py','batch_model.py')]:shutil.copy(ROOT/'src/tac_osm/development'/source,destination/name)
    cls='BatchedBeliefAgent' if variant=='batch' else 'BatchedCachedBeliefAgent';runtime=(ROOT/'scripts/runtime_belief_dev062.py').read_text().replace('from agent_model import BeliefAgent',f'from batch_model import {cls} as BeliefAgent').replace("for e in r['events']:state=m.observe(state,tensor(e,[46]))", "if r['events']:\n            packet=torch.tensor([r['events']],dtype=torch.float64)\n            if packet.ndim!=3 or packet.shape[-1]!=46 or not torch.isfinite(packet).all():raise ValueError('Finite public46-vector packet required')\n            state=m.observe_many(state,packet)");(destination/'runtime.py').write_text(runtime)
    guard="""import sys,runpy,builtins
sys.path.insert(0,'.')
import torch
original=builtins.__import__
def checked(name,globals=None,locals=None,fromlist=(),level=0):
    if level==0 and name.split('.')[0] in ('numpy','tac_osm','investigation_world'):raise ImportError('research dependency')
    return original(name,globals,locals,fromlist,level)
builtins.__import__=checked
runpy.run_path('runtime.py',run_name='__main__')
""";response=subprocess.run([sys.executable,'-I','-B','-c',guard],cwd=destination,input=(destination/'demo-requests.jsonl').read_text(),text=True,capture_output=True);assert response.returncode==0,response.stderr;compare([json.loads(v) for v in response.stdout.splitlines()],[json.loads(v) for v in (destination/'demo-expected.jsonl').read_text().splitlines()],{'values':0,'max_abs':0.});meta=json.loads((destination/'metadata.json').read_text());meta['runtime']=variant;write(destination/'metadata.json',meta)


@torch.no_grad()
def run(directory,output,exports,seed_start):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);root=Path(directory);assert json.loads((root/'summary.json').read_text())['status']=='COMPLETE';start=cpu_used();result={'status':'RUNNING','variants':{},'measurements':[],'failures':[],'scope':'Available packets only; no future observations. All initial/receipt/terminal model work counted; simulator/filter/JSON/tensor creation excluded. Same trained weights and semantic choices.'}
    try:
        paths=sorted(root.glob('group-*/tac/report.json'))+sorted(p for p in (root/f'group-{seed_start}').glob('*/report.json') if p.parent.name!='tac')
        for variant in ('batch','batch_cache'):
            diag={'checkpoints':0,'trajectories':0,'cells':0,'telemetry':{'values':0,'max_abs':0.}};result['variants'][variant]=diag
            for path in paths:
                r,m=model_for(path.parent,variant);tests=json.loads((path.parent/'test.json').read_text())['cells'];cells=r['validation']+tests if r['seed']==seed_start else tests[:2]
                for c in cells:
                    if cpu_used()-start>600:raise TimeoutError('Registered packet audit CPU cap')
                    actual,_=rollout(m,c['seed'],c['history'],c['worlds'],c['horizon'],False,c['error'],c['inspection']);compare(actual,c,diag['telemetry']);diag['cells']+=1;diag['trajectories']+=c['worlds']
                diag['checkpoints']+=1;write(output,result)
        for path in sorted((root/f'group-{seed_start}').glob('*/report.json')):
            r,base=model_for(path.parent,'base')
            for history in (32,128,512):
                for index in range(8):
                    trace=requests(base,6263001 if seed_start==92601 else 6283001,index,history)
                    for dtype in (torch.float32,torch.float64):
                        for variant in ('base','batch','batch_cache'):
                            if cpu_used()-start>600:raise TimeoutError('Registered packet audit CPU cap')
                            _,m=model_for(path.parent,variant);m=m.to(dtype);measurement=timed(m,trace) if variant=='base' else timed_packet(m,trace);result['measurements'].append(dict(arm=r['arm'],history=history,index=index,dtype=str(dtype),variant=variant,**measurement))
            for variant in ('batch','batch_cache'):
                _,m=model_for(path.parent,variant);portable(path.parent,r,m,Path(exports)/variant/r['arm'],variant)
        means={v:float(np.mean([r['total_model_cpu_ms'] for r in result['measurements'] if r['arm']=='tac' and r['dtype']=='torch.float64' and r['variant']==v])) for v in ('base','batch','batch_cache')};result['TAC_float64_mean_total_model_CPU_ms']=means;result['base_batch_ratio']=means['batch']/means['base'];result['cache_vs_equally_batched_base_ratio']=means['batch_cache']/means['batch'];result['base_batch_qualifies']=result['base_batch_ratio']<=.9;result['cache_batch_qualifies']=result['cache_vs_equally_batched_base_ratio']<=.9 and means['batch_cache']/means['base']<=.9;result['recommended_runtime']='batch_cache' if result['cache_batch_qualifies'] else 'batch' if result['base_batch_qualifies'] else 'base';result['status']='COMPLETE'
    except Exception:
        import traceback
        result['status']='FAILED';result['failures'].append(traceback.format_exc())
    result['audit_cpu_seconds']=cpu_used()-start;result['protocol_sha256']=digest(ROOT/'research/dev062-batch/PROTOCOL.md');write(output,result);print('PACKET_AUDIT',json.dumps({k:v for k,v in result.items() if k not in ('measurements','failures')}),flush=True)
    if result['failures']:raise RuntimeError(result['failures'][0])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--output',required=True);p.add_argument('--exports',required=True);p.add_argument('--seed-start',type=int,default=92601);a=p.parse_args();run(a.directory,a.output,a.exports,a.seed_start)
