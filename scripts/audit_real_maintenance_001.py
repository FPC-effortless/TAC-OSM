"""Strict real-sensor checkpoint replay, portable export and measured CPU costs."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import torch
from run_real_maintenance_001 import ROOT,PROTOCOL,evaluate,MaintenanceForecast
from run_learned_stack_dev055 import write,digest
from audit_identity_dev0615 import compare


@torch.no_grad()
def run(directory,output,exports,smoke=False):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);root=Path(directory);summary=json.loads((root/'summary.json').read_text());assert summary['status']==('SMOKE_ONLY' if smoke else 'COMPLETE');assert summary['protocol_sha256']==digest(PROTOCOL)
    for name,sha in json.loads((root/'manifest.json').read_text()).items():assert digest(root/name)==sha,name
    result={'status':'RUNNING','checkpoints':0,'windows_replayed':0,'portable_predictions':0,'telemetry':{'values':0,'max_abs':0.},'hardware':[],'native_action_replay':'NOT_APPLICABLE: source has no executable intervention/receipt contract','cross_host':'UNKNOWN until remote execution','scope':'Actual operational sensor forecasts; no repair-success, intervention-cost or commercial proof.'};test=torch.load(root/'_datasets/test.pt',weights_only=True);val=torch.load(root/'_datasets/validation.pt',weights_only=True)
    for path in sorted(root.glob('*/**/report.json')):
        r=json.loads(path.read_text());s=r['spec'];folder=path.parent;assert digest(folder/'agent.pt')==r['checkpoint_sha256']
        if not smoke:
            for source in ('src/tac_osm/development/maintenance_forecast.py','scripts/run_real_maintenance_001.py','research/real-maint-001/PROTOCOL.md'):assert subprocess.check_output(['git','show',f"{r['source']}:{source}"],cwd=ROOT)==(ROOT/source).read_bytes(),source
        m=MaintenanceForecast(r['arm'],s['width'],s['aux'],r['seed']).double().eval();m.load_state_dict(torch.load(folder/'agent.pt',weights_only=True));compare(evaluate(m,val,True),r['validation'],result['telemetry']);result['windows_replayed']+=len(val['x'])
        if (folder/'test.json').exists():compare(evaluate(m,test,True),json.loads((folder/'test.json').read_text()),result['telemetry']);result['windows_replayed']+=len(test['x'])
        if r['seed']==(10092701 if smoke else 92701):
            destination=Path(exports)/r['arm'];destination.mkdir(parents=True,exist_ok=True)
            for source,name in [(ROOT/'src/tac_osm/development/maintenance_forecast.py','agent_model.py'),(ROOT/'scripts/runtime_real_maintenance_001.py','runtime.py'),(folder/'agent.pt','agent.pt')]:shutil.copy(source,destination/name)
            (destination/'LICENSE-NOTICE.txt').write_text('Project metadata declares Apache-2.0; https://www.apache.org/licenses/LICENSE-2.0\n')
            write(destination/'metadata.json',{'arm':r['arm'],'spec':s,'seed':r['seed'],'checkpoint_sha256':digest(folder/'agent.pt'),'data_source':'MetroPT-3, UCI DOI10.24432/C5VW3R, CC BY4.0','scope':'bounded raw-sensor forecasting only; no repair action or commercial claim'})
            (destination/'README.txt').write_text('Python + PyTorch. python runtime.py < demo-requests.jsonl\nInput history24x15 past contiguous ten-minute sensor-bin means, listed in source metadata.\nTraining-only normalization is embedded in weights. Output next sensor means/std, not health certification or a repair policy.\nSource: MetroPT-3, Davari et al., UCI DOI10.24432/C5VW3R, CC BY4.0.\nOne real compressor; four reported failure intervals are evaluation labels only.\nGaussian marginal assumptions include binary sensors; forecast uncertainty is not calibrated physical-fault probability.\n')
            x=test['x'][:2].double();o=m(x);expected=[{'next_sensor_mean':o['mean'][i].tolist(),'next_sensor_std':(o['log_variance'][i].mul(.5).exp()*m.scale).tolist(),'scope':'observational forecast; no repair-success claim'} for i in range(len(x))];requests=''.join(json.dumps({'history':v.tolist()})+'\n' for v in x);(destination/'demo-requests.jsonl').write_text(requests);(destination/'demo-expected.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in expected));guard="""import sys,runpy,builtins
sys.path.insert(0,'.')
import torch
original=builtins.__import__
def checked(name,globals=None,locals=None,fromlist=(),level=0):
    if level==0 and name.split('.')[0] in ('numpy','tac_osm','investigation_world'):raise ImportError('research dependency')
    return original(name,globals,locals,fromlist,level)
builtins.__import__=checked
runpy.run_path('runtime.py',run_name='__main__')
""";response=subprocess.run([sys.executable,'-I','-B','-c',guard],cwd=destination,input=requests,text=True,capture_output=True);assert response.returncode==0,response.stderr;compare([json.loads(line) for line in response.stdout.splitlines()],expected,result['telemetry']);result['portable_predictions']+=len(expected)
            for dtype in (torch.float32,torch.float64):
                model=copy.deepcopy(m).to(dtype);sample=x[:1].to(dtype)
                for _ in range(3):model(sample)
                cpu=[];wall=[]
                for _ in range(30):
                    a=time.process_time_ns();b=time.perf_counter_ns();model(sample);wall.append((time.perf_counter_ns()-b)/1e6);cpu.append((time.process_time_ns()-a)/1e6)
                result['hardware'].append({'arm':r['arm'],'dtype':str(dtype),'batch':1,'cpu_ms_median':float(np.median(cpu)),'wall_ms_median':float(np.median(wall)),'parameter_bytes':sum(v.numel()*v.element_size() for v in model.parameters()),'input_bytes':sample.numel()*sample.element_size(),'physical_memory_traffic':'UNKNOWN'})
        result['checkpoints']+=1;write(output,result)
    result['status']='COMPLETE';result['host']={'cpu_models':sorted({line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')}),'python':sys.version,'torch':str(torch.__version__)};write(output,result);print('AUDIT_COMPLETE',json.dumps({k:result[k] for k in ('status','checkpoints','windows_replayed','portable_predictions','telemetry','host')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--output',required=True);p.add_argument('--exports',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();run(a.directory,a.output,a.exports,a.smoke)
