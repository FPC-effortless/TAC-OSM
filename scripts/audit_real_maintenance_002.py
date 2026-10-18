"""Strict independent source/checkpoint replay, never load reserved B."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import platform
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from tac_osm.development.residual_forecast import ResidualForecast
from data_real_maintenance_002 import FEATURES,digest,write
from run_real_maintenance_002 import evaluate,score
from audit_identity_dev0615 import compare
FROZEN=['src/tac_osm/development/residual_forecast.py','scripts/run_real_maintenance_002.py','scripts/data_real_maintenance_002.py','research/real-maint-002/PROTOCOL.md']


@torch.no_grad()
def audit(directory,output,exports):
    torch.set_num_threads(1);root=Path(directory);data=root/'_datasets';summary=json.loads((root/'summary.json').read_text());seal=json.loads((data/'seal-state.json').read_text());assert summary['status']=='COMPLETE' and summary['test_A_opened'] and not summary['test_B_opened'] and not seal['B_metrics_opened'];manifest=json.loads((data/'manifest.json').read_text());start=time.process_time();result={'status':'RUNNING','checkpoints':0,'test_A_checkpoint_replays':0,'predicted_A_windows':0,'portable_predictions':0,'telemetry':{'values':0,'max_abs':0.},'B_metrics_opened':False,'benchmarks':[]};write(output,result)
    try:
        for name in ('train','validation','sealed-a'):assert digest(data/(name+'.pt'))==manifest['data_hashes'][name]
        val=torch.load(data/'validation.pt',weights_only=True);test=torch.load(data/'sealed-a.pt',weights_only=True);guarded=set()
        for report_path in sorted(root.glob('*/*/report.json')):
            r=json.loads(report_path.read_text());folder=report_path.parent;assert digest(folder/'agent.pt')==r['checkpoint_sha256']
            if r['source'] not in guarded:
                for name in FROZEN:
                    expected=subprocess.check_output(['git','show',r['source']+':'+name],cwd=ROOT);assert (ROOT/name).read_bytes()==expected,'Frozen fit source changed: '+name
                guarded.add(r['source'])
            assert r['selected_update']==min([v for v in r['curves'] if v['eligible_for_selection']],key=lambda v:score(v['validation'],v['update']))['update']
            m=ResidualForecast(**r['spec'],seed=r['seed']).double().eval();m.load_state_dict(torch.load(folder/'agent.pt',weights_only=True));naive=torch.tensor(r['naive_scale'],dtype=torch.float64);compare(evaluate(m,val,naive),r['validation'],result['telemetry'])
            if folder.parent.name.startswith('group-'):
                expected=json.loads((folder/'test-a.json').read_text());compare(evaluate(m,test,naive,True),expected,result['telemetry']);result['test_A_checkpoint_replays']+=1;result['predicted_A_windows']+=len(test['x'])
            result['checkpoints']+=1;write(output,result)
            if r['seed']!=93001:continue
            dest=Path(exports)/r['arm'];dest.mkdir(parents=True,exist_ok=True);shutil.copy(ROOT/'src/tac_osm/development/residual_forecast.py',dest/'model.py');shutil.copy(ROOT/'scripts/runtime_residual_forecast_002.py',dest/'runtime.py');shutil.copy(folder/'agent.pt',dest/'agent.pt');write(dest/'metadata.json',{'spec':r['spec'],'seed':r['seed'],'features':FEATURES,'checkpoint_sha256':r['checkpoint_sha256'],'qualification':'RESEARCH_ONLY; inspect all-channel gate before use','source_doi':manifest['doi'],'source_license':manifest['license']});x=val['x'][:2].double();requests=''.join(json.dumps({'history':v.tolist()})+'\n' for v in x);(dest/'demo-requests.jsonl').write_text(requests)
            (dest/'README.txt').write_text('CPU PyTorch only. Run python runtime.py < demo-requests.jsonl. Point forecasts are not maintenance actions, calibrated fault probabilities or verified repairs. Data: Veloso/Gama/Ribeiro/Pereira, MetroPT2 DOI10.5281/zenodo.7766691, CC BY4.0. Project declares Apache2; no separate repository LICENSE file was present.\n')
            guard="""import sys,runpy,builtins
sys.path.insert(0,'.')
import torch
original=builtins.__import__
def checked(name,globals=None,locals=None,fromlist=(),level=0):
    if level==0 and name.split('.')[0] in ('numpy','tac_osm','investigation_world'):raise ImportError('research dependency')
    return original(name,globals,locals,fromlist,level)
builtins.__import__=checked
runpy.run_path('runtime.py',run_name='__main__')
""";response=subprocess.run([sys.executable,'-I','-B','-c',guard],cwd=dest,input=requests,text=True,capture_output=True);assert response.returncode==0,response.stderr;pred=[json.loads(v) for v in response.stdout.splitlines()];compare([r['forecast'] for r in pred],m(x).tolist(),result['telemetry']);result['portable_predictions']+=len(pred)
            for batch in (1,16):
                xx=val['x'][:batch].double()
                for _ in range(5):m(xx)
                cpu=[];wall=[]
                for _ in range(32):
                    c=time.process_time_ns();w=time.perf_counter_ns();m(xx);cpu.append((time.process_time_ns()-c)/1e6);wall.append((time.perf_counter_ns()-w)/1e6)
                result['benchmarks'].append({'arm':r['arm'],'batch':batch,'CPU_ms_median':sorted(cpu)[16],'wall_ms_median':sorted(wall)[16],'parameter_bytes':sum(p.numel()*p.element_size() for p in m.parameters()),'scope':'Model only; tensor/JSON/source preprocessing/verification excluded. DRAM/FLOPs UNKNOWN.'})
        assert result['checkpoints']==88 and result['test_A_checkpoint_replays']==80 and result['portable_predictions']==8;result['status']='COMPLETE'
    except Exception:
        import traceback
        result['status']='FAILED';result['failure']=traceback.format_exc();raise
    finally:
        result['audit_CPU_seconds']=time.process_time()-start;result['host']={'python':platform.python_version(),'torch':torch.__version__,'cpu_models':sorted(set(line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')))};write(output,result);print('AUDIT',json.dumps({k:v for k,v in result.items() if k!='benchmarks'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--output',required=True);p.add_argument('--exports',required=True);a=p.parse_args();audit(a.directory,a.output,a.exports)
