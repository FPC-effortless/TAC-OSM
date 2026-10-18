"""Restore policies, replay exact actions, export torch-only and audit native effects.

Native effects are recorded synthetic physics; this is not external validation.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import torch
from run_identity_dev0615 import ROOT,PROTOCOL,make,rollout,write,digest
from data_identity_dev0615 import IdentityWorld
from audit_fully_learned_dev058 import native_replay,PIN


def compare(actual,expected,drift):
    if isinstance(expected,dict):
        assert actual.keys()==expected.keys()
        for k in expected:compare(actual[k],expected[k],drift)
    elif isinstance(expected,list):
        assert len(actual)==len(expected)
        for a,b in zip(actual,expected,strict=True):compare(a,b,drift)
    elif isinstance(expected,float):
        assert np.isfinite(actual) and np.isfinite(expected)
        drift['values']+=1;drift['max_abs']=max(drift['max_abs'],abs(actual-expected))
        assert np.isclose(actual,expected,atol=1e-8,rtol=1e-8),(actual,expected)
    else:assert actual==expected,(actual,expected)


def export(folder,report,destination):
    destination.mkdir(parents=True,exist_ok=True);(destination/'LICENSE-NOTICE.txt').write_text('Project metadata declares Apache-2.0; https://www.apache.org/licenses/LICENSE-2.0\n')
    shutil.copy(ROOT/'src/tac_osm/development/robust_identity.py',destination/'agent_model.py')
    shutil.copy(ROOT/'scripts/runtime_identity_dev0615.py',destination/'runtime.py')
    shutil.copy(folder/'agent.pt',destination/'agent.pt')
    write(destination/'metadata.json',{'arm':report['arm'],'spec':report['spec'],'training_seed':report['seed'],'dtype':'float64','epsilon':1e-6,'checkpoint_sha256':digest(folder/'agent.pt')})
    (destination/'README.txt').write_text('Research agent: Python + PyTorch only. python runtime.py < demo-requests.jsonl\nPublic requests: reset, keys(6x16), entity_valid(6), events(list of 38-vectors),\ncandidates(13x19), valid(13). Events: type4, subject16, related16, bit1, tick1.\nActions: stop/inspect/repair type3 and subject16. Padding masks only.\nDeclared priors: manifest slots, cosine identity binding, learned type gates and\npublic relationship-key stream. Learned recurrent content, predictions and policy.\nDense bounded-history replay, FIFO64. No C5 or sparse-runtime claim.\nNo teacher, simulator or analytic repair/stop policy in deployment.\nOnly bounded synthetic evidence; external outcomes need independent verification.\n')


@torch.no_grad()
def standalone(folder,report,model,destination,demo_seed=6151003):
    export(folder,report,destination);requests=[];expected=[];w=IdentityWorld(demo_seed,0,4,128,'fresh');d,v,k,q,commands=w.public_candidates();state=model.initial(torch.tensor(k)[None].double(),torch.tensor(q)[None]);pending=w.public()
    for tick in range(18):
        d,v,k,q,commands=w.public_candidates();request={'reset':tick==0,'keys':k.tolist(),'entity_valid':q.tolist(),'events':pending.tolist(),'candidates':d.tolist(),'valid':v.tolist()};requests.append(request)
        for e in pending:state,_=model.observe(state,torch.tensor(e)[None].double())
        from tac_osm.development.robust_identity import choose
        out=model.decide(state,torch.tensor(d)[None].double(),torch.tensor(v)[None]);index=int(choose(out['logits'],torch.tensor(v)[None])[0]);expected.append({'index':index,'scores':out['logits'][0].tolist(),'completion_probability':float(out['completion'][0].sigmoid())})
        pending=w.public(w.act(commands[index]))
        if w.stopped:break
    text=''.join(json.dumps(r)+'\n' for r in requests);(destination/'demo-requests.jsonl').write_text(text);(destination/'demo-expected.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in expected))
    guard="""import sys,runpy,builtins
sys.path.insert(0,'.')
import torch
original=builtins.__import__
def checked(name,globals=None,locals=None,fromlist=(),level=0):
    if level==0 and name.split('.')[0] in ('numpy','tac_osm','investigation_world'):raise ImportError('Research import in deployment')
    return original(name,globals,locals,fromlist,level)
builtins.__import__=checked
runpy.run_path('runtime.py',run_name='__main__')
"""
    r=subprocess.run([sys.executable,'-I','-B','-c',guard],cwd=destination,input=text,text=True,capture_output=True);assert r.returncode==0,r.stderr;compare([json.loads(s) for s in r.stdout.splitlines()],expected,{'values':0,'max_abs':0.});return len(expected)


@torch.no_grad()
def run(directory,output,veritas=None,exports=None,smoke=False):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);root=Path(directory);p=json.loads(PROTOCOL.read_text());summary=json.loads((root/'summary.json').read_text());assert summary['status']==('SMOKE_ONLY' if smoke else 'COMPLETE')
    assert summary['protocol_sha256']==digest(PROTOCOL)
    for name,sha in json.loads((root/'manifest.json').read_text()).items():assert digest(root/name)==sha,name
    if veritas:
        assert subprocess.check_output(['git','-C',veritas,'rev-parse','HEAD'],text=True).strip()==PIN
        sys.path.insert(0,str(Path(veritas)/'src'))
    result={'status':'RUNNING','scope':'Strict float64 restore replay. Native coverage first registered seed/all arms only. Shared synthetic physics. Cross-host evidence requires remote execution.','checkpoints':0,'cells':0,'trajectories':0,'actions':0,'native_actions':0,'portable_decisions':0,'telemetry':{'values':0,'max_abs':0.},'parameter_changes':{},'prediction_sums':{},'host':{'cpu_models':sorted({line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')}),'python':sys.version,'torch':str(torch.__version__),'machine':subprocess.check_output(['uname','-a'],text=True).strip()},'cross_hardware':'UNKNOWN until comparison on another host'}
    paths=sorted(root.glob('*/**/report.json'));started=time.monotonic()
    for path in paths:
        r=json.loads(path.read_text());folder=path.parent;assert digest(folder/'agent.pt')==r['checkpoint_sha256'];
        if not smoke:
            for source in ('src/tac_osm/development/robust_identity.py','scripts/data_identity_dev0615.py','scripts/run_identity_dev0615.py','research/dev0615/protocol.json'):
                assert subprocess.check_output(['git','show',f"{r['source']}:{source}"],cwd=ROOT)==(ROOT/source).read_bytes(),source
        m=make(r['arm'],r['spec'],r['seed']).double().eval();initial={k:v.clone() for k,v in m.named_parameters()};m.load_state_dict(torch.load(folder/'agent.pt',weights_only=True));key=str(folder.relative_to(root));result['parameter_changes'][key]={k:float((v-initial[k]).abs().max()) for k,v in m.named_parameters()};result['checkpoints']+=1
        cells=r['validation']
        if (folder/'test.json').exists():cells=cells+json.loads((folder/'test.json').read_text())['cells']
        for cell_index,cell in enumerate(cells):
            phase='validation' if cell_index<len(r['validation']) else 'primary' if cell_index-len(r['validation'])<18 else 'secondary'
            captured=[];original_decide=m.decide
            def observed_decide(state,candidates,valid):
                out=original_decide(state,candidates,valid)
                from tac_osm.development.robust_identity import choose
                captured.append((out,choose(out['logits'],valid)));return out
            m.decide=observed_decide
            actual,_=rollout(m,cell['seed'],cell['entities'],cell['history'],len(cell['rows']),cell['family'],cell['horizon']);m.decide=original_decide;compare(actual,cell,result['telemetry']);result['cells']+=1
            for step,(out,chosen) in enumerate(captured):
                for i,row in enumerate(cell['rows']):
                    if step>=len(row['actions']):continue
                    action=row['actions'][step];n=cell['entities'];s=result['prediction_sums'].setdefault(f"{r['arm']}/{r['seed']}/{phase}",{'current_squared':0.,'current_count':0,'outcome_squared':0.,'outcome_count':0,'cost_squared':0.,'decisions':0});before=torch.tensor(action['before'],dtype=torch.float64);after=torch.tensor(action['after'],dtype=torch.float64);j=int(chosen[i]);s['current_squared']+=float(((out['current'][i,:n].sigmoid()-before)**2).sum());s['current_count']+=n;s['outcome_squared']+=float(((out['outcomes'][i,j,:n].sigmoid()-after)**2).sum());s['outcome_count']+=n;s['cost_squared']+=float((out['cost'][i,j]*20-action['cost'])**2);s['decisions']+=1
            for row in cell['rows']:
                w=IdentityWorld(row['seed'],row['index'],row['entities'],row['history'],cell['family'])
                for action in row['actions']:
                    w.act((action['kind'],action['target']));assert w.actions[-1]==action
                end=w.finish(row['writes'])
                for k in ('success','stopped','final_faults','cost','write_cost'):assert end[k]==row[k],k
                result['trajectories']+=1;result['actions']+=len(row['actions'])
                if veritas and r['seed']==p['export_seed'] and path.parent.parent.name.startswith('group-'):result['native_actions']+=native_replay(row)
        if r['seed']==p['export_seed'] and exports:result['portable_decisions']+=standalone(folder,r,m,Path(exports)/r['arm'])
        write(output,result);print('restored',key,result['trajectories'],flush=True)
    result['visited_prediction_scores']={key:{'current_state_Brier':s['current_squared']/max(1,s['current_count']),'one_action_outcome_Brier':s['outcome_squared']/max(1,s['outcome_count']),'one_action_cost_RMSE':(s['cost_squared']/max(1,s['decisions']))**.5,'decisions':s['decisions']} for key,s in result['prediction_sums'].items()}
    result['status']='COMPLETE';result['elapsed_seconds']=time.monotonic()-started;write(output,result);print('AUDIT_COMPLETE',json.dumps({k:result[k] for k in ('status','checkpoints','trajectories','native_actions','portable_decisions','telemetry','host')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--output',required=True);p.add_argument('--veritas');p.add_argument('--exports');p.add_argument('--smoke',action='store_true');a=p.parse_args()
    try:run(a.directory,a.output,a.veritas,a.exports,a.smoke)
    except Exception:
        from audit_failure_dev062 import preserve
        preserve(a.output)
        raise
