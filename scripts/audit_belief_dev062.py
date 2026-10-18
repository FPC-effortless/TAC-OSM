"""Independent checkpoint/trajectory/portable-interface replay for DEV062A."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import torch
from run_belief_dev062 import ROOT,PROTOCOL,make,rollout,fixed_metrics
from data_belief_dev062 import BeliefWorld
from run_learned_stack_dev055 import write,digest
from audit_identity_dev0615 import compare
from audit_fully_learned_dev058 import native_replay,PIN


def export(folder,report,destination):
    destination.mkdir(parents=True,exist_ok=True);(destination/'LICENSE-NOTICE.txt').write_text('Project metadata declares Apache-2.0; https://www.apache.org/licenses/LICENSE-2.0\n');shutil.copy(ROOT/'src/tac_osm/development/belief_agent.py',destination/'agent_model.py');shutil.copy(ROOT/'scripts/runtime_belief_dev062.py',destination/'runtime.py');shutil.copy(folder/'agent.pt',destination/'agent.pt');write(destination/'metadata.json',{'arm':report['arm'],'spec':report['spec'],'seed':report['seed'],'dtype':'float64','epsilon':1e-6,'checkpoint_sha256':digest(folder/'agent.pt')})
    (destination/'README.txt').write_text('Requires Python + PyTorch only. python runtime.py < demo-requests.jsonl\nPublic JSON-lines: reset, keys(2x16), events(new raw46-vectors), candidates(5x19), prices(sensor-error,inspection-price).\nEvents: source6(initial,relation,action-start,repair-receipt,exact-inspection,noise), subject16,other16,bit,action-type3,tick,exact,sensor-error,inspection-price.\nOutputs: learned choice index, scores, joint probabilities(00,01,10,11), completion and two fault marginals in one forward pass.\nPriors: public typed manifest, cosine identity binding, trained noise-type gate, FIFO64, two entity state labels.\nNo Bayes filter, teacher, physical transition function, analytical planner or stop override in deployment.\nPosterior-supervised bounded research model; independent external verification remains necessary. No commercial or universal calibration claim.\n')


@torch.no_grad()
def standalone(folder,r,m,destination,demo_seed=6261003):
    export(folder,r,destination);w=BeliefWorld(demo_seed,0,128);state=m.initial(torch.tensor(w.identities)[None].double());pending=w.events.copy();requests=[];expected=[]
    for tick in range(10):
        d,permutation=w.candidates();prices=[w.error,w.inspection];requests.append({'reset':tick==0,'keys':w.identities.tolist(),'events':[e.tolist() for e in pending],'candidates':d.tolist(),'prices':prices})
        for e in pending:state=m.observe(state,torch.tensor(e)[None].double())
        o=m.decide(state,torch.tensor(d)[None].double(),torch.tensor(prices,dtype=torch.float64)[None]);index=int(m.choice(o)[0]);p=o['posterior'][0];expected.append({'index':index,'scores':o['logits'][0].tolist(),'joint_public_state_probabilities':p.tolist(),'completion_probability':float(p[0]),'entity_fault_probabilities':[float(p[2]+p[3]),float(p[1]+p[3])]});start=len(w.events);w.act(int(permutation[index]));pending=w.events[start:]
        if w.stopped:break
    text=''.join(json.dumps(r)+'\n' for r in requests);(destination/'demo-requests.jsonl').write_text(text);(destination/'demo-expected.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in expected));guard="""import sys,runpy,builtins
sys.path.insert(0,'.')
import torch
original=builtins.__import__
def checked(name,globals=None,locals=None,fromlist=(),level=0):
    if level==0 and name.split('.')[0] in ('numpy','tac_osm','investigation_world'):raise ImportError('research import in exported model')
    return original(name,globals,locals,fromlist,level)
builtins.__import__=checked
runpy.run_path('runtime.py',run_name='__main__')
""";response=subprocess.run([sys.executable,'-I','-B','-c',guard],cwd=destination,input=text,text=True,capture_output=True);assert response.returncode==0,response.stderr;compare([json.loads(s) for s in response.stdout.splitlines()],expected,{'values':0,'max_abs':0.});return len(expected)


@torch.no_grad()
def run(directory,output,exports=None,veritas=None,smoke=False):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);root=Path(directory);summary=json.loads((root/'summary.json').read_text());assert summary['status']==('SMOKE_ONLY' if smoke else 'COMPLETE');assert summary['protocol_sha256']==digest(PROTOCOL)
    for name,sha in json.loads((root/'manifest.json').read_text()).items():assert digest(root/name)==sha,name
    if veritas:
        assert subprocess.check_output(['git','-C',veritas,'rev-parse','HEAD'],text=True).strip()==PIN;sys.path.insert(0,str(Path(veritas)/'src'))
    result={'status':'RUNNING','checkpoints':0,'cells':0,'trajectories':0,'native_actions':0,'portable_decisions':0,'telemetry':{'values':0,'max_abs':0.},'cross_host':'UNKNOWN until remote execution','scope':'Strict local/remote float64 restoration; synthetic native effects first seed only; no independent real-world validation.'};start=time.monotonic()
    for path in sorted(root.glob('*/**/report.json')):
        r=json.loads(path.read_text());folder=path.parent;assert digest(folder/'agent.pt')==r['checkpoint_sha256'];
        if not smoke:
            for source in ('src/tac_osm/development/belief_agent.py','scripts/data_belief_dev062.py','scripts/run_belief_dev062.py','research/dev062/protocol.json'):
                assert subprocess.check_output(['git','show',f"{r['source']}:{source}"],cwd=ROOT)==(ROOT/source).read_bytes(),source
        weights=torch.load(folder/'agent.pt',weights_only=True);gate={k.removeprefix('gate.'):v.float() for k,v in weights.items() if k.startswith('gate.')};m=make(r['arm'],r['spec'],r['seed'],gate).double().eval();m.load_state_dict(weights);cells=r['validation']
        if (folder/'test.json').exists():
            t=json.loads((folder/'test.json').read_text());cells=cells+t['cells'];fixed=torch.load(root/'test-fixed.pt',weights_only=True);compare(fixed_metrics(m,fixed,copy.deepcopy(m.gate).float()),t['fixed'],result['telemetry'])
        for c in cells:
            actual,_=rollout(m,c['seed'],c['history'],c['worlds'],c['horizon'],False,c['error'],c['inspection']);compare(actual,c,result['telemetry']);result['cells']+=1;result['trajectories']+=c['worlds']
            for row in c['rows']:
                w=BeliefWorld(c['seed'],row['index'],c['history'],c['error'],c['inspection'])
                for a in row['actions']:w.act(a['action']);assert w.actions[-1]==a
                for key,value in w.finish().items():assert row[key]==value,key
                if veritas and r['seed']==92601 and c['inspection']==.1:
                    translated=dict(row,entities=2,final_faults=row['faults'],success=row['completion'],write_cost=0.,observation_cost=row['costs']['observations']);result['native_actions']+=native_replay(translated)
        if exports and r['seed']==92601:result['portable_decisions']+=standalone(folder,r,m,Path(exports)/r['arm'])
        result['checkpoints']+=1;write(output,result);print('restored',str(folder.relative_to(root)),result['trajectories'],flush=True)
    result.update(status='COMPLETE',elapsed_seconds=time.monotonic()-start,host={'cpu_models':sorted({line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')}),'python':sys.version,'torch':str(torch.__version__),'machine':subprocess.check_output(['uname','-a'],text=True).strip()});write(output,result);print('AUDIT_COMPLETE',json.dumps({k:result[k] for k in ('status','checkpoints','trajectories','native_actions','portable_decisions','telemetry','host')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--output',required=True);p.add_argument('--exports');p.add_argument('--veritas');p.add_argument('--smoke',action='store_true');a=p.parse_args();run(a.directory,a.output,a.exports,a.veritas,a.smoke)
