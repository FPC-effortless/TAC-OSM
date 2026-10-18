"""Restore all DEV059 controls; exact decisions and tolerant float telemetry.

No retraining. Public trajectories use the unchanged DEV058 physical laws.
Native replay shares recorded synthetic effects, not independent real physics.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import numpy as np
import torch
from run_reliability_controls_dev059 import ROOT,PROTOCOL,create,evaluate
from run_learned_stack_dev055 import write,digest
from audit_fully_learned_dev058 import native_replay,PIN
from tac_osm.development.learned_agent_world import World


def compare(actual,reference):
    """Keep discrete actions/receipts/costs exact; floating heads may round."""
    left=dict(actual);right=dict(reference)
    left['rows']=[];right['rows']=[]
    for a,b in zip(actual['rows'],reference['rows'],strict=True):
        a=dict(a);b=dict(b)
        for key in ('completion_probabilities','stop_probabilities'):
            np.testing.assert_allclose(a.pop(key),b.pop(key),rtol=1e-5,atol=1e-6)
        left['rows'].append(a);right['rows'].append(b)
    assert left==right,'discrete decision, receipt, cost or work replay mismatch'


def export_model(model,arm,config,checkpoint,destination):
    destination.mkdir(parents=True,exist_ok=True)
    shutil.copy(ROOT/'src/tac_osm/development/learned_agent.py',destination/'agent_model.py')
    shutil.copy(ROOT/'src/tac_osm/development/reliability_controllers.py',destination/'baseline_model.py')
    shutil.copy(checkpoint,destination/'agent.pt')
    write(destination/'metadata.json',{'arm':arm,'config':config})
    (destination/'runtime.py').write_text('''import json
from pathlib import Path
import sys
import torch
from baseline_model import create
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
root=Path(__file__).resolve().parent
metadata=json.loads((root/'metadata.json').read_text())
model=create(metadata['arm'],metadata['config'],0).eval()
model.load_state_dict(torch.load(root/'agent.pt',weights_only=True,map_location='cpu'))
state=model.initial(1)
with torch.inference_mode():
    for line in sys.stdin:
        r=json.loads(line)
        if r.get('reset',False):state=model.initial(1)
        for e in r['events']:state,_=model.observe(state,torch.tensor([e],dtype=torch.float32))
        inputs=[torch.tensor([r[k]],dtype=t) for k,t in [('candidates',torch.float32),('valid',torch.bool),('queries',torch.float32),('query_valid',torch.bool)]]
        output=model(state,*inputs)
        print(json.dumps({'index':int(output['logits'].argmax(-1)),'logits':output['logits'][0].tolist()}),flush=True)
''')


@torch.no_grad()
def standalone(model,arm,config,checkpoint):
    requests=[];expected=[]
    for index in range(3):
        w=World(8595902,index,(2,4,6)[index],(32,96,128)[index]);state=model.initial(1);incoming=w.events
        for step in range(2*w.entities+4):
            d,v,q,qv,commands=w.candidates()
            requests.append({'reset':step==0,'events':[e.tolist() for e in incoming],
                             'candidates':d.tolist(),'valid':v.tolist(),'queries':q.tolist(),'query_valid':qv.tolist()})
            for e in incoming:state,_=model.observe(state,torch.from_numpy(e[None]))
            result=model(state,torch.from_numpy(d[None]),torch.from_numpy(v[None]),torch.from_numpy(q[None]),torch.from_numpy(qv[None]))
            choice=int(result['logits'].argmax(-1));expected.append({'index':choice,'logits':result['logits'][0].tolist()})
            incoming=w.act(commands[choice])
            if w.stopped:break
    with tempfile.TemporaryDirectory(prefix='dev059-export-') as name:
        destination=Path(name);export_model(model,arm,config,checkpoint,destination)
        guard='''import sys,runpy,builtins
sys.path.insert(0,'.')
import torch
original=builtins.__import__
def checked(name,*a,**k):
    if name.split('.')[0] in ('tac_osm','investigation_world'):
        raise RuntimeError('research dependency crossed deployment boundary')
    return original(name,*a,**k)
builtins.__import__=checked
runpy.run_path('runtime.py',run_name='__main__')
'''
        response=subprocess.run([sys.executable,'-I','-B','-c',guard],cwd=destination,
            input=''.join(json.dumps(r)+'\n' for r in requests),text=True,capture_output=True,check=True)
        assert [json.loads(line) for line in response.stdout.splitlines()]==expected,'standalone logits/decision mismatch'
    return len(expected)


def run(directory,output,veritas_root=None):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);started=time.monotonic()
    root=Path(directory);summary=json.loads((root/'summary.json').read_text())
    assert not summary['smoke'] and summary['status']=='COMPLETE'
    assert summary['protocol_sha256']==digest(PROTOCOL)
    for name,value in json.loads((root/'manifest.json').read_text()).items():assert digest(root/name)==value,name
    if veritas_root:
        vr=Path(veritas_root).resolve()
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=vr,text=True).strip()==PIN
        sys.path.insert(0,str(vr/'src'))
    result={'status':'RUNNING','checkpoints':0,'cells':0,'trajectories':0,'actions':0,'native_actions':0,
            'standalone_decisions':0,'complete_agent_failed_worlds':0,
            'float_telemetry_tolerance':{'rtol':1e-5,'atol':1e-6},'decision_receipt_cost_checks':'EXACT',
            'source':summary['source'],'limitation':'Shared synthetic physics and recorded native effects; independent process/host engineering verification, not independent physical replication.'}
    tuning=json.loads((root/'tuning.json').read_text());fits=json.loads((root/'fits.json').read_text())
    for report in tuning+fits:
        prefix='tune-' if report in tuning else ''
        if prefix:
            candidates=[p for p in root.glob('tune-*/report.json') if json.loads(p.read_text())==report]
            assert len(candidates)==1;directory=candidates[0].parent
        else:directory=root/f"{report['arm']}-{report['seed']}"
        assert report['status']=='COMPLETE' and report['updates']==1200
        assert digest(directory/'agent.pt')==report['checkpoint_sha256']
        model=create(report['arm'],report['config'],report['seed']).eval()
        model.load_state_dict(torch.load(directory/'agent.pt',weights_only=True))
        cells=list(report['validation']);test=directory/'test.json'
        if test.exists():cells+=json.loads(test.read_text())['cells']
        for cell in cells:
            actual=evaluate(model,cell['seed'],cell['entities'],cell['history'],cell['worlds'],cell['horizon_extra'])
            compare(actual,cell);result['cells']+=1
            for row in cell['rows']:
                world=World(row['seed'],row['index'],row['entities'],row['history'])
                assert __import__('hashlib').sha256(np.stack(world.events).astype(np.float32).tobytes()).hexdigest()==row['public_initial_sha256']
                for recorded in row['actions']:
                    world.act((recorded['kind'],recorded['target']))
                    assert world.actions[-1]==recorded
                physical=world.finish(row['writes'])
                for key in ('success','stopped','final_faults','cost','action_cost','observations','write_cost','failed_actions'):
                    assert physical[key]==row[key]
                result['trajectories']+=1;result['actions']+=len(row['actions'])
                result['complete_agent_failed_worlds']+=int(not row['success'])
                if veritas_root:result['native_actions']+=native_replay(row)
        result['standalone_decisions']+=standalone(model,report['arm'],report['config'],directory/'agent.pt')
        result['checkpoints']+=1
        print('restored',report['arm'],report['seed'],report['lr'],flush=True)
    result.update(status='PASS',seconds=time.monotonic()-started)
    write(output,result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',required=True);parser.add_argument('--output',required=True);parser.add_argument('--veritas-root')
    args=parser.parse_args();run(args.directory,args.output,args.veritas_root)
