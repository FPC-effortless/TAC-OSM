"""Restore every DEV061 checkpoint; exact trajectories, native replay, portability.

Float telemetry tolerates bounded CPU rounding; discrete actions/costs remain
exact. Veritas effects share synthetic physics, not independent real-world laws.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import numpy as np
import torch
from run_structured_memory_dev061 import ROOT,PROTOCOL,ResearchWorld,create,evaluate,prepare,raw_data,read_targets
from run_learned_stack_dev055 import digest,write
from audit_fully_learned_dev058 import native_replay,PIN


def compare(actual,expected,drift):
    a=dict(actual);b=dict(expected);a['rows']=[];b['rows']=[]
    for left,right in zip(actual['rows'],expected['rows'],strict=True):
        left=dict(left);right=dict(right)
        for key in ('completion_probability','stop_probability','current_brier'):
            x=np.array(left.pop(key));y=np.array(right.pop(key))
            assert x.shape==y.shape and np.isfinite(x).all() and np.isfinite(y).all()
            np.testing.assert_allclose(x,y,atol=1e-4,rtol=1e-5)
            drift['values']+=int(x.size);drift['max_absolute_difference']=max(drift['max_absolute_difference'],float(np.max(np.abs(x-y),initial=0)))
        a['rows'].append(left);b['rows'].append(right)
    assert a==b,'Changed action, receipt, charged cost, retrieval count or work'


def export(checkpoint,arm,config,destination):
    destination.mkdir(parents=True,exist_ok=True)
    for source,name in [('learned_agent.py','agent_model.py'),('temporal_agent.py','temporal_model.py'),('structured_memory.py','memory_model.py')]:shutil.copy(ROOT/'src/tac_osm/development'/source,destination/name)
    tree=ast.parse((ROOT/'src/tac_osm/development/structured_memory.py').read_text())
    chosen='PublicFeatureMLP' if arm=='public_mlp' else 'MemoryAgent'
    tree.body=[node for node in tree.body if isinstance(node,(ast.Import,ast.ImportFrom,ast.Try)) or isinstance(node,ast.ClassDef) and node.name==chosen]
    constructor='PublicFeatureMLP(config["width"],seed)' if arm=='public_mlp' else 'MemoryAgent(config,arm,seed)'
    (destination/'memory_model.py').write_text(ast.unparse(tree)+'\n\ndef create(arm,config,seed=0):\n    return '+constructor+'\n')
    if arm=='generic':shutil.copy(ROOT/'src/tac_osm/development/inference_memory.py',destination/'inference_model.py')
    shutil.copy(checkpoint,destination/'agent.pt');write(destination/'metadata.json',{'arm':arm,'config':config,'runtime':'skip_closed_metadata_copy' if arm=='generic' else 'registered'})
    (destination/'runtime.py').write_text('''import json
from pathlib import Path
import sys
import torch
from memory_model import create
torch.set_num_threads(1)
root=Path(__file__).resolve().parent
meta=json.loads((root/'metadata.json').read_text())
if meta['arm']=='generic':
    from inference_model import InferenceMemoryAgent
    model=InferenceMemoryAgent(meta['config'],'generic').eval()
else:model=create(meta['arm'],meta['config']).eval()
model.load_state_dict(torch.load(root/'agent.pt',weights_only=True,map_location='cpu'),strict=True)
state=model.initial(1)
with torch.inference_mode():
    for line in sys.stdin:
        r=json.loads(line)
        if r.get('reset',False):state=model.initial(1)
        for event in r['events']:state,_=model.observe(state,torch.tensor([event],dtype=torch.float32))
        inputs=[torch.tensor([r[k]],dtype=t) for k,t in [('candidates',torch.float32),('valid',torch.bool),('queries',torch.float32),('query_valid',torch.bool)]]
        out=model(state,*inputs)
        response={'index':int(out['logits'].argmax(-1)),'logits':out['logits'][0].tolist(),
                  'completion_probability':float(out['completion'][0].sigmoid()),
                  'current_fault_probability':out['current_logits'][0].sigmoid().tolist()}
        if 'outcomes' in out:
            response['next_fault_probability']=out['outcomes'][0].sigmoid().tolist()
            response['predicted_action_cost']=(20*out['cost'][0]).tolist()
        print(json.dumps(response),flush=True)
''')
    (destination/'README.txt').write_text('DEV061 experimental, bounded teacher-trained policy; NOT promoted or production-certified.\nPython+PyTorch only. python runtime.py < requests.jsonl\nPublic22-vector events,13x11 candidates,13valid mask,6x8 queries,6query_valid mask.\nGeneric: learned admission/addressing. Entity variants: explicit public categorical-ID routing,16 recurrent entity/relation slots+16 FIFO episodic slots; GRU updates/CDL/neural CASM/predictions/action+stop scores learned.\nThe public_mlp comparator deliberately uses a task-specific public feature parser.\nNo simulator, teacher, planner, repair rule or stop override is imported by the runtime.\nNeural CASM is not symbolic program discovery. No learned continuous identity association, uncertainty/POMDP or C5-scaling claim.\nIndependent execution verification is still required.\n')


@torch.no_grad()
def standalone(model,checkpoint,arm,config,destination=None):
    requests=[];expected=[]
    for i in range(3):
        w=ResearchWorld(5611109,i,(2,4,6)[i],(32,96,128)[i]);state=model.initial(1);incoming=w.events
        for step in range(2*w.entities+4):
            d,v,q,qv,commands=w.candidates()
            requests.append({'reset':step==0,'events':[e.tolist() for e in incoming],'candidates':d.tolist(),'valid':v.tolist(),'queries':q.tolist(),'query_valid':qv.tolist()})
            for e in incoming:state,_=model.observe(state,torch.from_numpy(e[None]))
            out=model(state,torch.from_numpy(d[None]),torch.from_numpy(v[None]),torch.from_numpy(q[None]),torch.from_numpy(qv[None]))
            choice=int(out['logits'].argmax(-1));response={'index':choice,'logits':out['logits'][0].tolist(),'completion_probability':float(out['completion'][0].sigmoid()),'current_fault_probability':out['current_logits'][0].sigmoid().tolist()}
            if 'outcomes' in out:response.update(next_fault_probability=out['outcomes'][0].sigmoid().tolist(),predicted_action_cost=(20*out['cost'][0]).tolist())
            expected.append(response);incoming=w.act(commands[choice])
            if w.stopped:break
    with tempfile.TemporaryDirectory(prefix='dev061-export-') as temp:
        folder=Path(temp);export(checkpoint,arm,config,folder)
        guard="""import sys,runpy,builtins
sys.path.insert(0,'.')
import torch
original=builtins.__import__
def checked(name,*args,**kwargs):
    if name.split('.')[0] in ('tac_osm','investigation_world','numpy'):raise RuntimeError('research dependency crossed portable boundary')
    return original(name,*args,**kwargs)
builtins.__import__=checked
runpy.run_path('runtime.py',run_name='__main__')
"""
        response=subprocess.run([sys.executable,'-I','-B','-c',guard],cwd=folder,input=''.join(json.dumps(r)+'\n' for r in requests),text=True,capture_output=True,check=True)
        actual=[json.loads(line) for line in response.stdout.splitlines()]
        assert actual==expected,'standalone logits/action mismatch'
        if destination:
            export(checkpoint,arm,config,destination)
            (destination/'demo-requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in requests))
            (destination/'demo-expected.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in expected))
    return len(expected)


def physical(row,result,native):
    w=ResearchWorld(row['seed'],row['index'],row['entities'],row['history'],row['family'])
    assert hashlib.sha256(np.asarray(w.events,np.float32).tobytes()).hexdigest()==row['public_initial_sha256']
    for recorded in row['actions']:
        w.act((recorded['kind'],recorded['target']));assert w.actions[-1]==recorded
    finish=w.finish(row['writes'])
    for key in ('success','stopped','final_faults','cost','action_cost','observations','write_cost','failed_actions'):assert finish[key]==row[key],key
    result['trajectories']+=1;result['actions']+=len(row['actions']);result['failed_rollouts_all_arms_splits']+=int(not row['success'])
    if native:result['native_actions']+=native_replay(row)


@torch.no_grad()
def run(directory,output,veritas=None,exports=None):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);root=Path(directory)
    if veritas:
        pin=subprocess.check_output(['git','-C',str(veritas),'rev-parse','HEAD'],text=True).strip();assert pin==PIN
        sys.path.insert(0,str(Path(veritas)/'src'))
    summary=json.loads((root/'summary.json').read_text());assert summary['status']=='COMPLETE' and summary['test_opened'];assert summary['protocol_sha256']==digest(PROTOCOL)
    for name,expected in json.loads((root/'manifest.json').read_text()).items():assert digest(root/name)==expected,name
    result={'status':'RUNNING','checkpoints':0,'trajectories':0,'actions':0,'native_actions':0,'standalone_decisions':0,'cells':0,
       'failed_rollouts_all_arms_splits':0,'telemetry_drift':{'values':0,'max_absolute_difference':0.},'write_counters':{},
       'scope':'Separate restoration/physical accounting/native execution. Shared synthetic physics; no independent real-world validation.'}
    p=json.loads(PROTOCOL.read_text());fixed_raw=raw_data(p['data']['fixed_query_seed'],p['data']['fixed_test_snapshots'])
    start=time.monotonic()
    reports=json.loads((root/'tuning.json').read_text())+json.loads((root/'fits.json').read_text())
    for source in {r['source'] for r in reports}:
        for name in ('scripts/run_structured_memory_dev061.py','src/tac_osm/development/structured_memory.py'):
            assert subprocess.check_output(['git','show',f'{source}:{name}'],cwd=ROOT)==(ROOT/name).read_bytes(),'Fitting/evaluation source changed after registration'
    for report in reports:
        arm,seed=report['arm'],report['seed']
        folder=next(path.parent for path in root.glob('*/report.json') if json.loads(path.read_text())['arm']==arm and json.loads(path.read_text())['seed']==seed and json.loads(path.read_text())['lr']==report['lr'])
        checkpoint=folder/'agent.pt';assert digest(checkpoint)==report['checkpoint_sha256']
        model=create(arm,report['config'],seed).eval();model.load_state_dict(torch.load(checkpoint,weights_only=True),strict=True)
        counters={'admitted':0,'episodic_writes':0,'observed_present':0};observe=model.observe
        def counted(state,event,present=None,force_write=False):
            updated,trace=observe(state,event,present,force_write);counters['admitted']+=int(trace['writes']);counters['episodic_writes']+=int(trace.get('episodic_writes',0));counters['observed_present']+=len(event) if present is None else int(present.sum());return updated,trace
        model.observe=counted
        cells=report['validation']
        if (folder/'test.json').exists():cells=cells+json.loads((folder/'test.json').read_text())['cells']
        for cell in cells:
            actual=evaluate(model,cell['seed'],cell['entities'],cell['history'],cell['worlds'],cell['family'])
            compare(actual,cell,result['telemetry_drift']);result['cells']+=1
            for row in cell['rows']:physical(row,result,bool(veritas))
        if (folder/'test.json').exists():
            expected=json.loads((folder/'test.json').read_text());fixed=prepare(model,fixed_raw);state=model.training_state(fixed)
            out=model(state,fixed['descriptors'],fixed['valid'],fixed['queries'],fixed['query_valid']);qv=fixed['query_valid']
            brier=((out['current_logits'].sigmoid()-fixed['current'])**2)
            np.testing.assert_allclose(float(brier[qv].mean()),expected['fixed_current_brier'],rtol=1e-5,atol=1e-4)
            error=float(((out['current_logits']>=0)!=fixed['current'].bool())[qv].float().mean());assert error==expected['fixed_current_error']
            for i,row in enumerate(expected['fixed_rows']):np.testing.assert_allclose(float(brier[i,qv[i]].mean()),row['brier'],rtol=1e-5,atol=1e-4)
            if 'state_read_logits' in out:
                labels=read_targets(state,fixed['queries']);chosen=out['state_read_logits'][:,:,0].argmax(-1);raw=state[0][:,:,-22:]
                for i,row in enumerate(expected['fixed_rows']):
                    select=chosen[i];latest=labels[i];matching=(raw[i,select.clamp_max(31),4:12].argmax(-1)==fixed['queries'][i].argmax(-1))&(select<32)
                    matching=matching&((raw[i,select.clamp_max(31),0]>0)|(raw[i,select.clamp_max(31),2]>0))
                    stale=matching&(raw[i,select.clamp_max(31),21]<raw[i,latest.clamp_max(31),21]-1e-7)&(latest<32)
                    actual={'queries':int(qv[i].sum()),'latest_hits':int(((select==latest)&qv[i]).sum()),'stale':int((stale&qv[i]).sum()),'wrong':int((~matching&qv[i]).sum()),'latest_available':int(((latest<32)&qv[i]).sum())}
                    assert all(actual[k]==row[k] for k in actual),'fixed retrieval mismatch'
        result['write_counters'][folder.name]=counters
        if (folder/'test.json').exists():result['standalone_decisions']+=standalone(model,checkpoint,arm,report['config'],Path(exports)/arm if exports and seed==9101 else None)
        result['checkpoints']+=1;print('audited',folder.name,flush=True)
    result['elapsed_seconds']=time.monotonic()-start;result['status']='PASS';write(output,result)


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--directory',required=True);a.add_argument('--output',required=True);a.add_argument('--veritas-root');a.add_argument('--export-root');args=a.parse_args();run(args.directory,args.output,args.veritas_root,args.export_root)
