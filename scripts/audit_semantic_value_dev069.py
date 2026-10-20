"""Recorded-only replay of frozen-body readout fits; no optimizer steps."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import torch
from run_semantic_value_dev069 import ROOT, parent_model, body_hash, rollout
from run_learned_stack_dev055 import write, digest
from audit_identity_dev0615 import compare
from audit_fully_learned_dev058 import native_replay, PIN
from audit_public_outcome_dev068 import export as parent_export
from data_belief_dev062 import BeliefWorld


def export(folder, report, destination):
    adapter = dict(report, spec=report['parent_spec'])
    parent_export(folder, adapter, destination)
    text=(ROOT/'src/tac_osm/development/semantic_value.py').read_text()
    text=text.replace('from .public_outcome import', 'from agent_outcome import')
    (destination/'agent_value.py').write_text(text)
    runtime=(ROOT/'scripts/runtime_public_outcome_dev068.py').read_text()
    runtime=runtime.replace('from agent_outcome import PublicOutcomeAgent', 'from agent_value import SemanticValueAgent')
    runtime=runtime.replace("PublicOutcomeAgent(spec['backend'], spec['width'], spec['aux'])", "SemanticValueAgent(spec['backend'], spec['width'], spec['aux'], readout=meta['readout'], policy_seed=meta['seed'])")
    (destination/'runtime.py').write_text(runtime)
    metadata=json.loads((destination/'metadata.json').read_text())
    metadata.update(readout=report['readout'], parent_checkpoint_sha256=report['parent_checkpoint_sha256'],
        body_sha256=report['body_sha256'], whole_world_model_qualified=False,
        evidence_stage='UNQUALIFIED_HEAD_ONLY_RESEARCH_EXPORT')
    write(destination/'metadata.json',metadata)
    with (destination/'README.txt').open('a') as out:
        out.write('\nDEV069: all memory/world weights are frozen from rejected DEV068 parents.\n'
                  'Only the value readout is retrained. Expected arm uses learned outcome utilities\n'
                  'and learned positive cost sensitivity; no physical utility/action rule is supplied.\n'
                  'No whole-world-model, CASM superiority or commercial qualification.\n')


@torch.no_grad()
def standalone(folder, report, model, destination):
    export(folder, report, destination)
    world=BeliefWorld(6691002,0,128)
    state=model.initial(torch.tensor(world.identities,dtype=torch.float64)[None])
    pending=world.events.copy();requests=[];expected=[]
    for depth in range(10):
        candidates,permutation=world.candidates()
        requests.append(dict(reset=depth==0,keys=world.identities.tolist(),events=[e.tolist() for e in pending],
            candidates=candidates.tolist(),prices=[world.error,world.inspection],remaining_fraction=(10-depth)/10))
        for event in pending:state=model.observe(state,torch.tensor(event,dtype=torch.float64)[None])
        out=model.decide(state,torch.tensor(candidates,dtype=torch.float64)[None],
            torch.tensor([[world.error,world.inspection]],dtype=torch.float64),torch.tensor([[(10-depth)/10]],dtype=torch.float64))
        index=int(model.choice(out)[0])
        expected.append(dict(index=index,scores=out['logits'][0].tolist(),paired_measurement_probabilities=out['posterior'][0].tolist(),
            future_verification_probabilities=out['future'][0].softmax(-1).tolist(),normalized_physical_cost_predictions=out['cost'][0].tolist()))
        start=len(world.events);world.act(int(permutation[index]));pending=world.events[start:]
        if world.stopped:break
    text=''.join(json.dumps(r)+'\n' for r in requests)
    (destination/'demo-requests.jsonl').write_text(text)
    (destination/'demo-expected.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in expected))
    guard="""import sys,runpy,builtins
sys.path.insert(0,'.')
import torch
original=builtins.__import__
def checked(name,globals=None,locals=None,fromlist=(),level=0):
    if level==0 and name.split('.')[0] in ('numpy','tac_osm','investigation_world'):raise ImportError('research dependency forbidden')
    return original(name,globals,locals,fromlist,level)
builtins.__import__=checked
runpy.run_path('runtime.py',run_name='__main__')
"""
    response=subprocess.run([sys.executable,'-I','-B','-c',guard],cwd=destination,input=text,text=True,capture_output=True)
    assert response.returncode==0,response.stderr
    compare([json.loads(line) for line in response.stdout.splitlines()],expected,dict(values=0,max_abs=0.))
    (destination/'demo-output.jsonl').write_text(response.stdout)
    return len(expected)


@torch.no_grad()
def run(directory,parent,output,exports=None,veritas=None):
    torch.set_num_threads(1);torch.set_deterministic_debug_mode('error');start=time.monotonic()
    root=Path(directory);parent=Path(parent);summary=json.loads((root/'summary.json').read_text())
    assert summary['status'] in ('VALIDATION_REJECTED','TEST_COMPLETE','SMOKE_ONLY','ADMINISTRATIVELY_STOPPED')
    for name,sha in json.loads((root/'manifest.json').read_text()).items():assert digest(root/name)==sha,name
    if summary['status'] in ('VALIDATION_REJECTED','ADMINISTRATIVELY_STOPPED'):
        assert not summary['tests_opened'] and not (root/'TEST-OPENED.json').exists()
        assert not list(root.glob('group-*/*-test.json'))
    files=['src/tac_osm/development/semantic_value.py','scripts/run_semantic_value_dev069.py',
        'research/dev069-semantic-value/PROTOCOL.md','research/dev069-semantic-value/protocol.json']
    for name in files:assert subprocess.check_output(['git','show',summary['source']+':'+name],cwd=ROOT)==(ROOT/name).read_bytes(),name
    if veritas:
        assert subprocess.check_output(['git','-C',veritas,'rev-parse','HEAD'],text=True).strip()==PIN
        sys.path.insert(0,str(Path(veritas)/'src'))
    result=dict(status='RUNNING',checkpoints=0,raw_checkpoints=0,training_snapshots=0,trajectories=0,native_actions=0,
        portable_decisions=0,body_hash_cases=0,semantic_cases=0,clock_cases=0,telemetry=dict(values=0,max_abs=0.),
        frozen_source=summary['source'],tests_opened=summary['tests_opened'],commercial_qualified=False,whole_world_model_qualified=False)
    reports=sorted(root.glob('*/*-report.json'));assert len(reports)==summary['fits']
    export_group='group-96901' if (root/'group-96901/summary.json').exists() else 'pilot-0'
    generator=torch.Generator().manual_seed(969009)
    for path in reports:
        r=json.loads(path.read_text());folder=path.parent;assert r['source']==summary['source']
        model,pr=parent_model(parent,r['parent_seed'],r['parent_arm'],r['seed'],r['readout'],folder.name.startswith('pilot-'))
        assert pr['checkpoint_sha256']==r['parent_checkpoint_sha256']
        assert body_hash(model)==r['body_sha256'];result['body_hash_cases']+=1
        for suffix in ('-raw',''):
            weights=torch.load(folder/f"{r['arm']}{suffix}.pt",weights_only=True)
            model.load_state_dict(weights);assert body_hash(model)==r['body_sha256'];result['body_hash_cases']+=1
        assert digest(folder/f"{r['arm']}.pt")==r['checkpoint_sha256'];model.eval();result['raw_checkpoints']+=1
        snapshot=torch.load(folder/'training-state.pt',weights_only=True)
        assert snapshot['source']==summary['source'] and snapshot['update']==r['updates']
        assert snapshot['lrs'][r['arm']]==r['lr'] and snapshot['curves'][r['arm']]==r['curves']
        for kind,suffix in [('ema',''),('raw','-raw')]:
            final=torch.load(folder/f"{r['arm']}{suffix}.pt",weights_only=True)
            assert final.keys()==snapshot[kind][r['arm']].keys()
            for name in final:assert torch.equal(final[name],snapshot[kind][r['arm']][name]),name
        cells=list(r['validation']);test=folder/f"{r['arm']}-test.json"
        if test.exists():
            value=json.loads(test.read_text());cells+=value['primary']+value['secondary']
        for cell in cells:
            actual=rollout(model,cell['seed'],cell['history'],len(cell['rows']),cell['error'],cell['inspection'])
            compare(actual,cell,result['telemetry']);result['trajectories']+=len(cell['rows'])
            if veritas and folder.name in ('pilot-0','group-96901'):
                for index,row in enumerate(cell['rows']):
                    world=BeliefWorld(cell['seed'],index,cell['history'],cell['error'],cell['inspection'])
                    for action in row['actions']:world.act(action)
                    native=dict(world.finish(),entities=2,final_faults=row['faults'],success=row['completion'],
                        write_cost=0.,observation_cost=row['costs']['observations'])
                    result['native_actions']+=native_replay(native)
        if r['readout']=='expected':
            x=torch.randn(8,30,generator=generator,dtype=torch.float64);y=x.clone()
            x[:,:21]=x[:,:21].softmax(-1);y[:,:21]=y[:,:21].softmax(-1)
            y[:,21:]=x[:,21:];mixed=.3*x+.7*y
            torch.testing.assert_close(model.policy(mixed),.3*model.policy(x)+.7*model.policy(y),atol=1e-11,rtol=1e-11)
            costlier=x.clone();costlier[:,21]+=.5
            assert bool((model.policy(costlier)>=model.policy(x)).all());result['semantic_cases']+=2
        world=BeliefWorld(6691002,0,32);events=torch.tensor(world.events,dtype=torch.float64)[None]
        keys=torch.tensor(world.identities,dtype=torch.float64)[None];candidates,_=world.candidates()
        data=dict(events=events,present=torch.ones(events.shape[:2],dtype=torch.bool),keys=keys,
            candidates=torch.tensor(candidates,dtype=torch.float64)[None],prices=torch.tensor([[.1,.1]],dtype=torch.float64),
            remaining=torch.ones(1,1,dtype=torch.float64))
        base=model(**data)
        for scale,shift in ((1,10000),(1000,0)):
            changed={**data,'events':events.clone()};changed['events'][:,:,42]=events[:,:,42]*scale+shift
            altered=model(**changed)
            for key in ('posterior','future','cost','logits'):torch.testing.assert_close(base[key],altered[key],atol=0,rtol=0)
            result['clock_cases']+=1
        if exports and folder.name==export_group:result['portable_decisions']+=standalone(folder,r,model,Path(exports)/r['arm'])
        result['checkpoints']+=1;write(output,result);print('restored',str(path.relative_to(root)),flush=True)
    result['partial_training_snapshots']=0;result['partial_frozen_body_cases']=0
    for folder in sorted(root.glob('group-*')):
        if (folder/'summary.json').exists():continue
        snapshot=torch.load(folder/'training-state.pt',weights_only=True)
        seed=int(folder.name.removeprefix('group-'));parent_seed=96801+seed-96901
        assert snapshot['source']==summary['source'] and snapshot['update']%1024==0
        assert snapshot['update']<8192
        for arm in snapshot['raw']:
            readout=arm.rsplit('_',1)[-1];backend=arm.rsplit('_',1)[0]+'_outcome'
            model,_=parent_model(parent,parent_seed,backend,seed,readout)
            expected_hash=body_hash(model)
            for kind in ('raw','ema'):
                model.load_state_dict(snapshot[kind][arm]);assert body_hash(model)==expected_hash
                result['partial_frozen_body_cases']+=1
        result['partial_training_snapshots']+=1
    result['training_snapshots']=len({p.parent for p in reports})
    result.update(status='COMPLETE',elapsed_seconds=time.monotonic()-start,source_files_sha256={name:digest(ROOT/name) for name in files})
    write(output,result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--parent',required=True)
    p.add_argument('--output',required=True);p.add_argument('--exports');p.add_argument('--veritas')
    a=p.parse_args();run(a.directory,a.parent,a.output,a.exports,a.veritas)
