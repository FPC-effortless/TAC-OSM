"""Registered semantic-interface probes; known laws are EVALUATION ONLY."""
import argparse
import json
from pathlib import Path
import torch
from run_semantic_value_dev069 import parent_model, body_hash, ROOT
from diagnose_world_interface_dev068 import intervene
from run_learned_stack_dev055 import write, digest


def run(directory, parent, output):
    torch.set_num_threads(1); torch.set_deterministic_debug_mode('error')
    root=Path(directory); summary=json.loads((root/'summary.json').read_text())
    result=dict(status='NOT_RUN_TEST_CLOSED', checkpoints=0, worlds=0, per_fit={},
        scope='Known-law expected future/cost inputs to unchanged learned Q; registered OOD robustness probe, not training/deployment/oracle optimality',
        source_sha256=digest(__file__), known_world_code_sha256=digest(ROOT/'scripts/diagnose_world_interface_dev068.py'),
        whole_model_qualified=False,commercial_qualified=False)
    if summary['status']!='TEST_COMPLETE':
        assert not summary['tests_opened'];write(output,result);return
    for path in sorted(root.glob('group-*/*-report.json')):
        r=json.loads(path.read_text()); folder=path.parent
        assert digest(folder/f"{r['arm']}.pt")==r['checkpoint_sha256']
        model,_=parent_model(parent,r['parent_seed'],r['parent_arm'],r['seed'],r['readout'])
        model.load_state_dict(torch.load(folder/f"{r['arm']}.pt",weights_only=True));model.eval()
        assert body_hash(model)==r['body_sha256']
        recorded=json.loads((folder/f"{r['arm']}-test.json").read_text())
        cells=[]
        for c in recorded['primary']:
            if c['history'] not in (128,1024):continue
            assert c['error']==.1 and c['inspection']==.1
            measured=intervene(model,c['seed'],c['history'],len(c['rows']))
            ordinary_faults=sum(row['faults'] for row in c['rows'])
            cells.append(dict(history=c['history'],predicted_completion=c['completion'],
                predicted_fault_bits=ordinary_faults,known_world=measured,
                completion_degradation=c['completion']-measured['completions']/measured['worlds'],
                extra_faults=measured['fault_bits']-ordinary_faults))
            result['worlds']+=measured['worlds']
        result['per_fit'][f"{r['arm']}/{r['seed']}"]=cells
        result['checkpoints']+=1;write(output,result)
    result['per_arm']={}
    for arm in sorted({name.split('/')[0] for name in result['per_fit']}):
        rows=[cells for name,cells in result['per_fit'].items() if name.startswith(arm+'/')]
        result['per_arm'][arm]=dict(seeds=len(rows),
            degradation_passing_seeds=sum(all(c['completion_degradation']<=.02 for c in cells) for cells in rows),
            no_extra_faults=all(c['extra_faults']<=0 for cells in rows for c in cells))
    result['status']='COMPLETE';write(output,result)
    print(json.dumps({k:v for k,v in result.items() if k!='per_fit'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--directory',required=True)
    parser.add_argument('--parent',required=True);parser.add_argument('--output',required=True)
    a=parser.parse_args();run(a.directory,a.parent,a.output)
