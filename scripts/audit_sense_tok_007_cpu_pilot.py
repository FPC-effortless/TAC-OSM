#!/usr/bin/env python3
"""Read-only checkpoint replay and endpoint diagnostics; never optimizes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import torch
from tac_osm.adaptive_token_controls import TokenComparisonArm
from tac_osm.adaptive_token_training import batch_at,digest
from tac_osm.adaptive_token_evaluation import codebook_interventions,bits_per_byte


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--dev-root',type=Path,required=True)
    p.add_argument('--fit-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    results=json.loads((a.run/'RESULTS.json').read_bytes())
    source=json.loads((a.run/'SOURCE_RECEIPT.json').read_bytes())
    if results['steps']!=128:
        raise ValueError('incomplete run, do not report completed fit')
    journal=[json.loads(line) for line in (a.run/'TRAINING.jsonl').read_text().splitlines()]
    assert [r['step'] for r in journal]==list(range(1,129))
    fit_bytes={}
    for row in source['source_files']:
        if row['split']=='FIT':
            raw=(a.fit_root/row['path']).read_bytes()
            assert digest(raw)==row['sha256']
            fit_bytes[row['path']]=raw
    journal_digest=hashlib.sha256()
    for row in journal:
        for name,offset,expected in row['samples']:
            assert digest(fit_bytes[name][offset:offset+256])==expected
        journal_digest.update((json.dumps(row['samples'],separators=(',',':'))+'\n').encode())
    assert journal_digest.hexdigest()==source['sample_order_sha256']
    states={}
    records=[]
    predictions={}
    for row in results['checkpoints']:
        path=a.run/row['path']
        assert digest(path.read_bytes())==row['sha256']
        cp=torch.load(path,map_location='cpu',weights_only=True)
        assert cp['step']==row['step']
        records.append({'step':cp['step'],'sha256':row['sha256'],
                        'optimizer_state_records':len(cp['optimizer']['state']),
                        'optimizer_parameter_steps':sorted({int(s['step']) for s in cp['optimizer']['state'].values()})})
        if cp['step'] not in (0,128):
            continue
        model=TokenComparisonArm('adaptive_codebook',seed=1).eval()
        model.load_state_dict(cp['model'],strict=True)
        assert sum(v.numel() for v in model.parameters())==101008
        states[cp['step']]=cp['model']
        full=[]
        total=0.
        forced={'byte':0.,'token':0.}
        for i,r in enumerate(x for x in source['source_files'] if x['split']=='EXPOSED_DEV'):
            raw=(a.dev_root/r['path']).read_bytes()
            assert digest(raw)==r['sha256']
            w=(r['path'],0,raw[:256])
            ids,y,_,_=batch_at([w],seed=1,step=i,batch_size=1,device='cpu')
            with torch.no_grad():
                out=model(ids)
                full.append(out)
                total+=float(torch.nn.functional.cross_entropy(out.reshape(-1,256).double(),y.flatten(),reduction='sum'))
                for action in forced:
                    f=model.backbone(ids,force=action)
                    forced[action]+=bits_per_byte(f,y)/6
                changed=ids.clone();changed[:,128:]=(changed[:,128:]+31)%256
                other=model(changed)
                assert torch.equal(out[:,:128],other[:,:128]),'future byte poison changed earlier prediction'
        joined=torch.cat(full,dim=0).contiguous()
        bpb=total/(1536*__import__('math').log(2))
        assert abs(bpb-results['evaluations'][str(cp['step'])]['bits_per_byte'])<1e-12
        prediction_file=a.output.with_name(a.output.stem+f'-logits-{cp["step"]}.pt')
        torch.save(joined,prediction_file)
        predictions[str(cp['step'])]={'logits_file':prediction_file.name,'logits_sha256':hashlib.sha256(joined.numpy().tobytes()).hexdigest(),
            'bpb':bpb,'forced_mode_bpb_posthoc':forced,'future_prefix_invariance':'PASS'}
    changes={name:float((states[128][name]-states[0][name]).double().norm()) for name in states[0]}
    assert any(x>0 for x in changes.values())
    assert max(records[-1]['optimizer_parameter_steps'])==128
    assert results['actual_sample_order_sha256']==source['sample_order_sha256']
    # Posthoc representation intervention on one exposed DEV prefix. No tuning.
    first=next(r for r in source['source_files'] if r['split']=='EXPOSED_DEV')
    raw=(a.dev_root/first['path']).read_bytes()[:256]
    ids,y,_,_=batch_at([(first['path'],0,raw)],seed=1,step=0,batch_size=1,device='cpu')
    intervention=codebook_interventions(model.backbone,ids,y)
    report={'status':'PASS_READ_ONLY_LOCAL_REPLAY_NOT_INDEPENDENT_QUALIFICATION',
            'checkpoint_records':records,'prediction_replay':predictions,'parameter_change_l2':changes,
            'posthoc_single_prefix_codebook_intervention':intervention,
            'journal_steps':len(journal),'sample_order_verified':True,'test_accessed':False,
            'new_optimizer_steps':0,'note':'Forced modes and value intervention are posthoc, same-checkpoint diagnostics, not trained comparator results.'}
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'prediction_replay':predictions,'changed_parameter_tensors':sum(x>0 for x in changes.values())}),flush=True)


if __name__=='__main__':
    main()
