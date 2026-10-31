#!/usr/bin/env python3
"""Read-only checkpoint replay and endpoint diagnostics; never optimizes."""
import argparse
from collections import Counter
import math
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
    if results['steps']!=4096:
        raise ValueError('incomplete run, do not report completed fit')
    parent_journal=[json.loads(line) for line in (a.run/'PARENT_JOURNAL.jsonl').read_text().splitlines()]
    continuation=[json.loads(line) for line in (a.run/'TRAINING.jsonl').read_text().splitlines()]
    assert len(parent_journal)==1024 and len(continuation)==3072
    journal=parent_journal+continuation
    assert [r['step'] for r in journal]==list(range(1,4097))
    fit_bytes={}
    for row in source['source_files']:
        if row['split']=='FIT':
            raw=(a.fit_root/row['path']).read_bytes()
            assert digest(raw)==row['sha256']
            fit_bytes[row['path']]=raw
    windows=[(name,i,raw[i:i+256]) for name,raw in fit_bytes.items() for i in range(0,len(raw)-255,256)]
    journal_digest=hashlib.sha256()
    prefix_digests={}
    for row in journal:
        _,_,_,receipt=batch_at(windows,seed=1,step=row['step']-1,batch_size=4,device='cpu')
        assert json.dumps(row['samples'])==json.dumps(receipt)
        for name,offset,expected in row['samples']:
            assert digest(fit_bytes[name][offset:offset+256])==expected
        journal_digest.update((json.dumps(row['samples'],separators=(',',':'))+'\n').encode())
        prefix_digests[row['step']]=journal_digest.hexdigest()
    assert journal_digest.hexdigest()==source['sample_order_sha256']
    states={}
    records=[]
    predictions={}
    parameter_names=[name for name,_ in TokenComparisonArm('adaptive_codebook',seed=1).named_parameters()]
    for row in results['checkpoints']:
        path=a.run/row['path']
        assert digest(path.read_bytes())==row['sha256']
        cp=torch.load(path,map_location='cpu',weights_only=True)
        assert cp['step']==row['step']
        assert cp['sample_order_sha256']==prefix_digests[cp['step']]
        records.append({'step':cp['step'],'sha256':row['sha256'],
                        'optimizer_state_records':len(cp['optimizer']['state']),
                        'optimizer_parameter_steps':sorted({int(s['step']) for s in cp['optimizer']['state'].values()}),
                        'optimizer_steps_by_parameter':{parameter_names[i]:int(s['step']) for i,s in cp['optimizer']['state'].items()}})
        assert len(cp['optimizer']['state'])==len(parameter_names)==11
        token_updates=sum(r['token_events']>0 for r in journal[:cp['step']])
        for name,count in records[-1]['optimizer_steps_by_parameter'].items():
            expected=token_updates if name in ('backbone.position_recency','backbone.token_codes.weight') else cp['step']
            assert count==expected,(name,count,expected)
        assert all(bool(torch.isfinite(v).all()) for v in cp['model'].values())
        if cp['step'] not in (1024,4096):
            continue
        model=TokenComparisonArm('adaptive_codebook',seed=1).eval()
        model.load_state_dict(cp['model'],strict=True)
        assert sum(v.numel() for v in model.parameters())==101008
        states[cp['step']]=cp['model']
        full=[]
        total=0.
        forced={'byte':0.,'token':0.}
        forced_equal={'byte':True,'token':True}
        code_counts=Counter()
        code_payloads={}
        route_counts=Counter()
        updates=0
        for i,r in enumerate(x for x in source['source_files'] if x['split']=='EXPOSED_DEV'):
            raw=(a.dev_root/r['path']).read_bytes()
            assert digest(raw)==r['sha256']
            w=(r['path'],0,raw[:256])
            ids,y,_,_=batch_at([w],seed=1,step=i,batch_size=1,device='cpu')
            with torch.no_grad():
                out,(route,),_=model.backbone(ids,audit=True)
                route_counts.update(route.base.decisions)
                updates+=route.base.recurrent_updates
                for event in route.codes:
                    code_counts[event.code_id]+=1
                    code_payloads.setdefault(event.code_id,set()).add(event.payload)
                full.append(out)
                total+=float(torch.nn.functional.cross_entropy(out.reshape(-1,256).double(),y.flatten(),reduction='sum'))
                for action in forced:
                    f=model.backbone(ids,force=action)
                    forced[action]+=bits_per_byte(f,y)/6
                    forced_equal[action] &= torch.equal(out,f)
                changed=ids.clone();changed[:,128:]=(changed[:,128:]+31)%256
                other=model(changed)
                assert torch.equal(out[:,:128],other[:,:128]),'future byte poison changed earlier prediction'
        joined=torch.cat(full,dim=0).contiguous()
        bpb=total/(1536*__import__('math').log(2))
        assert abs(bpb-results['evaluations'][str(cp['step'])]['bits_per_byte'])<1e-12
        prediction_file=a.output.with_name(a.output.stem+f'-logits-{cp["step"]}.pt')
        torch.save(joined,prediction_file)
        predictions[str(cp['step'])]={'logits_file':prediction_file.name,'logits_sha256':hashlib.sha256(joined.numpy().tobytes()).hexdigest(),
            'bpb':bpb,'forced_mode_bpb_posthoc':forced,'forced_mode_logits_bitwise_equal':forced_equal,'future_prefix_invariance':'PASS',
            'routing_decisions':dict(route_counts),'recurrent_updates':updates,'code_event_counts':dict(sorted(code_counts.items())),
            'distinct_payloads_by_code':{k:len(v) for k,v in sorted(code_payloads.items())},
            'code_assignment_entropy_bits':-sum((v/sum(code_counts.values()))*math.log2(v/sum(code_counts.values())) for v in code_counts.values()) if code_counts else None,
            'code_reuse_note':('Observed repeated IDs; payload diversity and entropy do not establish semantic identities.' if code_counts else 'No completed token event; assignment entropy is undefined and code reuse cannot be measured.')}
    changes={name:float((states[4096][name]-states[1024][name]).double().norm()) for name in states[1024]}
    assert any(x>0 for x in changes.values())
    assert max(records[-1]['optimizer_parameter_steps'])==4096
    assert records[0]['sha256']==results['parent_checkpoint_sha256']
    assert results['additional_optimizer_steps']==3072
    assert len(records)==7
    protocol=json.loads((a.run/'FROZEN_PROTOCOL.json').read_bytes())
    assert records[0]['sha256']==protocol['parent_checkpoint']['sha256']
    cfg=protocol['config']
    for i,row in enumerate(continuation):
        expected=cfg['final_lr']+(cfg['optimizer']['lr']-cfg['final_lr'])*(i+1)/16 if i<16 else cfg['final_lr']+(cfg['optimizer']['lr']-cfg['final_lr'])*.5*(1+math.cos(math.pi*(i-16)/3055))
        assert row['lr']==expected
    assert results['actual_sample_order_sha256']==source['sample_order_sha256']
    # Posthoc representation intervention on one exposed DEV prefix. No tuning.
    first=next(r for r in source['source_files'] if r['split']=='EXPOSED_DEV')
    raw=(a.dev_root/first['path']).read_bytes()[:256]
    ids,y,_,_=batch_at([(first['path'],0,raw)],seed=1,step=0,batch_size=1,device='cpu')
    intervention=codebook_interventions(model.backbone,ids,y)
    report={'status':'PASS_READ_ONLY_LOCAL_REPLAY_NOT_INDEPENDENT_QUALIFICATION',
            'intermediate_checkpoint_sample_order_verified':True,'checkpoint_records':records,'prediction_replay':predictions,'parameter_change_l2':changes,
            'posthoc_single_prefix_codebook_intervention':intervention,
            'journal_steps':len(journal),'continuation_steps':len(continuation),'branch_dependent_adam_counters_verified':True,'batches_without_token_events':sum(r['token_events']==0 for r in journal),'learning_rate_schedule_verified':True,'parent_checkpoint_unchanged':True,'sample_order_verified':True,'test_accessed':False,
            'new_optimizer_steps':0,'note':'Forced modes and value intervention are posthoc, same-checkpoint diagnostics, not trained comparator results.'}
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'prediction_replay':predictions,'changed_parameter_tensors':sum(x>0 for x in changes.values())}),flush=True)


if __name__=='__main__':
    main()
