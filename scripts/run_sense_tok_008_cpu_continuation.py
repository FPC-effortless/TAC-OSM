#!/usr/bin/env python3
"""Owner-requested CPU continuation: exactly 896 extra updates, 1024 total.

Resume original model, AdamW and RNG. Frozen learning-rate warm restart.
45-minute hard ceiling, no TEST, no qualification or automatic retries.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import signal
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
import torch
from tac_osm.adaptive_token_controls import TokenComparisonArm
from tac_osm.adaptive_token_training import batch_at,digest,sample_order_digest
from scripts.validate_research_parity import check_model_parameter_target


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fit-root',type=Path,required=True)
    p.add_argument('--dev-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--parent-run',type=Path,required=True)
    args=p.parse_args()
    protocol_path=ROOT/'research/mathematical-engineering/sense_tok_008_cpu_continuation/FAIR_COMPARISON.json'
    protocol_raw=protocol_path.read_bytes()
    protocol=json.loads(protocol_raw)
    cfg=protocol['config']
    if (cfg['steps'],cfg['seed'],cfg['batch_size'],cfg['history_bytes'])!=(1024,1,4,256):
        raise ValueError('approved bound changed')
    if args.output.exists():
        raise ValueError('one-fit pilot already exists; no automatic retry permitted')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(1)
    sources={'FIT':[],'EXPOSED_DEV':[]}
    omitted={}
    for row in protocol['source_files']:
        root=args.fit_root if row['split']=='FIT' else args.dev_root
        path=root/row['path']
        raw=path.read_bytes()
        git=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        if digest(raw)!=row['sha256'] or git!=row['git_blob_sha1'] or len(raw)!=row['size_bytes']:
            raise ValueError('publisher source changed: '+row['path'])
        if row['split']=='FIT':
            sources['FIT'].extend((row['path'],i,raw[i:i+256]) for i in range(0,len(raw)-255,256))
            omitted[row['path']]=len(raw)%256
        else:
            sources['EXPOSED_DEV'].append((row['path'],0,raw[:256]))
    dev_hashes={digest(raw) for _,_,raw in sources['EXPOSED_DEV']}
    original=len(sources['FIT'])
    sources['FIT']=[r for r in sources['FIT'] if digest(r[2]) not in dev_hashes]
    expected_order=sample_order_digest(sources['FIT'],seed=1,steps=1024,batch_size=4)
    model=TokenComparisonArm('adaptive_codebook',seed=1).cpu()
    counts=model.parameter_receipt()
    check_model_parameter_target(counts['trainable_parameters'],counts['deployed_parameters'])
    if counts['trainable_parameters']!=101008:
        raise ValueError('actual original model count changed')
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'FROZEN_PROTOCOL.json').write_bytes(protocol_raw)
    (args.output/'SOURCE_RECEIPT.json').write_text(json.dumps({'fit_windows':len(sources['FIT']),
        'excluded_exact_dev_window_matches':original-len(sources['FIT']), 'omitted_tail_bytes':omitted,
        'source_files':protocol['source_files'],'sample_order_sha256':expected_order,
        'qualification':'PUBLISHER_MATCHED_EXPOSED_DEV_NOT_INDEPENDENT','test_read':False},indent=2)+'\n')
    optcfg=cfg['optimizer']
    optimizer=torch.optim.AdamW(model.parameters(),lr=optcfg['lr'],betas=tuple(optcfg['betas']),
        eps=optcfg['eps'],weight_decay=optcfg['weight_decay'])
    parent_path=args.parent_run/'checkpoint-000128.pt'
    if digest(parent_path.read_bytes())!=protocol['parent_checkpoint']['sha256']:
        raise ValueError('parent checkpoint changed')
    parent=torch.load(parent_path,map_location='cpu',weights_only=True)
    if parent['step']!=128:
        raise ValueError('wrong continuation step')
    model.load_state_dict(parent['model'],strict=True)
    optimizer.load_state_dict(parent['optimizer'])
    torch.set_rng_state(parent['torch_rng'])
    if any(int(s['step'])!=128 for s in optimizer.state.values()):
        raise ValueError('AdamW moments not at parent step')
    started=time.monotonic()
    deadline=started+2700
    def stop(_signal,_frame):
        raise TimeoutError('approved 45-minute CPU budget exhausted')
    signal.signal(signal.SIGALRM,stop)
    signal.signal(signal.SIGTERM,stop)
    signal.alarm(2700)
    order=hashlib.sha256()
    previous=[json.loads(line) for line in (args.parent_run/'TRAINING.jsonl').read_text().splitlines()]
    if [r['step'] for r in previous]!=list(range(1,129)):
        raise ValueError('parent journal missing updates')
    for r in previous:
        order.update((json.dumps(r['samples'],separators=(',',':'))+'\n').encode())
    if order.hexdigest()!=parent['sample_order_sha256']:
        raise ValueError('parent sample-order mismatch')
    shutil.copy2(parent_path,args.output/'checkpoint-000128.pt')
    (args.output/'PARENT_SOURCE_RECEIPT.json').write_bytes((args.parent_run/'SOURCE_RECEIPT.json').read_bytes())
    (args.output/'PARENT_JOURNAL.jsonl').write_bytes((args.parent_run/'TRAINING.jsonl').read_bytes())
    evaluation={}
    checkpoints=[{'step':128,'path':'checkpoint-000128.pt','sha256':digest(parent_path.read_bytes())}]
    def evaluate(step):
        model.eval()
        total=0.
        rows=[]
        with torch.no_grad():
            for i,w in enumerate(sources['EXPOSED_DEV']):
                ids,y,_,_=batch_at([w],seed=1,step=i,batch_size=1,device='cpu')
                logits,(audit,),_=model.backbone(ids,audit=True)
                loss=torch.nn.functional.cross_entropy(logits.flatten(0,1).double(),y.flatten(),reduction='sum')
                total+=float(loss)
                rows.append({'file':w[0],'bytes':256,'nll_sum_nats':float(loss),
                             'byte_choices':audit.base.decisions.count('BYTE'),
                             'token_choices':audit.base.decisions.count('TOKEN'),
                             'token_events':len(audit.codes),'active_codes':audit.active_code_count})
        result={'bits_per_byte':total/(1536*math.log(2)),'rows':rows}
        evaluation[str(step)]=result
        print(json.dumps({'stage':'EXPOSED_DEV','step':step,'bpb':result['bits_per_byte']}),flush=True)
    def checkpoint(step):
        path=args.output/f'checkpoint-{step:06d}.pt'
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'torch_rng':torch.get_rng_state(),
                    'step':step,'seed':1,'protocol_sha256':digest(protocol_raw),
                    'sample_order_sha256':order.hexdigest(),'parent_checkpoint_sha256':protocol['parent_checkpoint']['sha256'],'qualification':'EXPLORATORY_OWNER_EXCEPTION'},path)
        checkpoints.append({'step':step,'path':path.name,'sha256':digest(path.read_bytes())})
    steps=128
    log=args.output/'TRAINING.jsonl'
    try:
        evaluate(128)
        if abs(evaluation['128']['bits_per_byte']-json.loads((args.parent_run/'RESULTS.json').read_bytes())['evaluations']['128']['bits_per_byte'])>1e-12:
            raise ValueError('parent DEV predictions failed resume replay')
        with log.open('x') as journal:
            for step in range(128,1024):
                if time.monotonic()>deadline:
                    raise TimeoutError('deadline reached')
                model.train()
                ids,y,_,receipt=batch_at(sources['FIT'],seed=1,step=step,batch_size=4,device='cpu')
                order.update((json.dumps(receipt,separators=(',',':'))+'\n').encode())
                local_step=step-128
                lr=cfg['final_lr']+(optcfg['lr']-cfg['final_lr'])*(local_step+1)/16 if local_step<16 else cfg['final_lr']+(optcfg['lr']-cfg['final_lr'])*.5*(1+math.cos(math.pi*(local_step-16)/879))
                for group in optimizer.param_groups:
                    group['lr']=lr
                optimizer.zero_grad(set_to_none=True)
                logits,audits,_=model.backbone(ids,mode='surrogate',audit=True)
                loss=torch.nn.functional.cross_entropy(logits.reshape(-1,256),y.flatten())
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite loss')
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
                steps=step+1
                record={'step':steps,'loss_nats':float(loss.detach()),'lr':lr,'preclip_gradient_norm':float(norm),
                        'samples':receipt,'byte_choices':sum(a.base.decisions.count('BYTE') for a in audits),
                        'token_choices':sum(a.base.decisions.count('TOKEN') for a in audits),
                        'token_events':sum(len(a.codes) for a in audits),
                        'active_codes':sorted({c.code_id for a in audits for c in a.codes})}
                journal.write(json.dumps(record)+'\n');journal.flush()
                if steps%32==0:
                    print(json.dumps({k:record[k] for k in ('step','loss_nats','byte_choices','token_choices','token_events')}),flush=True)
                if steps%128==0:
                    checkpoint(steps)
        if order.hexdigest()!=expected_order:
            raise ValueError('sample order drift')
        evaluate(1024)
        status='COMPLETED_UNQUALIFIED_CPU_CONTINUATION'
    except BaseException as exc:
        status='INCOMPLETE_'+type(exc).__name__
        raise
    finally:
        signal.alarm(0)
        result={'status':status,'steps':steps,'parameters':counts,'evaluations':evaluation,'checkpoints':checkpoints,
                'actual_sample_order_sha256':order.hexdigest(),'expected_sample_order_sha256':expected_order,
                'elapsed_seconds':time.monotonic()-started,'fits':0,'continuations':1,'additional_optimizer_steps':steps-128,'parent_checkpoint_sha256':protocol['parent_checkpoint']['sha256'],'test_read':False,'gpu_used':False,
                'qualification':'OWNER_APPROVED_NONCOMPARATIVE_CPU_EXCEPTION_NOT_CAPABILITY_EVIDENCE'}
        (args.output/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({'status':status,'steps':steps,'elapsed_seconds':result['elapsed_seconds']}),flush=True)


if __name__=='__main__':
    main()
