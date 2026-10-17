#!/usr/bin/env python3
"""Frozen forward dynamics and selected local Jacobians, not retraining."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
PIN='ec0ff80b058308dad8de57eab7d223ec9a81dcba'


def trace(checkout,pilot,continuation,output):
    checkout=Path(checkout);pilot=Path(pilot);continuation=Path(continuation);output=Path(output)
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout,text=True).strip()==PIN
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=checkout,text=True).strip()
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip()
    if output.exists():raise ValueError('preserve earlier traces')
    output.mkdir(parents=True)
    spec=importlib.util.spec_from_file_location('dynamics_tool',ROOT/'src/tac_osm/development/mtsk_dynamics.py')
    dynamics=importlib.util.module_from_spec(spec);spec.loader.exec_module(dynamics)
    assert 'tac_osm' not in sys.modules
    spec=importlib.util.spec_from_file_location('frozen',checkout/'scripts/run_local_mtsk_dev034.py')
    frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
    torch.set_num_threads(1)
    partial=json.loads((ROOT/'docs/DEV-034-PARTIAL-AUDIT.json').read_text())
    continued=json.loads((ROOT/'docs/DEV-034-CONTINUATION-REPLAY.json').read_text())
    snapshots=[('initial_multi','mtsk_local_multi',None,None),
        ('failure_multi139','mtsk_local_multi',pilot/'mtsk_local_multi.failure.pt',partial['arms']['mtsk_local_multi']['failure_snapshot_sha256']),
        ('continued_multi216','mtsk_local_multi',continuation/'continuation.pt',continued['checkpoint_sha256'])]
    for arm in ('mtsk_local_tied','mtsk_global_multi','mtsk_global_tied'):
        snapshots.append((arm+'216',arm,pilot/(arm+'.pt'),partial['arms'][arm]['checkpoint_sha256']))
    records=[];matrices={};conditional={};full_trace=[];source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for name,arm,path,expected_hash in snapshots:
        model=frozen.build(arm);model.eval()
        if path is not None:
            assert hashlib.sha256(path.read_bytes()).hexdigest()==expected_hash
            model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['model'])
        kernel=model.memory
        for step in (0,139,215):
            cases=frozen.training_cases(step,False)
            inp=frozen.public_stream_tensors([c.observations for c in cases])
            receipts=[(c.world.act(frozen.Action('inspect',random.Random(f'dev034:{step}:{b}').randrange(8))),) for b,c in enumerate(cases)]
            obs=frozen.public_stream_tensors(receipts)
            with torch.no_grad():
                x=torch.cat((model.event_encoder(inp[0]),model.event_encoder(obs[0][:,0])[:,None]),1)
                entities=torch.cat((inp[1],obs[1][:,0,None]),1)
                state=kernel.initial(x[:,0])
            points={};maxima=[(-1.,None),(-1.,None)];count=0;amplifying=0;maximum=0.;state_max=0.;failure=None;raw_gains=[]
            with torch.no_grad():
                for t in range(513):
                    if kernel.local:
                        index=entities[:,t,None,None].expand(-1,1,kernel.bank_scalars)
                        old=state.gather(1,index).squeeze(1);features=x[:,t]+kernel.bank_embedding[entities[:,t]]
                    else:old=state;features=x[:,t,None]+kernel.bank_embedding
                    parts=dynamics.components(kernel,old,features)
                    if not torch.isfinite(parts['next']).all():
                        failure=dict(position=t,reason='nonfinite forward proposal');break
                    assert torch.equal(parts['next'],kernel.proposal(old,features))
                    gain=parts['row_gain'];raw_gains.append(gain.clone());count+=gain.numel();amplifying+=int((gain>1+1e-6).sum());maximum=max(maximum,float(gain.max()))
                    for b in range(2):
                        bank=int(entities[b,t]) if kernel.local else int(gain[b].amax(dim=(-2,-1)).argmax())
                        i=(b,) if kernel.local else (b,bank)
                        rowgain=float(gain[i].max())
                        point=dict(snapshot=name,training_step=step,partner=b,position=t,bank=bank,event_entity=int(entities[b,t]),
                            conditional_row_gain=rowgain,packed=old.clone(),features=features.clone(),index=i)
                        if t in (0,255,511,512):points[(b,t)]=point
                        if rowgain>maxima[b][0]:maxima[b]=(rowgain,point)
                        full_trace.append(dict(snapshot=name,training_step=step,partner=b,position=t,bank=bank,
                            conditional_row_gain=rowgain,max_memory=float(parts['memory'][i].abs().max())))
                    state=kernel.update(state,x[:,t],entities[:,t]);state_max=max(state_max,float(state.abs().max()))
                if failure is None:
                    direct=model.encode(*inp);direct=model.update(direct,*(z[:,0] for z in obs))
                    assert torch.equal(direct,state),'original batched encoder/receipt replay differs'
            conditional[f'{name}:{step}']=torch.stack(raw_gains) if raw_gains else torch.empty(0,dtype=torch.float64)
            for _,point in maxima:
                if point is not None:points[(point['partner'],point['position'])]=point
            jacobians=[]
            for key,point in sorted(points.items()):
                try:
                    metrics=dynamics.jacobian_metrics(kernel,point['packed'],point['features'],point['index'])
                except FloatingPointError as error:
                    clean={k:v for k,v in point.items() if k not in ('packed','features','index')}
                    clean['jacobian_failure']=str(error);jacobians.append(clean);continue
                matrix=metrics.pop('jacobian');matrix_key=f'{name}:{step}:{key[0]}:{key[1]}'
                matrices[matrix_key]=matrix
                clean={k:v for k,v in point.items() if k not in ('packed','features','index')}
                clean.update(metrics,matrix_key=matrix_key);jacobians.append(clean)
            row=dict(snapshot=name,arm=arm,checkpoint_sha256=expected_hash,training_step=step,
                fingerprints=[c.fingerprint for c in cases],local_writer=kernel.local,
                conditional_row_observations=count,conditional_rows_amplifying=amplifying,
                maximum_conditional_row_gain=maximum,max_abs_augmented_state=state_max,
                forward_failure=failure,selected_jacobians=jacobians)
            records.append(row)
            (output/'progress.json').write_text(json.dumps(dict(source=source,records=records),indent=2,allow_nan=False)+'\n')
            print(json.dumps({k:row[k] for k in ('snapshot','training_step','maximum_conditional_row_gain','conditional_rows_amplifying','forward_failure')}),flush=True)
    path=output/'selected-jacobians.pt';torch.save(matrices,path)
    gainpath=output/'conditional-gains.pt';torch.save(conditional,gainpath)
    (output/'per-event-trace.json').write_text(json.dumps(full_trace,allow_nan=False)+'\n')
    report=dict(experiment='DEV-035 frozen MTSK dynamics',source=source,core_source=PIN,
        protocol_sha256=hashlib.sha256((ROOT/'research/dev035/protocol.json').read_bytes()).hexdigest(),
        records=records,conditional_sha256=hashlib.sha256(gainpath.read_bytes()).hexdigest(),jacobian_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        trace_sha256=hashlib.sha256((output/'per-event-trace.json').read_bytes()).hexdigest(),
        optimizer_steps=0,scientific_test_accessed=False,architecture_improvement_established=False,
        limitation='One-step conditional row bounds and selected state Jacobians, not long-product growth, parameter gradients, information retention or an independent seed replication')
    (output/'dynamics-report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkout',required=True);p.add_argument('--pilot',required=True);p.add_argument('--continuation',required=True);p.add_argument('--output',required=True);a=p.parse_args();trace(a.checkout,a.pilot,a.continuation,a.output)
