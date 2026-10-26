"""Fixed, FIT-only functional proposal diagnosis; never advances an optimizer."""
import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.func import functional_call
from scripts import me109_residual_training as run
from scripts.me109_observatory_adapter import authenticated_inputs
from scripts.me109_observatory_inputs import FIELDS
from tac_osm.role_optimizer import BlockProtectedAdamW

OUT = run.ROOT/'research/mathematical-engineering/diagtrain001/RESULTS.json'


class LossView(nn.Module):
    def __init__(self, model, rawscale):
        super().__init__()
        self.model = model
        self.register_buffer('rawscale', rawscale)

    def forward(self, batch):
        forecast, _, trace = self.model(*batch[:6], return_trace=True)
        total, parts = run.role_objective(self.model, forecast, trace, batch, self.rawscale)
        error = (self.model.raw_quantiles(forecast)[:, 0, 1]-batch[6][:, 0]).abs()/self.rawscale[0]
        def pm(start, stop):
            mask = batch[7][start:stop, 0]
            return torch.where(mask,error[start:stop],torch.zeros_like(error[start:stop])).sum()/mask.sum().clamp_min(1)
        return dict(total=total, whole_point=.5*parts['point'], gap_pm=pm(0,16), general_pm=pm(16,32), **parts)


def scalars(values):
    return {name: float(value.detach()) for name,value in values.items()}


def main(replay=False, observed_output=None, inward=False):
    out=OUT.parent/'INWARD_RESULTS.json' if inward else OUT
    if out.exists() and not replay:
        raise RuntimeError('Immutable diagnostic output exists')
    torch.set_num_threads(1)
    arrays, receipt = authenticated_inputs()
    batches = [[torch.from_numpy(arrays[f'batch{b}_{field}'].copy()) for field in FIELDS] for b in range(3)]
    rows=[]
    for identity in run.budget_audit()['models']:
        path=run.OUT/'residual_guarded'/f"seed-{identity['seed']}"/'final.pt'
        before_hash=run.sha(path)
        assert before_hash==identity['checkpoint_sha256']
        cp=torch.load(path,map_location='cpu',weights_only=False)
        # Construct without loading or preprocessing any source rows; original
        # buffers and every learned tensor are restored strictly from checkpoint.
        model=run.ResidualRoleSensorWorld()
        model.load_state_dict(cp['model'],strict=True)
        params=tuple(model.parameters()); named=tuple(model.named_parameters())
        assert len(params)==45 and sum(p.numel() for p in params)==104663
        _,blockmap=run.blocks(model)
        opt=BlockProtectedAdamW(run.parameter_groups(model),blocks=list(blockmap.values()))
        opt.load_state_dict(cp['optimizer'])
        assert all(a is b for a,b in zip(params,opt._params))
        old=[p.detach().clone() for p in params]
        moments=[{k:v.clone() for k,v in opt.state[p].items()} for p in params]
        versions=model.versions()
        view=LossView(model,torch.from_numpy(arrays['rawscale'].copy()))
        for b,batch in enumerate(batches):
            values=view(batch)
            grads=torch.autograd.grad(values['whole_point'],params,retain_graph=True,allow_unused=True)
            point=[torch.zeros_like(p) if g is None else g.detach() for p,g in zip(params,grads)]
            combined=torch.autograd.grad(values['total'],params,allow_unused=True)
            for p,g in zip(params,combined): p.grad=torch.zeros_like(p) if g is None else g.detach().clone()
            norm=float(torch.nn.utils.clip_grad_norm_(params,1.))
            guarded,_,guard=opt.propose(point)
            # A zero protected gradient leaves the identical AdamW proposal intact.
            raw,_,_=opt.propose([torch.zeros_like(p) for p in params])
            if inward:
                from tac_osm.inward_role_guard import inward_candidates
                guarded,guard=inward_candidates(params,point,raw,list(blockmap.values()))
            def magnitude(candidate):
                return sum(float((c.double()-p.detach().double()).square().sum()) for c,p in zip(candidate,params))**.5
            effects={}
            for which,target in [('same',b),('next',(b+1)%3)]:
                with torch.no_grad():
                    baseline=scalars(view(batches[target]))
                    current={f'model.{name}':p for name,p in named}
                    identity_result=scalars(functional_call(view,current,(batches[target],),strict=False))
                    assert baseline==identity_result
                    effects[which]=dict(batch=target,baseline=baseline)
                    for label,candidate in [('guarded',guarded),('unguarded',raw)]:
                        replacement={f'model.{name}':c.detach() for (name,_),c in zip(named,candidate)}
                        after=scalars(functional_call(view,replacement,(batches[target],),strict=False))
                        effects[which][label]=dict(after=after,delta={k:after[k]-baseline[k] for k in baseline})
            rows.append(dict(seed=identity['seed'],batch=b,checkpoint_sha256=before_hash,
                input_sha256=receipt['batches'][b]['input_sha256'],preclip_norm=norm,
                raw_displacement_norm=magnitude(raw),guarded_displacement_norm=magnitude(guarded),
                guard=guard,effects=effects))
            assert model.versions()==versions
            for p,value,state in zip(params,old,moments):
                assert torch.equal(p,value)
                for k,v in state.items(): assert torch.equal(opt.state[p][k],v)
            assert {int(s['step']) for s in opt.state.values()}=={4096}
            assert run.sha(path)==before_hash
    summary={}
    for which in ('same','next'):
        summary[which]={}
        for label in ('guarded','unguarded'):
            summary[which][label]={k:dict(increased=sum(r['effects'][which][label]['delta'][k]>0 for r in rows),
                decreased=sum(r['effects'][which][label]['delta'][k]<0 for r in rows),
                mean_delta=sum(r['effects'][which][label]['delta'][k] for r in rows)/len(rows))
                for k in ('total','whole_point','gap_pm','general_pm')}
    result=dict(study='DIAGTRAIN001-INWARD001' if inward else 'DIAGTRAIN001',scope='POSTHOC_FIT_ONLY_NO_MODEL_FITS',
        protocol_sha256=run.sha(OUT.parent/('INWARD_PROTOCOL_V1.json' if inward else 'PROTOCOL.json')),
        training_sufficiency='UNRESOLVED: no intermediate validation checkpoints or budget-controlled continuation',
        no_optimizer_steps=True,no_checkpoint_writes=True,no_report_or_protected_access=True,
        original_optimizer_steps=4096,checks=rows,summary=summary)
    if observed_output is not None:
        observed_output=Path(observed_output)
        if observed_output.exists(): raise RuntimeError('current-host output already exists')
        run.write(observed_output,result)
    if replay:
        expected=json.loads(out.read_text())
        discrepancies=[]
        def compare(a,b,path='result'):
            if isinstance(a,dict):
                for key in a: compare(a[key],b[key],path+'.'+key)
            elif isinstance(a,list):
                for i,(x,y) in enumerate(zip(a,b)): compare(x,y,path+f'[{i}]')
            else:
                try: run._compare_numeric(a,b,path)
                except (AssertionError,ValueError): discrepancies.append(dict(path=path,actual=a,original=b))
        compare(result,expected)
        print('CURRENT_HOST_SUMMARY',json.dumps(summary,sort_keys=True),flush=True)
        print('CROSS_HOST_DISCREPANCIES',json.dumps(discrepancies,sort_keys=True),flush=True)
        run._compare_numeric(result,json.loads(out.read_text()))
    else: run.write(out,result)
    print(json.dumps(summary,indent=2))
    print('DIAGTRAIN001_REPLAY_PASS' if replay else 'DIAGTRAIN001_AUDIT_PASS')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--replay',action='store_true');parser.add_argument('--observed-output',type=Path);parser.add_argument('--inward',action='store_true')
    args=parser.parse_args();main(args.replay,args.observed_output,args.inward)
