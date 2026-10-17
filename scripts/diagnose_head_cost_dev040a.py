"""Post-hoc DEV-040A diagnostics. Reuses an inspected cohort; no confirmatory claims."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from tac_osm.development.schema_maintenance import make_pair, ROLES, DEADLINES, execute, exact_public_targets  # noqa: E402

ARMS = ('bounded_mtsk', 'masked_local', 'schema41')
GRID = (0., .05, .1, .15, .2, .25, .3, .4, .5, .6, .7, .75, .8, .9, 1.)
SWEEP = (0., .1, .25, .5)
HEADS = ('original', 'mlp_uniform', 'mlp_balanced')


def auc(y, score):
    """Mann-Whitney AUC, half credit for ties; undefined for a one-class label."""
    y, score = np.asarray(y).ravel(), np.asarray(score).ravel()
    if len(y) != len(score) or not np.isfinite(score).all():
        raise ValueError('invalid AUC inputs')
    n1, n0 = int(y.sum()), int((1-y).sum())
    if not n1 or not n0:
        return None
    order = np.argsort(score, kind='stable')
    sorted_scores, labels = score[order], y[order]
    _, starts, counts = np.unique(sorted_scores, return_index=True, return_counts=True)
    ranks = np.repeat(starts + (counts + 1) / 2, counts)
    return float((np.dot(ranks, labels) - n1*(n1+1)/2) / (n1*n0))


def loss_weights(y, conditions, alpha):
    if not 0 <= alpha <= 1:
        raise ValueError('focus mass outside [0,1]')
    focus = np.zeros_like(y, dtype=bool)
    for i,c in enumerate(conditions):
        if c['deadline'] != 'before':
            focus[i,c['target_entity'],0] = True
    if not focus.any() or focus.all():
        raise ValueError('both strata needed')
    return alpha*focus/focus.sum() + (1-alpha)*(~focus)/(~focus).sum()


def physical(y, score, threshold):
    chosen = np.asarray(score) > threshold
    outcomes = np.where(chosen, y[...,1], y[...,0])
    failures = int(outcomes.sum())
    unnecessary = int((chosen & (y[...,0] == 0)).sum())
    services = int(chosen.sum())
    return dict(failures=failures, unnecessary_services=unnecessary, services=services,
                total_cost=4*failures+services, entities=chosen.size,
                cost_per_entity=(4*failures+services)/chosen.size)


def choose_threshold(y, p):
    score = p[...,0]-p[...,1]
    return min(GRID, key=lambda t:(physical(y,score,t)['total_cost'],abs(t-.25),t))


def exact_frontier(y, score):
    """All distinct strict-threshold decisions, including both endpoints."""
    flat = np.asarray(score).ravel()
    w,s = y[...,0].ravel(),y[...,1].ravel()
    order = np.argsort(-flat,kind='stable')
    _,starts,counts = np.unique(-flat[order],return_index=True,return_counts=True)
    ends=starts+counts-1
    failure=int(w.sum())+np.cumsum((s-w)[order])[ends]
    unnecessary=np.cumsum((w[order]==0).astype(int))[ends]
    return np.column_stack((np.r_[int(w.sum()),failure],np.r_[0,unnecessary])).astype(int)


def weakly_covers(a,b):
    """For every point of B, A has <= failures AND <= unnecessary services."""
    best=np.full(max(int(a[:,1].max()),int(b[:,1].max()))+1,np.inf)
    np.minimum.at(best,a[:,1],a[:,0])
    best=np.minimum.accumulate(best)
    return bool(np.all(best[b[:,1]]<=b[:,0]))


def interval(values):
    values=[float(v) for v in values]
    mean=statistics.mean(values)
    half=4.302652729911275*statistics.stdev(values)/math.sqrt(3)
    return dict(mean=mean,seed_values=values,exploratory_95_t_df2=[mean-half,mean+half])


def cases(split,reps):
    out=[]
    for delay in (32,128,512):
        for role in ROLES:
            for deadline in DEADLINES:
                for rep in range(reps):
                    out.extend(make_pair(400000+rep,512,delay,role,deadline,domain=f'dev040a-{split}-v1'))
    return out


def truth(cs):
    return np.array([[[c.wait_targets[e],c.service_targets[e],c.current_targets[e]] for e in range(8)] for c in cs])


def last_keys(case):
    latest={o.entity:o for o in case.observations}
    return [(latest[e].kind,latest[e].value,
             ('due' if latest[e].due<=case.tick+case.horizon else 'pending') if latest[e].kind=='scheduled' else 'none') for e in range(8)]


def fit_last_event(cs):
    counts={}
    y=truth(cs)
    for c,labels in zip(cs,y):
        for key,value in zip(last_keys(c),labels):
            total,n=counts.get(key,(np.zeros(3),0))
            counts[key]=(total+value,n+1)
    return {k:v/n for k,(v,n) in counts.items()},y.mean(axis=(0,1))


def predict_last_event(fit,cs):
    table,default=fit
    return np.array([[table.get(k,default) for k in last_keys(c)] for c in cs])


def save_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def label_metrics(y,p,conditions):
    result=[]
    n=len(y);targets=np.array([c['target_entity'] for c in conditions])
    for stratum,mask in [('all',np.ones(n,dtype=bool))]+[(r,np.array([c['role']==r for c in conditions])) for r in ROLES]+[(d,np.array([c['deadline']==d for c in conditions])) for d in DEADLINES]:
        for label,q,selection in [('wait_all',0,'all'),('wait_target',0,'target'),('wait_other',0,'other'),('service_all',1,'all'),('current_all',2,'all')]:
            use=np.repeat(mask[:,None],8,axis=1)
            if selection=='target':
                use &= np.arange(8)[None,:]==targets[:,None]
            if selection=='other':
                use &= np.arange(8)[None,:]!=targets[:,None]
            yy,pp=y[...,q][use],p[...,q][use]
            prob=np.clip(pp,1e-7,1-1e-7)
            ece=0.
            for lo in np.arange(0,1,.1):
                take=(pp>=lo)&(pp<(lo+.1) if lo<.9 else pp<=1)
                if take.any():ece+=take.mean()*abs(yy[take].mean()-pp[take].mean())
            result.append(dict(stratum=stratum,label=label,n=len(yy),positives=int(yy.sum()),accuracy=float(((pp>.5)==yy).mean()),
                auc=auc(yy,pp),brier=float(np.square(pp-yy).mean()),nll=float(-(yy*np.log(prob)+(1-yy)*np.log(1-prob)).mean()),ece10=float(ece)))
        result.append(dict(stratum=stratum,label='decision_score',n=int(mask.sum()*8),positives=int((y[mask,...,0]>y[mask,...,1]).sum()),accuracy=None,
            auc=auc((y[mask,...,0]-y[mask,...,1])>.25,(p[mask,...,0]-p[mask,...,1])),brier=None,nll=None,ece10=None))
    return result


def refit(ck,train,alpha,seed):
    import torch
    from torch import nn
    import torch.nn.functional as F
    torch.manual_seed(seed+41000)
    h=nn.Sequential(nn.Linear(44,64),nn.Tanh(),nn.Linear(64,2))
    h.load_state_dict(ck['heads']['original'])
    y=train['labels'];focus=torch.zeros_like(y,dtype=torch.bool)
    for i,c in enumerate(train['conditions']):
        if c['deadline']!='before':focus[i,c['target_entity'],0]=True
    w=alpha*focus.float()/focus.sum()+(1-alpha)*(~focus).float()/(~focus).sum()
    np.testing.assert_allclose(w.numpy(),loss_weights(y.numpy(),train['conditions'],alpha),atol=1e-9)
    opt=torch.optim.AdamW(h.parameters(),lr=.002,weight_decay=.0001)
    for _ in range(500):
        opt.zero_grad(set_to_none=True)
        loss=(F.cross_entropy(h(train['features']['trained']).reshape(-1,2),y.reshape(-1),reduction='none').reshape_as(y)*w).sum()
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite refit')
        loss.backward();opt.step()
    if alpha==.5:
        for k,v in h.state_dict().items():torch.testing.assert_close(v,ck['heads']['mlp_balanced'][k],rtol=0,atol=0)
    return h


def run(evidence,output):
    import torch
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    output.mkdir(parents=True,exist_ok=True)
    cs=cases('evaluation',16);dev=cases('calibration',4);training=cases('head-train',8)
    y,dy=truth(cs),truth(dev)
    for c in cs:
        assert exact_public_targets(c.observations,c.tick,c.horizon)==c.wait_targets
    table=fit_last_event(training)
    lp,ld=predict_last_event(table,cs),predict_last_event(table,dev)
    lt=choose_threshold(dy,ld)
    last=physical(y,lp[...,0]-lp[...,1],lt)
    public=physical(y,y[...,0]-y[...,1],.25)
    # Independently execute both references in the physical simulator.
    for c,p in zip(cs,lp):
        selected=tuple(e for e in range(8) if p[e,0]-p[e,1]>lt)
        actual=execute(c.world,c.horizon,selected,0)
        expected=physical(truth([c]),(p[:,0]-p[:,1])[None],lt)
        assert actual['total_cost']==expected['total_cost'] and actual['failures']==expected['failures']
    refs=dict(public=public,last_event_devfit=last,last_event_threshold=lt,
              last_event_wait_accuracy=float(((lp[...,0]>.5)==y[...,0]).mean()),
              last_event_table=[dict(key=list(k),probabilities=v.tolist()) for k,v in table[0].items()])
    ref_strata=[]
    for role in ROLES:
        for deadline in DEADLINES:
            mask=np.array([r==role and d==deadline for _delay in (32,128,512) for r in ROLES for d in DEADLINES for _rep in range(16) for _partner in (0,1)])
            for name,pred,t in (('public',y[...,0]-y[...,1],.25),('last_event_devfit',lp[...,0]-lp[...,1],lt)):
                ref_strata.append(dict(reference=name,role=role,deadline=deadline,**physical(y[mask],pred[mask],t)))
    refs['strata']=ref_strata
    save_csv(output/'reference-strata.csv',ref_strata)
    metrics=[];strata=[];frontier=[];summaries=[];raw={}
    for seed in (4001,4002,4003):
        folder=evidence/f'DEV-040A-RAW-seed{seed}-NOT-CONFIRMATORY'
        report=json.loads((folder/'report.json').read_text())
        record=dict(seed=seed,arms={})
        raw[str(seed)]={}
        for arm in ARMS:
            for name,field in ((f'{arm}.pt','checkpoint_sha256'),(f'{arm}-fit-features.pt','fit_features_sha256'),(f'{arm}-evaluation-features.pt','evaluation_features_sha256')):
                assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==report['arms'][arm][field]
            ck=torch.load(folder/f'{arm}.pt',map_location='cpu',weights_only=True)
            fit=torch.load(folder/f'{arm}-fit-features.pt',map_location='cpu',weights_only=True)
            ev=torch.load(folder/f'{arm}-evaluation-features.pt',map_location='cpu',weights_only=True)
            np.testing.assert_array_equal(ev['labels'].numpy(),y)
            rows=report['rows'][arm]
            for c,r in zip(cs,rows):assert c.fingerprint==r['fingerprint']
            ps={name:np.asarray([r['probabilities'][name] for r in rows]) for name in HEADS}
            ds={name:np.asarray(report['arms'][arm]['calibrations'][name]['probabilities']) for name in HEADS}
            # Save all new heads and inference tensors; inference is independently rerun below.
            new={}
            for alpha in SWEEP:
                name='weight_'+str(alpha)
                head=refit(ck,fit['train'],alpha,seed)
                with torch.no_grad():
                    pl=head(ev['features']['trained']);dl=head(fit['calibration']['features']['trained'])
                    ps[name]=pl.softmax(-1)[...,1].numpy();ds[name]=dl.softmax(-1)[...,1].numpy()
                new[name]=dict(weights=copy.deepcopy(head.state_dict()),logits=pl,dev_logits=dl,alpha=alpha)
                with torch.no_grad():torch.testing.assert_close(head(ev['features']['trained']),pl,rtol=0,atol=0)
            np.testing.assert_array_equal(ps['weight_0.5'],ps['mlp_balanced'])
            torch.save(new,output/f'{seed}-{arm}-weight-sweep.pt')
            raw[str(seed)][arm]={}
            arm_summary={}
            for name,p in ps.items():
                t=choose_threshold(dy,ds[name]);score=p[...,0]-p[...,1]
                result=physical(y,score,t)
                if name in HEADS:assert t==report['arms'][arm]['calibrations'][name]['threshold']
                critical=np.array([r['role']=='reissued_cancel' and r['deadline']!='before' for r in rows])
                targets=np.array([r['target_entity'] for r in rows]);idx=np.arange(len(y))
                result.update(threshold=t,reissued_accuracy=float(((p[idx,targets,0]>.5)==y[idx,targets,0])[critical].mean()))
                result['test_best_grid_cost']=min(physical(y,score,v)['cost_per_entity'] for v in GRID)
                result['calibration_cost']=physical(dy,ds[name][...,0]-ds[name][...,1],t)['cost_per_entity']
                arm_summary[name]=result
                raw[str(seed)][arm][name]=dict(probabilities=p.tolist(),calibration_probabilities=ds[name].tolist(),threshold=t)
                metrics.extend(dict(seed=seed,arm=arm,head=name,**m) for m in label_metrics(y,p,rows))
                for role in ROLES:
                    for deadline in DEADLINES:
                        mask=np.array([r['role']==role and r['deadline']==deadline for r in rows])
                        z=physical(y[mask],score[mask],t)
                        baseline=physical(y[mask],y[mask,...,0]-y[mask,...,1],.25)
                        strata.append(dict(seed=seed,arm=arm,head=name,role=role,deadline=deadline,**z,public_cost=baseline['cost_per_entity']))
                for threshold in [-1.000001]+[v/200 for v in range(201)]:
                    frontier.append(dict(seed=seed,arm=arm,head=name,threshold=threshold,**physical(y,score,threshold)))
            u=exact_frontier(y,ps['mlp_uniform'][...,0]-ps['mlp_uniform'][...,1])
            b=exact_frontier(y,ps['mlp_balanced'][...,0]-ps['mlp_balanced'][...,1])
            # Grid regret separates threshold selection from the quality of the available score.
            record['arms'][arm]=dict(heads=arm_summary,uniform_weakly_covers_balanced=weakly_covers(u,b),balanced_weakly_covers_uniform=weakly_covers(b,u))
        summaries.append(record)
    save_csv(output/'label-metrics.csv',metrics);save_csv(output/'stratified-costs.csv',strata);save_csv(output/'threshold-frontiers.csv',frontier)
    analysis=dict(study='DEV-040A-POSTHOC-DIAGNOSTICS',status='INSPECTED_COHORT_EXPLORATORY',references=refs,by_seed=summaries,cost_differences={})
    for arm in ARMS:
        differences=[(r['arms'][arm]['heads']['mlp_balanced']['cost_per_entity']-r['arms'][arm]['heads']['mlp_uniform']['cost_per_entity']) for r in summaries]
        analysis['cost_differences'][arm]=interval(differences)
    # Summarize query ranking/calibration without asserting that AUC identifies the causal mechanism.
    analysis['label_summary']={arm:{head:{label:{metric:(None if any(r[metric] is None for r in metrics if r['arm']==arm and r['head']==head and r['label']==label and r['stratum']=='all') else statistics.mean(r[metric] for r in metrics if r['arm']==arm and r['head']==head and r['label']==label and r['stratum']=='all')) for metric in ('auc','brier','nll')} for label in ('wait_target','wait_other','service_all','decision_score')} for head in ('mlp_uniform','mlp_balanced')} for arm in ARMS}
    (output/'raw-predictions.json').write_text(json.dumps(dict(labels=y.tolist(),conditions=[{k:r[k] for k in ('role','deadline','delay','target_entity','fingerprint')} for r in rows],predictions=raw),allow_nan=False))
    (output/'analysis.json').write_text(json.dumps(analysis,indent=2,allow_nan=False)+'\n')
    plot(output,frontier,summaries)
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()}
    (output/'SHA256.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('DIAGNOSTIC COMPUTATION COMPLETE; aggregate output waits for archive upload',flush=True)


def plot(out,rows,summaries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,arm in zip(axes,ARMS):
        for head in ('mlp_uniform','mlp_balanced'):
            selected=[r for r in rows if r['arm']==arm and r['head']==head]
            thresholds=sorted({r['threshold'] for r in selected})
            xx=[statistics.mean(r['unnecessary_services']/r['entities'] for r in selected if r['threshold']==t) for t in thresholds]
            yy=[statistics.mean(r['failures']/r['entities'] for r in selected if r['threshold']==t) for t in thresholds]
            ax.plot(xx,yy,label=head)
        ax.set(title=arm,xlabel='Unnecessary services / entity',ylabel='Failures / entity');ax.legend()
    fig.suptitle('DEV-040A: mean threshold tradeoff over three seeds (exploratory)');fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(out/f'failure-service-frontier.{ext}',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    weights=[0.,1/36,.1,.25,.5]
    for j,arm in enumerate(ARMS):
        for r in summaries:
            names=['weight_0.0','mlp_uniform','weight_0.1','weight_0.25','weight_0.5']
            for i,metric in enumerate(('reissued_accuracy','cost_per_entity')):
                axes[i,j].plot(weights,[r['arms'][arm]['heads'][n][metric] for n in names],marker='o',label=str(r['seed']))
                axes[i,j].set(title=arm,xlabel='Total loss mass on mature target-wait labels',ylabel=metric)
        axes[0,j].legend()
    fig.suptitle('Post-hoc dose response: fixed features, same MLP initialization, 500 updates');fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(out/f'weight-sweep.{ext}',dpi=160)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('evidence',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.evidence,args.output)
