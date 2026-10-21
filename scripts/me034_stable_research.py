"""ME034 locked stable-input study. Final labels are a separate, once-only stage."""
import argparse
import csv
import io
import json
import math
import pickle
import platform
import resource
import shutil
import subprocess
import time
import zipfile
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from tac_osm.me034_stable_inputs import COUNTS,build_model,EWMABank
from tac_osm.me032_conditional_memory import FrozenContext
from scripts.me033_audit import verify as verify_parent
from scripts.me033_semantic_research import offered_items,save_progress,predict
from scripts.me031_crossfit_research import ROOT,SEEDS,WEIGHTS,subset,targets,roles_manifest,load_source,atomic_checkpoint,file_sha,write_json
from scripts.me032_conditional_research import context_features
from scripts.me025_real_transformer_parity import fit_only_popularity
from scripts.me014b_exact_action_memory import batch,publisher_bytes,member_for_zip,REQUIRED,VOCAB,DEV_CUT,END_CUT
from scripts.me019_authentic_history import original_128_windows,SOURCE_SHA
from scripts.me022_native_readout import scores

ARMS=tuple(COUNTS)
PROTOCOL=ROOT/'research/mathematical-engineering/me034/PROTOCOL.md'
FINAL_MOD=6


def auxiliary(data,context):
    duration=torch.cat((data['metadata'][:,[32,64,96],0],data['duration']),-1)
    return torch.cat((context,duration.unsqueeze(-1)),-1)


def restore_context(state):
    c=FrozenContext(state['emb.weight'],float(state['duration_mean']),float(state['duration_scale']),
        state['item_popularity'],state['feature_mean'],state['feature_scale'])
    c.load_state_dict(state,strict=True)
    return torch.jit.script(c).eval()


def construct(arm,inc,seed):
    return build_model(arm,inc['common_state'],inc['embedding'],seed,inc['context_state']['item_popularity'],
        inc['ewma_mean'],inc['ewma_scale'],inc['global_logit'])


def fit_one(arm,data,context,inc,seed,name,out):
    start=time.process_time();net=torch.jit.script(construct(arm,inc,seed))
    groups=[[],[]]
    for n,p in net.named_parameters():
        if p.requires_grad:groups[int(not n.startswith('memory.'))].append(p)
    opt=torch.optim.AdamW([{'params':groups[0],'lr':.001},{'params':groups[1],'lr':.003}],weight_decay=1e-4)
    gen=torch.Generator().manual_seed(seed+300);y=targets(data);offered=offered_items(data)
    trace=[];charged=time.process_time()-start
    before={k:v.detach().clone() for k,v in net.state_dict().items()}
    active=[p for p in net.parameters() if p.requires_grad]
    for step in range(1,301):
        t=time.process_time();idx=torch.randint(len(y),(128,),generator=gen)
        net.train();opt.zero_grad(set_to_none=True)
        loss=(F.binary_cross_entropy_with_logits(net(data['items'][idx],data['metadata'][idx],offered[idx],context[idx]),y[idx],reduction='none')*WEIGHTS).sum(-1).mean()
        if not torch.isfinite(loss):raise ArithmeticError('nonfinite authentic loss')
        loss.backward();gn=torch.nn.utils.clip_grad_norm_(active,1.)
        if not torch.isfinite(gn):raise ArithmeticError('nonfinite authentic gradient')
        memory_grad=sum(float(p.grad.square().sum()) for n,p in net.named_parameters() if n.startswith('memory.') and p.requires_grad and p.grad is not None)**.5
        embedding_grad=sum(float(p.grad.square().sum()) for n,p in net.named_parameters() if n.endswith('emb.weight') and p.grad is not None)**.5
        opt.step();dt=time.process_time()-t;charged+=dt
        trace.append({'step':step,'real_loss':float(loss.detach()),'cpu_seconds':dt,'memory_gradient_after_clip':memory_grad,'embedding_gradient_after_clip':embedding_grad})
        if step in (100,200,300):save_progress(out,name,net,opt,gen,step,trace,{'arm':arm,'seed':seed})
        if charged>180:raise RuntimeError('registered S CPU cap exceeded; preserve partial')
    net.eval();net.save(str(out/(name+'.jit.pt')))
    changed=[n for n,v in net.state_dict().items() if not torch.equal(v,before[n])]
    if arm!='native_adapt':assert not any(n.endswith('emb.weight') for n in changed)
    if arm not in ('common_adapt','ewma_mlp','current_only'):
        assert not any(n.startswith('common.') for n in changed)
        assert any(t['memory_gradient_after_clip']>0 for t in trace)
    return {'arm':arm,'name':name,'seed':seed,'steps':300,'histories':38400,'responses':153600,
        'trainable_parameters':sum(p.numel() for p in net.parameters() if p.requires_grad),
        'stored_model_parameters':sum(p.numel() for p in net.parameters()),
        'charged_cpu_seconds':charged,'changed_tensor_names':changed,
        'positive_memory_gradient_steps':sum(t['memory_gradient_after_clip']>0 for t in trace),
        'positive_embedding_gradient_steps':sum(t['embedding_gradient_after_clip']>0 for t in trace)}


def fit_tree(data,context,seed,name,out):
    import sklearn
    from threadpoolctl import threadpool_limits
    from sklearn.ensemble import HistGradientBoostingClassifier
    start=time.process_time()
    # Same 26 common channels AND prior logit, all four real targets. Practical
    # reference: boosting exposure and parameterization cannot match Adam steps.
    x=context[...,:27].reshape(-1,27).numpy();y=targets(data).reshape(-1).numpy()
    w=np.broadcast_to(WEIGHTS.numpy(),(len(data['y']),4)).reshape(-1)
    model=HistGradientBoostingClassifier(max_iter=100,learning_rate=.05,max_leaf_nodes=15,
        min_samples_leaf=20,l2_regularization=1.,early_stopping=False,random_state=seed)
    with threadpool_limits(limits=1):model.fit(x,y,sample_weight=w)
    cpu=time.process_time()-start
    if cpu>180:raise RuntimeError('tree CPU cap exceeded')
    with (out/(name+'.pickle')).open('wb') as fd:pickle.dump(model,fd,protocol=5)
    return {'arm':'boosted','name':name,'seed':seed,'boosting_iterations':100,
        'training_rows':len(y),'charged_cpu_seconds':cpu,'sklearn':sklearn.__version__,
        'serialized_bytes':(out/(name+'.pickle')).stat().st_size,'parameter_parity':False,'exposure_parity':False}


def fit(out):
    out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    start=time.process_time();wall=time.perf_counter()
    parent=verify_parent(ROOT/'results/me033')
    data=load_source(ROOT/'results/me022')['train'];roles=roles_manifest(data)
    verification_cpu=time.process_time()-start
    assert roles==json.loads((ROOT/'results/me033/roles.json').read_text())
    write_json(out/'roles.json',roles)
    record={'study':'ME034_STABLE_INPUT_NATIVE_RECURRENCE','complete':False,'fit_complete':False,
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'protocol_sha256':file_sha(PROTOCOL),'parent_results_sha256':file_sha(ROOT/'results/me033/results.json'),
        'source_snapshot_sha256':file_sha(ROOT/'results/me022/data-snapshot.pt'),
        'shared_source_verification_CPU_seconds':verification_cpu,
        'python':platform.python_version(),'torch':torch.__version__,'threads':1,
        'hardware':Path('/proc/cpuinfo').read_text().split('model name')[1].splitlines()[0],
        'cells':[],'final_labels_opened':False,'all_models_frozen_before_scoring':False,
        'candidate_promoted':False,'parity':{
            'native_frozen_vs_gru_parameters':'PASS 8236 versus8244; gap0.0972%',
            'data_exposure':'PASS neural arms: shared R-only parents and identical S sampling/four targets',
            'selection':'PASS fixed300, no search or held-out checkpoint selection',
            'evaluation':'PASS identical CV targets and one final new-user evaluation',
            'full_compute':'FAIL full FLOP coverage unavailable; operator subtotal measured, CPU actual spend unequal',
            'other_parameters':'FAIL intentional freezing treatment and unmatched practical references',
            'boosted_exposure':'FAIL separate practical reference, not equal-Adam exposure'}}
    write_json(out/'results.json',record)
    for seed in SEEDS:
        for rotation,role in enumerate(roles):
            t=time.process_time();tag=f'r{rotation}-s{seed}'
            cell=next(c for c in parent['cells'] if c['seed']==seed and c['rotation']==rotation)
            inc=torch.load(ROOT/'results/me033'/(tag+'-inherited.pt'),weights_only=True)
            r,s=[subset(data,torch.tensor(role[k]['indices'])) for k in ('representation','readout')]
            cs=inc['context_state'];context=restore_context(cs)
            rc=auxiliary(r,context_features(context,r));sc=auxiliary(s,inc['context_S'])
            temp=EWMABank(cs['item_popularity'],torch.zeros(12),torch.ones(12))
            with torch.no_grad():features=temp.features(r['items'],r['metadata'],offered_items(r),rc).reshape(-1,12)
            inc['ewma_mean']=features.mean(0);inc['ewma_scale']=features.std(0,unbiased=False).clamp_min(.05)
            inc['global_logit']=float(torch.logit(r['y'].mean().clamp(.0001,.9999)))
            # Independent lineage assertions: E, popularity, duration all trained
            # solely on representation users, not readout/evaluation users.
            encoder=ROOT/'results/me031'/f'encoder-r{rotation}-s{seed}.pt'
            old=torch.load(encoder,weights_only=True)
            assert torch.equal(old['state_dict']['emb.weight'],inc['embedding'])
            assert torch.equal(fit_only_popularity(r),cs['item_popularity'])
            assert float(r['duration'].mean())==float(cs['duration_mean'])
            sets=[set(role[k]['users']) for k in ('representation','readout','evaluation')]
            assert all(not sets[a]&sets[b] for a,b in ((0,1),(0,2),(1,2)))
            assert all(u%16==0 for us in sets for u in us)
            lineage={'R_users':len(sets[0]),'S_users':len(sets[1]),'T_users':len(sets[2]),
                'R_intersect_S':0,'R_intersect_T':0,'S_intersect_T':0,
                'encoder_sha256':file_sha(encoder),'R_embedding_exactly_inherited':True,
                'R_only_popularity_and_duration_recomputed':True,'pretraining_rerun':False}
            atomic_checkpoint(out/(tag+'-inherited.pt'),inc)
            shutil.copyfile(ROOT/'results/me033'/('context-'+tag+'.jit.pt'),out/('context-'+tag+'.jit.pt'))
            prep=time.process_time()-t;students=[]
            offset=(SEEDS.index(seed)+rotation)%len(ARMS)
            model_seed=seed+rotation*1000+68000
            for arm in ARMS[offset:]+ARMS[:offset]:
                st=fit_one(arm,s,sc,inc,model_seed,arm+'-'+tag,out);students.append(st)
                print(json.dumps({'frozen_model':st['name'],'cpu_seconds':st['charged_cpu_seconds'],'final_labels_opened':False}),flush=True)
            tree=fit_tree(s,sc,model_seed,'boosted-'+tag,out)
            shared=cell['parent_shared_charged_cpu_seconds']+prep
            if any(shared+st['charged_cpu_seconds']>300 for st in (*students,tree)):
                raise RuntimeError('registered full-pipeline cap exceeded')
            record['cells'].append({'tag':tag,'seed':seed,'rotation':rotation,'students':students,'tree':tree,
                'shared_charged_cpu_seconds':shared,'new_preparation_cpu_seconds':prep,'lineage':lineage})
            write_json(out/'results.json',record)
    record['fit_complete']=True;record['all_models_frozen_before_scoring']=True
    record['fit_physical_CPU_seconds']=time.process_time()-start;record['fit_wall_seconds']=time.perf_counter()-wall
    record['peak_RSS_KiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write_json(out/'frozen-payloads.json',{p.name:file_sha(p) for p in out.iterdir() if p.is_file() and p.name!='results.json'})
    write_json(out/'results.json',record)
    record['crossfit_evaluation']=evaluate(out,data,record,False)
    write_json(out/'results.json',record)


def verify_frozen(out):
    r=json.loads((out/'results.json').read_text())
    if not r['fit_complete'] or not r['all_models_frozen_before_scoring']:raise ValueError('not all models frozen')
    if file_sha(PROTOCOL)!=r['protocol_sha256']:raise ValueError('protocol changed')
    for name,digest in json.loads((out/'frozen-payloads.json').read_text()).items():
        if file_sha(out/name)!=digest:raise ValueError('frozen artifact hash mismatch: '+name)
    return r


def read_final(out):
    """Reject all other user IDs BEFORE interpreting any measurement or outcome."""
    raw,source=publisher_bytes()
    if source['zip_sha256']!=SOURCE_SHA:raise ValueError('publisher SHA mismatch')
    people={};stats={'other_user_rows_skipped_before_outcomes':0,'invalid_rows':0,'eligible_records':0}
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        member=member_for_zip(z);source['member']=member
        with z.open(member) as fd:
            reader=csv.DictReader(io.TextIOWrapper(fd,encoding='utf-8-sig',newline=''))
            if not set(REQUIRED).issubset(reader.fieldnames or []):raise ValueError('source schema changed')
            for row in reader:
                try:uid=int(row['user_id'])
                except (ValueError,TypeError):continue
                if uid<0 or uid%16!=FINAL_MOD:
                    stats['other_user_rows_skipped_before_outcomes']+=1;continue
                try:ts=float(row['timestamp'])
                except (ValueError,TypeError):continue
                if not math.isfinite(ts) or ts<=1e9 or ts>=END_CUT:continue
                try:item=int(row['video_id']);d=float(row['video_duration']);ratio=float(row['watch_ratio'])
                except (ValueError,TypeError):stats['invalid_rows']+=1;continue
                if not 0<=item<VOCAB or not math.isfinite(d) or d<=0 or not math.isfinite(ratio) or ratio<0:
                    stats['invalid_rows']+=1;continue
                people.setdefault(uid,[]).append((ts,item,math.log1p(d)/12.,int(ratio>=1.)))
                stats['eligible_records']+=1
    for rows in people.values():rows.sort(key=lambda x:x[0])
    data=original_128_windows(people,DEV_CUT,END_CUT,limit=6000)
    if len(set(data['users']))<100 or len(data['y'])<1500:raise ValueError('registered final cohort insufficient; no replacement')
    atomic_checkpoint(out/'final-data.pt',{'data':data,'source':source,'parse_stats':stats})
    return data,source,stats


def tree_predict(path,ctx):
    with path.open('rb') as f:model=pickle.load(f)
    return model.predict_proba(ctx[:,-1,:27].numpy())[:,1].astype(np.float64)


def evaluate(out,data,record,final):
    from sklearn.metrics import roc_auc_score
    roles=json.loads((out/'roles.json').read_text());arms=(*ARMS,'boosted')
    n=len(data['y']);preds={s:{a:np.zeros(n) for a in arms} for s in SEEDS};metrics=[]
    for cell in record['cells']:
        idx=torch.arange(n) if final else torch.tensor(roles[cell['rotation']]['evaluation']['indices'])
        t=subset(data,idx);tag=cell['tag']
        inc=torch.load(out/(tag+'-inherited.pt'),weights_only=True)
        if final:ctx=auxiliary(t,context_features(torch.jit.load(str(out/('context-'+tag+'.jit.pt'))),t))
        else:ctx=auxiliary(t,inc['context_T'])
        pp={}
        for st in cell['students']:
            net=torch.jit.load(str(out/(st['name']+'.jit.pt'))).eval()
            pp[st['arm']]=predict(net,t,ctx)
        pp['boosted']=tree_predict(out/(cell['tree']['name']+'.pickle'),ctx)
        for arm,p in pp.items():
            if final:preds[cell['seed']][arm]+=p/3.
            else:preds[cell['seed']][arm][idx.numpy()]=p
            np.save(out/(('final-' if final else 'cv-')+arm+'-'+tag+'-predictions.npy'),p)
            metrics.append({'seed':cell['seed'],'rotation':cell['rotation'],'arm':arm,'metrics':scores(p,t['y'].numpy())})
    y=data['y'].numpy().ravel();pooled={str(s):{a:scores(p,y) for a,p in row.items()} for s,row in preds.items()}
    means={a:{k:float(np.mean([pooled[str(s)][a][k] for s in SEEDS])) for k in ('Brier','log_loss','ROC_AUC','ECE10')} for a in arms}
    result={'pooled_per_seed':pooled,'means':means,'metrics_per_cell':metrics,'rows_per_seed':n,'users':len(set(data['users'])),
        'independent_first_exposure':final,'final_rotations_averaged_within_seed':final}
    if final:
        for s,row in preds.items():
            for a,p in row.items():np.save(out/f'final-ensemble-{a}-s{s}.npy',p)
        users,inv=np.unique(data['users'],return_inverse=True);counts=np.bincount(inv)
        delta=np.mean([(preds[s]['native_frozen']-y)**2-(preds[s]['common_adapt']-y)**2 for s in SEEDS],axis=0)
        sums=np.bincount(inv,weights=delta);rng=np.random.default_rng(34034);bd=[];ad=[]
        for _ in range(3000):
            draw=rng.integers(len(users),size=len(users));w=np.bincount(draw,minlength=len(users))[inv]
            bd.append(sums[draw].sum()/counts[draw].sum())
            ad.append(np.mean([roc_auc_score(y,preds[s]['native_frozen'],sample_weight=w)-roc_auc_score(y,preds[s]['common_adapt'],sample_weight=w) for s in SEEDS]))
        bci=np.quantile(bd,[.0125,.9875]).tolist();aci=np.quantile(ad,[.0125,.9875]).tolist()
        seed_b=[pooled[str(s)]['native_frozen']['Brier']-pooled[str(s)]['common_adapt']['Brier'] for s in SEEDS]
        seed_a=[pooled[str(s)]['native_frozen']['ROC_AUC']-pooled[str(s)]['common_adapt']['ROC_AUC'] for s in SEEDS]
        gates={'Brier_margin_001':bci[1]<-.001 and all(v<0 for v in seed_b),
            'AUC_positive':aci[0]>0 and all(v>0 for v in seed_a),
            'calibration':means['native_frozen']['ECE10']-means['common_adapt']['ECE10']<=.003}
        gates['candidate']=all(gates.values())
        result['primary']={'native_minus_common_Brier':float(np.mean(seed_b)),'per_seed_Brier':seed_b,
            'native_minus_common_AUC':float(np.mean(seed_a)),'per_seed_AUC':seed_a,
            'Brier_Bonferroni_CI975':bci,'AUC_Bonferroni_CI975':aci,
            'bootstrap_draws':3000,'bootstrap_seed':34034,'conditional_on_fitted_seeds':True,
            'gates':gates,'stop_native_KuaiRec_recipe':not gates['candidate']}
    return result


def final_stage(out):
    torch.set_num_threads(1);record=verify_frozen(out)
    if record['final_labels_opened'] or (out/'final-open-receipt.json').exists():raise ValueError('final access already consumed; replay only')
    # A durable receipt consumes access before networking/parsing. A source
    # failure cannot silently switch to another population or change the recipe.
    write_json(out/'final-open-receipt.json',{'source_commit':record['source_commit'],'protocol_sha256':record['protocol_sha256'],
        'frozen_manifest_sha256':file_sha(out/'frozen-payloads.json'),'user_remainder':6,
        'window':['2020-08-14','2020-09-06'],'maximum_windows':6000,'all_models_verified':True})
    start=time.process_time();wall=time.perf_counter()
    record['final_labels_opened']=True;write_json(out/'results.json',record)
    data,source,stats=read_final(out)
    record['final_evaluation']=evaluate(out,data,record,True)
    record['final_source']=source;record['final_parse_stats']=stats
    record['final_CPU_seconds']=time.process_time()-start;record['final_wall_seconds']=time.perf_counter()-wall
    record['complete']=True;write_json(out/'results.json',record)
    print(json.dumps(record['final_evaluation']['primary']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='results/me034');p.add_argument('--stage',choices=('fit','final'),required=True)
    a=p.parse_args();(fit if a.stage=='fit' else final_stage)(Path(a.out))
