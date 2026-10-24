"""ME094 prospectively registered sequential native capability search.

No architecture-winner or equal-compute claim. Five fixed-index weight-only fine-tunes, all original
models Git-frozen before exposed REPORT scoring. No sealed source access.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,random,subprocess,time,platform
from pathlib import Path
import numpy as np
import torch
from scripts.me076_train_system1_real import load_original as load_base,evaluation_origins,sha,write,digest

from scripts.validate_research_parity import check_manifest
from tac_osm.parameter_budget import budgeted_optimizer,checked_optimizer_step,save_training_checkpoint,validate_runtime_evidence
from tac_osm.variants.median_anchored_distribution import log_density,mixture_crps,implicit_median,mixture_quantiles
from tac_osm.variants.median_anchored_world import MedianAnchoredWorld

from torch.nn import functional as F



ROOT=Path('research/mathematical-engineering/me094');OUT=Path('research/replay-bundles/ME094');MANIFEST=ROOT/'FAIR_COMPARISON.json'
SEEDS=tuple(range(94001,94006));STEPS=4096;BATCH=24;HORIZONS=(1,4,16);COUNT=101173
CODE=['scripts/me094_train_world.py','scripts/me094_parallel_fit.py','scripts/clock_value_train.py','scripts/me091_visual_source.py','scripts/me091_train_world.py','src/tac_osm/gradient_surgery.py','src/tac_osm/variants/visual_world.py','src/tac_osm/variants/bounded_value_world.py','src/tac_osm/variants/implicit_quantile.py','src/tac_osm/variants/clock_value_world.py','src/tac_osm/variants/equilibrium_parallel_world.py','src/tac_osm/variants/streaming_mixture_world.py','src/tac_osm/variants/reusable_execution.py','src/tac_osm/variants/horizon_query_world.py',
      'src/tac_osm/variants/kinematic_world.py','scripts/me079_train_world.py',
      'scripts/me076_train_system1_real.py','scripts/me064_mixed_real_fit.py',
      'scripts/me057_train_fused_anchor_robot.py','src/tac_osm/parameter_budget.py',
      'scripts/validate_research_parity.py','scripts/me081_train_world.py','research/median-anchored-search/SEARCH.json','research/bounded-value-search/SEARCH.json','research/clock-value-search/SEARCH.json','scripts/equilibrium_parallel_train.py']


CONFIG={'me094':(.2,94001)}
STUDY='me094';NLL_WEIGHT=.2
CODE += ['src/tac_osm/variants/robust_visual_world.py','src/tac_osm/variants/multimode_distribution.py']
CODE += [f'research/replay-bundles/ME093/seed-{seed}/final.pt' for seed in range(93001,93006)]
CODE += ['research/mathematical-engineering/me093/CAPABILITY_RESULTS.json','src/tac_osm/variants/median_anchored_world.py','src/tac_osm/variants/median_anchored_distribution.py']

def configure(study):
    global STUDY,ROOT,OUT,MANIFEST,SEEDS,NLL_WEIGHT
    STUDY=study;NLL_WEIGHT,seed=CONFIG[study]
    ROOT=Path('research/mathematical-engineering')/study
    OUT=Path('research/replay-bundles')/study.upper()
    MANIFEST=ROOT/'FAIR_COMPARISON.json';SEEDS=tuple(range(seed,seed+5))


def load_initial(model,weights):
    missing,unexpected=model.load_state_dict(weights,strict=False)
    if set(missing)!={'decision_head.weight','decision_head.bias'} or unexpected:raise RuntimeError('Unexpected parent mapping')


def visual_parent(seed):
    return Path(f'research/replay-bundles/ME093/seed-{seed-1000}/final.pt')


def load_original(cache):
    from scripts.equilibrium_parallel_train import load_original as with_noise
    import pyarrow.parquet as pq
    reg,fit,report,identified,evidence=with_noise(cache)
    for name,rows in [('FIT',fit),('REPORT',report)]:
        hashes={}
        for (task,ep),row in rows.items():
            path=cache/task/f'data/chunk-{ep//1000:03d}'/f'episode_{ep:06d}.parquet'
            hashes[f'{task}:{ep}']=sha(path)
            table=pq.read_table(path,columns=['capture_time','frame_index'])
            clock=np.asarray(table.column('capture_time').to_numpy(),dtype=np.float64)
            if not np.array_equal(table.column('frame_index').to_numpy(),np.arange(len(clock))):raise RuntimeError('Video/parquet retained frame indices do not align')
            elapsed=np.r_[1/15,np.diff(clock)]
            if len(elapsed)!=len(row['state']) or not np.isfinite(elapsed).all() or (elapsed<=0).any():
                raise RuntimeError('Capture clock must be finite strictly increasing')
            row['elapsed']=torch.from_numpy(elapsed.astype(np.float32))[:,None]
        if digest(hashes)!=evidence[name+'_rows_sha256']:raise RuntimeError('Clock source bytes differ')
    from scripts.me091_visual_source import acquire
    visual_cache=Path('/tmp/me091-visual-source')
    if not (visual_cache/'SOURCE_INDEX.json').exists():acquire(visual_cache)
    index=json.loads((visual_cache/'SOURCE_INDEX.json').read_text())
    source_records={}
    for role,rows in [('FIT',fit),('REPORT',report)]:
        for (task,ep),row in rows.items():
            record=next(r for r in index['records'] if r['task']==task and r['episode']==ep)
            path=visual_cache/task/f'{ep:06d}.npz'
            if record['role']!=role or sha(path)!=record['cache_sha256']:raise RuntimeError('Visual source role/cache differs')
            with np.load(path) as data:images=data['images']
            if len(images)!=len(row['state']) or hashlib.sha256(images.tobytes()).hexdigest()!=record['pixel_sha256']:raise RuntimeError('Visual frame alignment/pixels differ')
            row['images']=torch.from_numpy(images)
            source_records[f'{task}:{ep}']={k:record[k] for k in ['role','frames','source_path','source_revision','source_sha256','pixel_sha256','pixels','alignment']}
    evidence['visual_source']={'camera':index['camera'],'records':source_records,'canonical_sha256':digest(source_records),'license':index['license'],'credit':index['credit'],'source_bytes':index['source_bytes']}
    if MANIFEST.exists():
        frozen=json.loads(MANIFEST.read_text())['visual_source']
        if evidence['visual_source']!=frozen:raise RuntimeError('Frozen visual source contract differs')
    return reg,fit,report,identified,evidence


def schedule(rows,identified,seed):
    from scripts.me081_train_world import pools
    pp=pools(rows,identified);tasks=sorted({k[0] for k in pp});allrows={t:[] for t in tasks};moving={t:[] for t in tasks};moveflags=set()
    for (task,ep),(times,flags) in pp.items():
        for t,flag in zip(times,flags):
            entry=(task,ep,int(t));allrows[task].append(entry)
            if flag:moving[task].append(entry);moveflags.add(entry)
    rng=np.random.default_rng(seed);orders=[];weights=[]
    for step in range(STEPS):
        entries=[];ww=[]
        for b in range(BATCH):
            task=tasks[(step*BATCH+b)%len(tasks)];has_motion=bool(moving[task]);selected=moving[task] if b%2==0 and has_motion else allrows[task]
            entry=selected[int(rng.integers(len(selected)))];entries.append(entry);rate=len(moving[task])/len(allrows[task])
            ww.append(1/(.5+.5/rate) if entry in moveflags and has_motion else 2. if has_motion else 1.)
        orders.append(entries);weights.append(ww)
    if len(orders)!=STEPS:raise RuntimeError('Actual schedule length mismatch')
    return orders,weights,{'sample_order_sha256':digest(orders),'importance_sha256':digest(weights),
       'unique_FIT_eligible_origins':sum(len(v) for v in allrows.values()),'task_motion_prevalence':{t:len(moving[t])/len(allrows[t]) for t in tasks}}


def batch(rows,entries,history=32):
    from scripts.equilibrium_parallel_train import batch as base_batch
    x,a,y,valid=base_batch(rows,entries,history)
    elapsed=torch.stack([rows[(task,ep)]['elapsed'][t-history+1:t+1] for task,ep,t in entries])
    images=torch.stack([rows[(task,ep)]['images'][t-history+1:t+1] for task,ep,t in entries]).float()/255
    return x,a,y,valid,elapsed,images


def long_queries(rows,queries):
    return [(task,ep,t) for task,ep,t in queries if t>=32 and not bool(rows[(task,ep)]['human'][t-32:t+1].any())]


def task_losses(prediction,targets,current,valid,importance,physical_scale,normalizer):
    h=targets.new_tensor((1,4,16))[None,None,:,None]
    crps=(mixture_crps(prediction,targets)*physical_scale/(normalizer*h)).mean(-1)
    nll=(-log_density(prediction,targets)+physical_scale.log()).mean(-1)
    factual=(((targets-current[:,:,None]).abs()*physical_scale).mean(-1)>=.025).float()
    bce=F.binary_cross_entropy_with_logits(prediction['motion_logit'],factual,reduction='none')
    point=((implicit_median(prediction)-targets.double()).abs()*physical_scale/(normalizer*h)).mean(-1)
    score=crps+NLL_WEIGHT*nll+.2*bce+20*point
    first=torch.zeros_like(valid);first[:,:,0]=True
    moving=(factual[:,:,0]>0)[:,:,None]
    masks=(first&~moving,first&moving,~first)
    return tuple((importance*(score*valid*mask).sum((1,2))/valid.sum((1,2)).clamp_min(1)).mean() for mask in masks)


def integrity():
    reg=json.loads(MANIFEST.read_text());check_manifest(reg,'preflight')
    for p,d in reg['implementation_sha256'].items():
        if sha(p)!=d:raise RuntimeError('Frozen source changed: '+p)
    return reg


def qualify(cache):
    if MANIFEST.exists():raise RuntimeError('Never replace a frozen registration')
    _,fit,report,identified,evidence=load_original(cache)
    queries,query_evidence=evaluation_origins(report,identified)
    m=MedianAnchoredWorld(identified);n=sum(p.numel() for p in m.parameters())
    if n!=COUNT:raise RuntimeError('Actual parameter count differs')
    schedules={str(seed):schedule(fit,identified,seed)[2] for seed in SEEDS}
    origins={}
    for seed in SEEDS:
        path=visual_parent(seed)
        if hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+str(path)])).hexdigest()!=sha(path):raise RuntimeError('Parent not Git frozen')
        original=torch.load(path,map_location='cpu',weights_only=True)
        if original['updates']!=3072 or original['FIT_only_identified']!={k:v.tolist() for k,v in identified.items()}:raise RuntimeError('Parent provenance differs')
        load_initial(m,original['weights'])
        origins[str(seed)]={'path':str(path),'sha256':sha(path),'source_updates':3072,'new_updates':STEPS,'pipeline_updates':17408,'retained_values':96595,'new_decision_values':4578,'native_visual_lineage_updates':13312,'location_lineage_updates':9216,'old_local_noise_lineage_updates':7168,'new_values_updates':4096,'optimizer_reset':True,'all_parameters_trainable':True,'mapping':'ALL5 fixed indexed ME093 parents, entire96595values retained exactly;4578newdecision values zero weights with gatebias-2,timebias7.75,retrievalbias0. No parent/seed/checkpoint selection.'}
    reg=json.loads(Path('research/mathematical-engineering/me093/FAIR_COMPARISON.json').read_text())
    reg.update(study_id='ME094_MEDIAN_ANCHORED_NATIVE_WORLD',registered_before_first_fit=True,initialization=origins,implementation_sha256={p:sha(p) for p in CODE},models=[{'architecture':m.architecture,'seed':s,'trainable_parameters':n,'deployed_parameters':n} for s in SEEDS],schedules=schedules,visual_source=evidence['visual_source'])
    reg['recipe']={'nll_weight':.2,'study':'me094','registered_search_family':'ME094_EXPLICIT_MEDIAN_AND_CAUSAL_RELEVANCE_RETRIEVAL'}
    reg['training_budget']={'updates':STEPS,'batch_size':BATCH,'history':32,'dense_prefixes':32,'horizons':list(HORIZONS),'seeds':list(SEEDS),'optimizer':'AdamW retained lr0.0001/newdecision lr0.0003,wd0.0001,clip1,rotating PCGrad','selection':'Final4096new only, ALL5fixed ME093 parents. No restarts/bestcheckpoint/bestseed/extra fits under this registration; all originals Git frozen before REPORT','sampling':'same original task-balanced origin proposal and importance correction; fixed schedules','loss':'Proper CRPS+0.2physical NLL+20explicit exact-median MAE+0.2factual global motion BCE. Same quietH1/movingH1/long partition and deterministic PCGrad. No component class labels or extra quiet-only penalty.','compute':'4096new updates/98304windows/3145728numeric+image prefixes each.17408charged pipeline updates, native/visual13312ownlineage,location9216,oldlocalnoise7168,newdecision4096. No matched model winner claim.'}
    reg['unresolved_capability']='Can explicit median learning, coherent multimodal shape recentering and relevance-weighted causal value retrieval jointly meet quiet/movement/uncertainty gates?'
    reg['evaluation']['decision']='Every unchangedME081gate plus existingME081gain/NLLconditions, original95%and additional99.61538%episodeintervalupper<0, strict original replay. No relaxation; exposedDEVonly. One recipe only; failure exhausts this registration.'
    reg['evaluation']['statistics']='5000paired episode bootstrap seed80080; original95%plus99.61538%interval for13declaredME082..ME094recipes, conditionalfittedmodels/adaptiveexposedDEV; not independent proof.'
    reg['evaluation']['runtime_scope']='same primary8/secondary32 factual forecasts, explicit1331scalar carried state. Optimized final-only decode must match full-prefix outputs; no arbitrary-history accuracy claim.'
    reg['adaptation_ledger']={'ME082-ME093':'60original model versions, every full gateFAIL, all preserved','ME094':'ONE new integrated median/retrieval architecture, same source and forecast cutoffs'}
    reg['capability_to_value']['skill']='Joint quiet/movement physical forecasting with an explicit predictive median and coherent multimodal uncertainty';reg['capability_to_value']['external_pilot']='NOT_MEASURED'
    reg['literature'].append({'paper':'Quantile Regression, Koenker and Bassett1978','url':'https://doi.org/10.2307/1913643','reuse':'Absolute loss targets a conditional median. Translating a proper distribution by an independently learned median minus its implicit base median preserves normalized multimodal shape; no new quantile theorem claimed.'})
    check_manifest(reg,'preflight');write(MANIFEST,reg)
    write(ROOT/'SOURCE_QUALIFICATION.json',{'source':evidence,'queries':query_evidence,'schedules':schedules,'secondary32_queries_sha256':digest(long_queries(report,queries)),'secondary32_queries':len(long_queries(report,queries)),'actual_trainable_parameters':n,'actual_deployed_parameters':n,'horizon_available_queries':{str(h):sum(t+h<len(report[(task,ep)]['state']) and not bool(report[(task,ep)]['human'][t:t+h+1].any()) for task,ep,t in queries) for h in HORIZONS}})
    print('ME094_PREFIT_PASS',n,flush=True)


def fit_all(cache,only_seed=None):
    registration=integrity()
    selected=SEEDS if only_seed is None else (only_seed,)
    if any(seed not in SEEDS for seed in selected):raise RuntimeError('Undeclared seed')
    if any((OUT/f'seed-{seed}/final.pt').exists() for seed in selected):raise RuntimeError('Never overwrite original fits')
    _,fit,_,identified,_=load_original(cache);index=[]
    normalizer=float(np.mean(identified['delta_scale']*identified['state_scale']))
    for seed in selected:
        torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
        m=MedianAnchoredWorld(identified);folder=OUT/f'seed-{seed}'
        origin=registration['initialization'][str(seed)]
        if sha(origin['path'])!=origin['sha256']:raise RuntimeError('Warm-start original changed')
        original=torch.load(origin['path'],map_location='cpu',weights_only=True)
        if original['updates']!=3072 or original['FIT_only_identified']!={k:v.tolist() for k,v in identified.items()}:raise RuntimeError('Unregistered parent/statistics')
        load_initial(m,original['weights'])
        orders,weights,stats=schedule(fit,identified,seed)
        if stats!=json.loads(MANIFEST.read_text())['schedules'][str(seed)]:raise RuntimeError('FIT schedule changed')
        opt=budgeted_optimizer(m,torch.optim.AdamW,[{'params':[p for name,p in m.named_parameters() if not name.startswith('decision_head.')],'lr':.0001},{'params':list(m.decision_head.parameters()),'lr':.0003}],lr=.0001,weight_decay=.0001,budget_manifest=MANIFEST,budget_architecture=m.architecture,budget_seed=seed,budget_receipt=folder/'live-budget.json')
        trace=[];start=time.process_time();valid_count=0
        for step,(picks,ww) in enumerate(zip(orders,weights),1):
            x,a,y,valid,elapsed,images=batch(fit,picks);valid_count+=int(valid.sum())
            pred=m(x,a,elapsed=elapsed,images=images);losses=task_losses(pred,y,x,valid,torch.tensor(ww),m.state_scale,normalizer);loss=sum(losses)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite authentic loss')
            from tac_osm.gradient_surgery import projected_task_gradients
            opt.zero_grad(set_to_none=True);surgery=projected_task_gradients(losses,m.parameters(),rotation=step-1);norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
            if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradient')
            checked_optimizer_step(opt)
            if step==1 or step%256==0:
                trace.append({'step':step,'FIT_loss':float(loss.detach()),**surgery,'task_losses':[float(v.detach()) for v in losses]});print(json.dumps({'seed':seed,**trace[-1]}),flush=True)
        checkpoint=folder/'final.pt'
        save_training_checkpoint({'weights':m.state_dict(),'optimizer':opt.state_dict(),'seed':seed,'architecture':m.architecture,
           'updates':STEPS,'training_lineage':origin,'pipeline_optimizer_updates':17408,'retained_native_updates':13312,'visual_weight_updates':13312,'location_weight_updates':9216,'local_noise_weight_updates':7168,'decision_weight_updates':4096,'torch_rng':torch.get_rng_state(),'numpy_rng':(np.random.get_state()[0],np.random.get_state()[1].tolist(),*np.random.get_state()[2:]),
           'python_rng':random.getstate(),'FIT_only_identified':{k:v.tolist() for k,v in identified.items()},'schedule':stats,
           'valid_supervised_targets':valid_count,'prefit_registration_sha256':sha(MANIFEST)},checkpoint)
        rec={'seed':seed,'parameters':sum(p.numel() for p in m.parameters()),'deployed_parameters':sum(p.numel() for p in m.parameters()),
             'actual_steps':STEPS,'training_lineage':origin,'pipeline_optimizer_updates':17408,'retained_native_updates':13312,'visual_weight_updates':13312,'location_weight_updates':9216,'local_noise_weight_updates':7168,'decision_weight_updates':4096,'actual_examples':STEPS*BATCH,'actual_observed_prefixes':STEPS*BATCH*32,
             'valid_dense_target_presentations':valid_count,'training_cpu_seconds':time.process_time()-start,'sampling':stats,
             'checkpoint_sha256':sha(checkpoint),'trace':trace,'environment':{'torch':torch.__version__,'python':platform.python_version(),'threads':torch.get_num_threads()}}
        write(folder/'FIT_RECEIPT.json',rec);index.append(rec)
        if only_seed is None:write(OUT/'FIT_INDEX.json',index)
    print('ALL5_ORIGINALS_COMPLETE_NO_REPORT_SCORED' if only_seed is None else f'ORIGINAL_SEED_COMPLETE_{only_seed}',flush=True)


def frozen():
    for seed in SEEDS:
        for name in ('final.pt','live-budget.json','FIT_RECEIPT.json'):
            p=OUT/f'seed-{seed}'/name
            if hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+str(p)])).hexdigest()!=sha(p):raise RuntimeError('All original model bytes must be Git-frozen before scoring')


def finalize():
    reg=integrity();frozen();final=copy.deepcopy(reg);final['prefit_registration_sha256']=sha(MANIFEST)
    for row in final['models']:
        folder=OUT/f"seed-{row['seed']}";receipt=json.loads((folder/'FIT_RECEIPT.json').read_text())
        checkpoint=torch.load(folder/'final.pt',weights_only=True,map_location='cpu')
        m=MedianAnchoredWorld(checkpoint['FIT_only_identified']);m.load_state_dict(checkpoint['weights'],strict=True)
        measured={'trainable_parameters':sum(p.numel() for p in m.parameters() if p.requires_grad),
                  'deployed_parameters':sum(p.numel() for p in m.parameters()),'checkpoint_sha256':sha(folder/'final.pt')}
        runtime={'receipt_path':str(folder/'live-budget.json'),'receipt_sha256':sha(folder/'live-budget.json'),
                 'checkpoint_path':str(folder/'final.pt'),'checkpoint_sha256':sha(folder/'final.pt')}
        validate_runtime_evidence({**row, **measured, 'runtime_evidence':runtime, '_prefit_registration_sha256':sha(MANIFEST), '_expected_optimizer_steps':STEPS})
        row['measured']={**measured,'actual_steps':STEPS,'training_receipt':receipt,'runtime_evidence':runtime}
    status=check_manifest(final,'final');write(ROOT/'FINAL_COMPARISON.json',final);print(status,flush=True)


def score_one(m,rows,identified,queries,history=8):
    physical=torch.as_tensor(identified['state_scale'],dtype=torch.float64);results=[];m.eval()
    with torch.no_grad():
        for i in range(0,len(queries),24):
            picks=queries[i:i+24];x,a,y,valid,elapsed,images=batch(rows,picks,history);raw=m.forecast_last(x.double(),a.double(),elapsed=elapsed.double(),images=images.double())
            raw={k:v.double() for k,v in raw.items()}
            qq=mixture_quantiles(raw);pred={'lower':qq[0],'median':qq[1],'upper':qq[2]}
            target=y[:,-1].double();current=x[:,-1].double();available=valid[:,-1]
            density=log_density(raw,target)-physical.log()
            velocity=(x[:,-1]-x[:,-2]).double()
            command=(a[:,-1]*m.action_scale+m.action_mean-m.state_mean)/m.state_scale-x[:,-1]
            servo=torch.clamp(m.servo_gain*command,-m.servo_step_limit,m.servo_step_limit).double()
            for j,(task,ep,t) in enumerate(picks):
                for hi,h in enumerate(HORIZONS):
                    if not available[j,hi]:continue
                    truth=target[j,hi];center=pred['median'][j,hi];lower=pred['lower'][j,hi];upper=pred['upper'][j,hi]
                    movement=float(((truth-current[j]).abs()*physical).mean());event=movement>=.025
                    q=raw['motion_logit'][j,hi].sigmoid()
                    refs={'persistence':current[j],'velocity':current[j]+h*torch.clamp(velocity[j],-m.servo_step_limit.double(),m.servo_step_limit.double()),'servo':current[j]+h*servo[j]}
                    r={'task':task,'episode':ep,'frame':t,'horizon':h,'moving':event,'physical_movement':movement,
                       'mae':float(((center-truth).abs()*physical).mean()),'motion_probability':float(q),'motion_brier':float((q-float(event)).square()),
                       'coverage':float(((lower<=truth)&(truth<=upper)).double().mean()),'width':float(((upper-lower)*physical).mean()),
                       'interval_score':float(((upper-lower+20*(lower-truth).clamp_min(0)+20*(truth-upper).clamp_min(0))*physical).mean()),
                       'physical_marginal_nll':float(-density[j,hi].mean()),
                       'prediction':center.tolist(),'target':truth.tolist(),'lower':lower.tolist(),'upper':upper.tolist()}
                    r.update({name+'_mae':float(((value-truth).abs()*physical).mean()) for name,value in refs.items()});results.append(r)
    return results


def run_scores(cache,replay=False):
    integrity();frozen();_,_,report,identified,evidence=load_original(cache);queries,_=evaluation_origins(report,identified);index=[]
    for seed in SEEDS:
        folder=OUT/f'seed-{seed}';checkpoint=folder/'final.pt';fit=json.loads((folder/'FIT_RECEIPT.json').read_text())
        if sha(checkpoint)!=fit['checkpoint_sha256']:raise RuntimeError('Checkpoint changed')
        m=MedianAnchoredWorld(identified);m.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['weights'],strict=True);m.double()
        for history in (8,32):
            selected=queries if history==8 else long_queries(report,queries)
            rows=score_one(m,report,identified,selected,history)
            path=folder/('ORIGINAL_PREDICTIONS.json' if history==8 else 'ORIGINAL_PREDICTIONS_H32.json')
            if replay:
                oldrows=json.loads(path.read_text())
                if len(oldrows)!=len(rows):raise RuntimeError('Denominator changed')
                for old,new in zip(oldrows,rows):
                    for key in old:
                        if isinstance(old[key],(float,list)):
                            if not np.allclose(old[key],new[key],rtol=1e-5,atol=1e-6):raise RuntimeError(f'Strict replay failed {seed}/HISTORY{history}/{key}')
                        elif old[key]!=new[key]:raise RuntimeError('Identity changed')
            else:
                if path.exists():raise RuntimeError('Never overwrite original scores')
                write(path,rows)
            index.append({'seed':seed,'observed_history':history,'checkpoint_sha256':sha(checkpoint),'predictions_sha256':sha(path),'forecasts':len(rows)})
    write(OUT/('INDEPENDENT_REPLAY.json' if replay else 'ORIGINAL_SCORE_INDEX.json'),
          {'status':'PASS','original_models':5,'original_forecasts':sum(r['forecasts'] for r in index),'models':index,
           'no_refit':True,'rtol':1e-5,'atol':1e-6,'source_sha256':evidence['original_160_episode_source_sha256']})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['qualify','fit','finalize','score','replay']);parser.add_argument('--seed',type=int);parser.add_argument('--study',choices=tuple(CONFIG),default='me094');parser.add_argument('--cache',type=Path,default=Path('/tmp/me077-original-source'));args=parser.parse_args();torch.set_num_threads(1);configure(args.study)
    if args.mode=='qualify':qualify(args.cache)
    elif args.mode=='fit':fit_all(args.cache,args.seed)
    elif args.mode=='finalize':finalize()
    else:run_scores(args.cache,args.mode=='replay')
