"""Original real-data native trajectory training, frozen scoring and replay."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
import torch
from scripts.me069_workflow_source import digest,write_json
from scripts.me071_persistent_workflow import schedules,verify_blob
from scripts.me072_marked_world import source as inherited_source,compare_scores
from scripts.validate_research_parity import check_manifest
from tac_osm.workflow_me069 import EventBelief
from tac_osm.workflow_me071 import detach_belief
from tac_osm.workflow_me073 import (TrajectoryModel,STEPS,HORIZONS,PARTICLES,teacher_path,generate)

ROOT=Path('research/mathematical-engineering/me073')
BUNDLE=Path('research/replay-bundles/ME073')
SEEDS=(73001,73002,73003,73004,73005)
EPOCHS,BATCH,CHUNK=6,32,32
AGES=(2,6,16,32,64)


from tac_osm.parameter_budget import budgeted_optimizer, checked_optimizer_step, save_training_checkpoint

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def derive_preprocessing(cases,pre):
    pre=copy.deepcopy(pre);n=len(pre['activities'])+1
    total=np.zeros(n);survive=np.zeros(n);pairs={};by_current={}
    for c in cases:
        for i,a in enumerate(c['activity']):
            a=int(a);total[a]+=1
            if i+1<c['n']:
                b=int(c['activity'][i+1]);dt=float(c['gap'][i+1]);survive[a]+=1
                pairs.setdefault((a,b),[]).append(dt);by_current.setdefault(a,[]).append(dt)
    global_gaps=[g for row in by_current.values() for g in row]
    locations=np.zeros((n,n,3));scale=np.zeros_like(locations);zero=np.zeros((n,n))
    for a in range(n):
        for b in range(n):
            row=pairs.get((a,b),[])
            # Avoid unstable small-pair estimates; back off using FIT only.
            if len(row)<16:row=by_current.get(a,global_gaps)
            positive=np.log([g for g in row if g>0])
            if not len(positive):positive=np.log([g for g in global_gaps if g>0])
            locations[a,b]=np.quantile(positive,[.15,.5,.85])
            std=float(np.clip(np.std(positive)/2.,.2,1.1))
            ratio=(std-.15)/1.1;scale[a,b]=math.log(ratio/(1-ratio))
            p0=(sum(g==0 for g in row)+.5)/(len(row)+1);zero[a,b]=math.log(p0/(1-p0))
    total[0]=total[1:].sum();survive[0]=survive[1:].sum()
    s=(survive+.5)/(total+1)
    pre.update(step_survival_logit=np.log(s/(1-s)).tolist(),pair_clock_locations=locations.tolist(),
               pair_clock_scale_logit=scale.tolist(),pair_clock_zero_logit=zero.tolist())
    return pre


def source(cache,frozen=True):
    cohorts,pre,receipt=inherited_source(cache)
    if frozen:
        plan=json.loads((ROOT/'FAIR_COMPARISON.json').read_text());original=json.loads((ROOT/'FIT_PREPROCESSING.json').read_text())
        if digest(original)!=plan['preprocessing_sha256']:raise ValueError('changed frozen preprocessing')
        if receipt['source_sha256']!=plan['real_world_source']['sha256']:raise ValueError('changed original source')
        for key in ('activities','resources'):
            if pre[key]!=original[key]:raise ValueError('source vocabulary changed')
        pre=original
    else:pre=derive_preprocessing(cohorts['FIT'],pre)
    return cohorts,pre,receipt


def pack(cases,indices,start=0,length=CHUNK):
    b=len(indices);n=cases[0]['history_log_prior'].shape[-1]
    activity=np.zeros((b,length),np.int64);resources=np.zeros_like(activity)
    features=np.zeros((b,length,8),np.float32);hours=np.zeros((b,length,1),np.float32)
    history=np.zeros((b,length,n),np.float32);target=np.zeros((b,length,STEPS),np.int64)
    seconds=np.zeros((b,length,STEPS),np.float32);present=np.zeros((b,length,STEPS),bool);valid=np.zeros((b,length),bool)
    first_a=np.zeros(b,np.int64);first_r=first_a.copy();first_f=np.zeros((b,8),np.float32)
    for row,index in enumerate(indices):
        c=cases[index];first_a[row]=c['activity'][0];first_r[row]=c['resource'][0];first_f[row]=c['features'][0]
        end=min(start+length,c['n']);size=max(0,end-start)
        if not size:continue
        sl=slice(start,end);activity[row,:size]=c['activity'][sl];resources[row,:size]=c['resource'][sl]
        features[row,:size]=c['features'][sl];hours[row,:size,0]=c['gap'][sl]/3600.
        history[row,:size]=c['history_log_prior'][sl];valid[row,:size]=True
        for j in range(STEPS):
            future=np.arange(start,end)+j+1;exists=future<c['n'];safe=np.minimum(future,c['n']-1)
            target[row,:size,j]=np.where(exists,c['activity'][safe],0)
            # Incremental gaps for exact autoregressive suffix likelihood.
            seconds[row,:size,j]=np.where(exists,c['gap'][safe],0)
            present[row,:size,j]=exists
    return tuple(torch.from_numpy(x) for x in (activity,resources,features,hours,history,target,seconds,present,valid,first_a,first_r,first_f))


def observed(model,data,belief=None,all_steps=True):
    encoded=model.event_encoding(*data[:3])
    if belief is None:belief=model.initial_from_context(model.event_encoding(*data[9:12]))
    states=[]
    for j in range(encoded.shape[1]):
        belief=model.observe(belief,encoded[:,j],data[3][:,j],data[8][:,j])
        if all_steps:states.append(belief)
    if all_steps:
        merged=EventBelief(*(torch.stack([s[i] for s in states],1).flatten(0,1) for i in range(3)))
        args=(merged,encoded.flatten(0,1),data[0].flatten(),data[4].flatten(0,1))
    else:args=(belief,encoded[:,-1],data[0][:,-1],data[4][:,-1])
    return args,belief


def target_counts(cases):
    return sum(min(STEPS,c['n']-t) for c in cases for t in range(c['n']))*EPOCHS


def register(cache):
    torch.set_num_threads(1);cohorts,pre,receipt=source(cache,False)
    model=TrajectoryModel(pre);count=sum(x.numel() for x in model.parameters());targets=target_counts(cohorts['FIT'])
    plan={'schema_version':'tac-osm.parity.v1','study_id':'ME073_AUTOREGRESSIVE_NATIVE_MARKED_TRAJECTORIES',
          'study_type':'capability_development','registered_before_first_fit':True,'architecture_count':1,
          'no_architecture_superiority_claim':True,
          'scientific_purpose':'Train one integrated native autoregressive marked trajectory model, activity-conditioned stable incremental clocks and lawful first-event context initialization. No comparator or causal-policy claim.',
          'real_world_source':{'name':'original publisher-verified BPI Challenge2012 loan process log','sha256':receipt['source_sha256'],
                               'license':'4TU General Terms of Use; raw source external, commercial reuse not qualified','observed_real_outcomes':True},
          'split_sha256':{k:digest([c['id'] for c in cohorts[k]]) for k in ('FIT','TEMP','CQR','REPORT')},
          'preprocessing_sha256':digest(pre),'trainable_parameters':count,'seeds':list(SEEDS),
          'training_budget':{'epochs':EPOCHS,'batch_size':BATCH,'backprop_events':CHUNK,'expected_targets_per_seed':targets,
                             'observed_prefix_exposures':sum(c['n'] for c in cohorts['FIT'])*EPOCHS,
                             'optimizer':'AdamW lr.001 wd.0001 clip1','selection':'all five final six-epoch originals; no DEV tuning/selection or calibration'},
          'schedule_sha256':{str(s):digest(schedules(cohorts['FIT'],s)) for s in SEEDS},
          'repository_storage_plan':'Five original model/optimizer/RNG checkpoints in research/replay-bundles/ME073/checkpoints (expected <15MB), actual schedules/source/frozen preprocessing, original QMC forecast distributions, representative generated paths, factual suffix scores, independent GitHub clean-checkout replay; freeze weights before REPORT. Recoverable failed states retained, never overwritten with retraining.',
          'horizons':list(HORIZONS),'query_ages':list(AGES),'rollout_steps':STEPS,'particles':PARTICLES,
          'development_support_before_fit':{str(a):sum(c['n']>=a for c in cohorts['REPORT']) for a in AGES},
          'input_availability':'Original observed events/resource/time features; context only first observed event/resource/calendar. Untimed REG_DATE/AMOUNT_REQ and case IDs excluded. Original TEST no scoring; published file parsed for inherited chronological split.',
          'roles':'FIT only training/statistics; TEMP/CQR unused; previously exposed last925 DEV REPORT. No new TEST or external qualification.',
          'loss':'Exact teacher-forced 8-step suffix likelihood components: survival+mark+.2 zero/log-time density+.05 mark Brier; stop after first EOR. Weight observed prefix ages1..6 by2, others1; no heldout-derived weights.',
          'readout':'Free-run 256 deterministic scrambled Sobol trajectories, shared native step updates and conditional marks/clocks. Rao-Blackwell survival weights. h1 activity exact; h2/4/8 finite-particle normalized smoothed categorical probabilities (epsilon1e-6). Weighted .05/.50/.95 elapsed quantiles. Exact factual suffix density likelihood reported separately; future labels never enter generation.',
          'numerical_contract':{'primary_atol':1e-5,'primary_rtol':1e-5,'metric_atol':1e-4,'metric_rtol':1e-5},
          'claim_limits':'Different surviving query-age populations; source EOR proxy only. Conditional-survival quantiles may reverse across horizons due population selection, while every path clock is nondecreasing. No process-legality, memory superiority, causal action, external operator value or commercial claim.',
          'prior_art':['Tax et al., Predictive Business Process Monitoring with LSTM Neural Networks,CAiSE2017,https://arxiv.org/abs/1612.02130',
                       'Shchur et al., Intensity-Free Learning of Temporal Point Processes,ICLR2020,https://arxiv.org/abs/1909.12127',
                       'Bengio et al., Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks,NeurIPS2015,https://arxiv.org/abs/1506.03099 (exposure bias identified; no scheduled-sampling claim)',
                       'Dawid, The Well-Calibrated Bayesian,JASA1982,https://doi.org/10.1080/01621459.1982.10477856 (coherence does not establish empirical calibration)']}
    write_json(ROOT/'FAIR_COMPARISON.json',plan);write_json(ROOT/'FIT_PREPROCESSING.json',pre);write_json(ROOT/'SOURCE.json',receipt)
    print(check_manifest(plan));print(json.dumps({'parameters':count,'targets':targets,'support':plan['development_support_before_fit']}))


def train(cache):
    torch.set_num_threads(1);plan=json.loads((ROOT/'FAIR_COMPARISON.json').read_text());check_manifest(plan);verify_blob(ROOT/'FAIR_COMPARISON.json')
    cohorts,pre,receipt=source(cache);folder=BUNDLE/'checkpoints';folder.mkdir(parents=True,exist_ok=True)
    if list(folder.glob('*.pt')):raise ValueError('original fit files exist; never overwrite')
    records=[]
    for seed in SEEDS:
        torch.manual_seed(seed);model=TrajectoryModel(pre);opt=budgeted_optimizer(model, torch.optim.AdamW, model.parameters(),lr=.001,weight_decay=.0001)
        order=schedules(cohorts['FIT'],seed)
        if digest(order)!=plan['schedule_sha256'][str(seed)]:raise ValueError('changed registered order')
        steps=prefixes=targets=0;trace=[];begin=time.process_time();path=folder/f'native-s{seed}.pt'
        def save(status):
            save_training_checkpoint({'model':model.state_dict(),'optimizer':opt.state_dict(),'torch_rng_state':torch.get_rng_state(),
                        'preprocessing':pre,'seed':seed,'epoch':epoch,'trace':trace,'schedule_sha256':digest(order),
                        'source_sha256':receipt['source_sha256'],'fit_status':status,'updates':steps,'actual_targets':targets},path)
        try:
            for epoch,batches in enumerate(order,1):
                losses=[]
                for indices in batches:
                    belief=None;longest=max(cohorts['FIT'][i]['n'] for i in indices)
                    for start in range(0,longest,CHUNK):
                        data=pack(cohorts['FIT'],indices,start);opt.zero_grad(set_to_none=True)
                        args,belief=observed(model,data,belief)
                        valid=data[8].flatten()
                        # Padded observations carry state but do not need imagined paths.
                        active_args=(EventBelief(*(x[valid] for x in args[0])),*(x[valid] for x in args[1:]))
                        early=1.+(data[2][...,6].flatten()[valid]<=math.log1p(5)/5.+1e-7).float()
                        loss,_,mask,_,_=teacher_path(model,*active_args,data[5].flatten(0,1)[valid],data[6].flatten(0,1)[valid],data[7].flatten(0,1)[valid],valid[valid],early)
                        if not torch.isfinite(loss):raise ValueError('nonfinite original real-data loss')
                        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                        if not torch.isfinite(norm):raise ValueError('nonfinite gradient')
                        checked_optimizer_step(opt);belief=detach_belief(belief);steps+=1;prefixes+=int(data[8].sum());targets+=int(mask.sum());losses.append(float(loss.detach()))
                trace.append({'epoch':epoch,'loss':float(np.mean(losses)),'prefix_exposures':prefixes,'targets':targets,'updates':steps})
                save('epoch_complete');print(json.dumps({'seed':seed,**trace[-1]}),flush=True)
        except BaseException as error:
            save('FAILED_RECOVERABLE');write_json(BUNDLE/'FAILED_FIT.json',{'seed':seed,'error':repr(error),'recoverable_path':str(path),'trace':trace});raise
        records.append({'seed':seed,'path':str(path),'checkpoint_sha256':sha(path),'epochs':EPOCHS,'actual_targets':targets,
                        'actual_prefix_exposures':prefixes,'actual_updates':steps,'optimizer_cpu_seconds':time.process_time()-begin})
        write_json(BUNDLE/'CHECKPOINTS.json',records)
    plan['completed_model_records']=records;print(check_manifest(plan,'final'),flush=True);write_json(BUNDLE/'FINAL_REGISTRATION.json',plan)
    write_json(BUNDLE/'ENVIRONMENT.json',{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
        'platform':platform.platform(),'CPU_threads':1,'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'training_dtype':'float32','replay_dtype':'float64; original trained weights promoted without refitting'})


def cast_data(data,dtype):return tuple(x.to(dtype) if x.is_floating_point() else x for x in data)


def metrics(forecasts,paths,data,joint,path_masks):
    n=data[4].shape[-1];rows={};cumulative=data[6][:,-1].cumsum(-1)
    for k,h in enumerate(HORIZONS):
        yes=data[7][:,-1,h-1];target=data[5][:,-1,h-1];t=cumulative[:,h-1];count=int(yes.sum())
        prob=forecasts[:,k,:n+1];conditional=prob[:,:n]/prob[:,:n].sum(-1,keepdim=True)
        labels=torch.where(yes,target,torch.full_like(target,n));low,median,high=forecasts[:,k,n+1:n+4].unbind(-1);width=high-low
        raw={'reached_query_cases':len(t),'future_record_cases':count,'record_boundary_cases':int((~yes).sum()),
             'QMC_activity_or_boundary_NLL':float(-prob.gather(-1,labels[:,None]).log().mean()),
             'QMC_joint_multiclass_Brier':float((prob.square().sum(-1)-2*prob.gather(-1,labels[:,None]).squeeze(-1)+1).mean()),
             'record_boundary_Brier':float((prob[:,-1]-(~yes).to(prob.dtype)).square().mean()),
             'conditional_activity_accuracy':float((conditional.argmax(-1)[yes]==target[yes]).double().mean()) if count else None,
             'QMC_conditional_activity_NLL':float(-conditional.gather(-1,target[:,None]).squeeze(-1)[yes].log().mean()) if count else None,
             'waiting_median_MAE_hours':float((median[yes]-t[yes]).abs().mean()/3600.) if count else None,
             'conditional_time_90_coverage':float(((low<=t)&(t<=high))[yes].double().mean()) if count else None,
             'mean_interval_width_hours':float(width[yes].mean()/3600.) if count else None,
             'mean_interval_score_hours':float((width+20*(low-t).clamp_min(0)+20*(t-high).clamp_min(0))[yes].mean()/3600.) if count else None,
             'width_p50_p90_p99_hours':(torch.quantile(width[yes],width.new_tensor([.5,.9,.99]))/3600).tolist() if count else None,
             'conditional_delay_over24h_Brier':float((forecasts[:,k,n+4]-(t>86400).double()).square()[yes].mean()) if count else None,
             'conditional_zero_delay_Brier':float((forecasts[:,k,n+5]-(t==0).double()).square()[yes].mean()) if count else None,
             'conditional_time_pinball_hours':(torch.maximum((t[:,None]-forecasts[:,k,n+1:n+4])*t.new_tensor([.05,.5,.95]),(t[:,None]-forecasts[:,k,n+1:n+4])*t.new_tensor([-.95,-.5,-.05]))[yes].mean(0)/3600).tolist() if count else None,
             'exact_observed_suffix_joint_NLL_through_h':float(joint[:,:h].sum(-1).mean()),
             'exact_suffix_factor_count':int(path_masks[:,:h].sum()),
             'generated_path_time_monotone':bool((paths[:,:,1:,1]>=paths[:,:,:-1,1]).all())}
        rows[str(h)]=raw
    return rows


@torch.no_grad()
def score(cache):
    torch.set_num_threads(1);cohorts,pre,_=source(cache);records=json.loads((BUNDLE/'CHECKPOINTS.json').read_text())
    if len(records)!=5:raise ValueError('all five original models required')
    out={};arrays={}
    for row in records:
        verify_blob(row['path'],row['checkpoint_sha256']);saved=torch.load(row['path'],map_location='cpu',weights_only=False)
        if saved['preprocessing']!=pre:raise ValueError('changed exact preprocessing')
        model=TrajectoryModel(pre);model.load_state_dict(saved['model'],strict=True)
        opt=budgeted_optimizer(model, torch.optim.AdamW, model.parameters());opt.load_state_dict(saved['optimizer']);model.double().eval()
        out[str(row['seed'])]={}
        for age in AGES:
            cases=[c for c in cohorts['REPORT'] if c['n']>=age];all_f=[];all_p=[];all_j=[];all_m=[]
            for start in range(0,len(cases),32):
                data=cast_data(pack(cases,list(range(start,min(start+32,len(cases)))),length=age),torch.float64)
                args,_=observed(model,data,all_steps=False);f,p=generate(model,*args)
                _,joint,masks,_,_=teacher_path(model,*args,data[5][:,-1],data[6][:,-1],data[7][:,-1],data[8][:,-1])
                all_f.append(f);all_p.append(p);all_j.append(joint);all_m.append(masks)
            data=cast_data(pack(cases,list(range(len(cases))),length=age),torch.float64)
            f,p,j,m=map(lambda x:torch.cat(x), (all_f,all_p,all_j,all_m))
            out[str(row['seed'])][str(age)]={'cases':len(cases),'IDs_sha256':digest([c['id'] for c in cases]),
                'support':'sparse development <25 cases' if len(cases)<25 else 'exposed development only','horizons':metrics(f,p,data,j,m)}
            key=f's{row["seed"]}_age{age}';arrays[key]=f.numpy();arrays[key+'_paths']=p.numpy();arrays[key+'_exact_suffix_NLL']=j.numpy()
            print(json.dumps({'scored_seed':row['seed'],'query_age':age,'cases':len(cases)}),flush=True)
    np.savez_compressed(BUNDLE/'ORIGINAL_FORECASTS.npz',**arrays)
    write_json(BUNDLE/'ORIGINAL_SCORES.json',{'scope':'exposed DEV original records; generated marginals finite-QMC, exact suffix likelihood separately; EOR retrospective proxy, not verified business closure','seeds':out})


def replay(cache):
    paths=[BUNDLE/'ORIGINAL_FORECASTS.npz',BUNDLE/'ORIGINAL_SCORES.json']
    for p in paths:verify_blob(p)
    original={p:p.read_bytes() for p in paths}
    with np.load(paths[0]) as z:arrays={k:v.copy() for k,v in z.items()}
    audit=ROOT/'REPLAY_AUDIT.json';audit.unlink(missing_ok=True)
    try:
        score(cache);tol=json.loads((ROOT/'FAIR_COMPARISON.json').read_text())['numerical_contract']
        with np.load(paths[0]) as z:
            if set(z)!=set(arrays):raise ValueError('original forecast identities differ')
            for k,v in z.items():np.testing.assert_allclose(v,arrays[k],atol=tol['primary_atol'],rtol=tol['primary_rtol'])
        compare_scores(json.loads(original[paths[1]]),json.loads(paths[1].read_text()),tol)
    except Exception as error:
        for p in paths:(p.parent/('FAILED_REPLAY_'+p.name)).write_bytes(p.read_bytes())
        write_json(audit,{'status':'FAIL','error':repr(error),'original_files_restored':True});print(audit.read_text(),flush=True);raise
    finally:
        for p,raw in original.items():p.write_bytes(raw)
    write_json(audit,{'status':'PASS','original_models_restored':5,'all_real_REPORT_metrics_recomputed':True,
        'original_free_run_distributions_and_paths_recomputed':True,'parameters_refit':False,'TEST_scored':False})
    print(audit.read_text(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--phase',choices=('register','train','score','replay'),required=True);a=p.parse_args()
    {'register':register,'train':train,'score':score,'replay':replay}[a.phase](a.cache)
