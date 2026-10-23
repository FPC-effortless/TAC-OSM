"""ME081 prospectively registered solo native capability training and replay.

No architecture-winner or equal-compute claim. Five cold starts, all original
models Git-frozen before exposed REPORT scoring. No sealed source access.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,random,subprocess,time,platform
from pathlib import Path
import numpy as np
import torch
from scripts.me076_train_system1_real import load_original,evaluation_origins,sha,write,digest

from scripts.validate_research_parity import check_manifest
from tac_osm.parameter_budget import budgeted_optimizer,checked_optimizer_step,save_training_checkpoint,validate_runtime_evidence
from tac_osm.variants.streaming_mixture_world import quantile,log_density
from tac_osm.variants.horizon_query_world import HorizonQueryWorld,horizon_distribution_loss

ROOT=Path('research/mathematical-engineering/me081');OUT=Path('research/replay-bundles/ME081');MANIFEST=ROOT/'FAIR_COMPARISON.json'
SEEDS=tuple(range(81001,81006));STEPS=2048;BATCH=24;HORIZONS=(1,4,16);COUNT=100165
CODE=['scripts/me081_train_world.py','src/tac_osm/variants/streaming_mixture_world.py','src/tac_osm/variants/reusable_execution.py','src/tac_osm/variants/horizon_query_world.py',
      'src/tac_osm/variants/kinematic_world.py','scripts/me079_train_world.py',
      'scripts/me076_train_system1_real.py','scripts/me064_mixed_real_fit.py',
      'scripts/me057_train_fused_anchor_robot.py','src/tac_osm/parameter_budget.py',
      'scripts/validate_research_parity.py']


def pools(rows,identified):
    result={};physical=identified['state_scale']
    for key,row in sorted(rows.items()):
        human=row['human'].numpy();cs=np.r_[0,np.cumsum(human)];t=np.arange(32,len(human)-1)
        t=t[(cs[t+2]-cs[t-32])==0]
        if len(t):
            x=row['state'].numpy();flags=np.mean(np.abs(x[t+1]-x[t])*physical,-1)>=.025
            result[key]=(t,flags)
    return result


def schedule(rows,identified,seed):
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
    return orders,weights,{'sample_order_sha256':digest(orders),'importance_sha256':digest(weights),
       'unique_FIT_eligible_origins':sum(len(v) for v in allrows.values()),'task_motion_prevalence':{t:len(moving[t])/len(allrows[t]) for t in tasks}}


def batch(rows,entries,history=32):
    xs=[];actions=[];labels=[];valid=[]
    for task,ep,t in entries:
        row=rows[(task,ep)]
        if t<history:raise ValueError('actual prior observations and lagged commands required')
        xs.append(row['state'][t-history+1:t+1]);actions.append(row['action'][t-history:t])
        yy=[];mm=[]
        for origin in range(t-history+1,t+1):
            ys=[];masks=[]
            for h in HORIZONS:
                dest=origin+h;okay=dest<len(row['state']) and not bool(row['human'][origin:dest+1].any())
                ys.append(row['state'][min(dest,len(row['state'])-1)]);masks.append(okay)
            yy.append(torch.stack(ys));mm.append(masks)
        labels.append(torch.stack(yy));valid.append(mm)
    return torch.stack(xs),torch.stack(actions),torch.stack(labels),torch.tensor(valid,dtype=torch.bool)


def long_queries(rows,queries):
    return [(task,ep,t) for task,ep,t in queries if t>=32 and not bool(rows[(task,ep)]['human'][t-32:t+1].any())]


def integrity():
    reg=json.loads(MANIFEST.read_text());check_manifest(reg,'preflight')
    for p,d in reg['implementation_sha256'].items():
        if sha(p)!=d:raise RuntimeError('Frozen source changed: '+p)
    return reg


def qualify(cache):
    if MANIFEST.exists():raise RuntimeError('Never replace a frozen registration')
    _,fit,report,identified,evidence=load_original(cache)
    queries,query_evidence=evaluation_origins(report,identified)
    m=HorizonQueryWorld(identified)
    n=sum(p.numel() for p in m.parameters())
    if n!=COUNT:raise RuntimeError('Actual parameter count differs')
    schedules={str(seed):schedule(fit,identified,seed)[2] for seed in SEEDS}
    reg={'schema_version':'tac-osm.parity.v2','study_id':'ME081_HORIZON_QUERY_CRPS_LONG_CONTEXT',
         'study_type':'single_model_training','registered_before_first_fit':True,
         'no_architecture_superiority_claim':True,
         'real_world_source':{'name':'BodenAI/RW-RL-HIL-Dataset original160 episode subset; revision9290f5a96dc91ea58256e81d9ce2dd9621890b51;120FIT/40previously exposedREPORT',
           'license':'CC-BY-4.0; BodenAI and RW-RL attribution','observed_real_outcomes':True,'sha256':evidence['original_160_episode_source_sha256']},
         'models':[{'architecture':m.architecture,'seed':s,'trainable_parameters':n,'deployed_parameters':n} for s in SEEDS],
         'implementation_sha256':{p:sha(p) for p in CODE},
         'training_budget':{'updates':STEPS,'batch_size':BATCH,'history':32,'dense_prefixes':32,'horizons':list(HORIZONS),
           'optimizer':'AdamW lr0.0007 wd0.0001 clip1','seeds':list(SEEDS),'selection':'final only; no sweep/restart/warm start; five cold starts',
           'sampling':'same nominal 50%uniform+50%movement proposal and balance importance as ME078/079; finite task round-robin counts differ by at most1/component',
           'loss':'proper physical normalized CRPS +0.02physical NLL +0.2factual motion BCE; shared horizon-query decoder with ordered component scales, all eligible dense prefixes/horizons; natural target importance weights',
           'compute':'actual2048updates/49152windows/1572864observed prefix presentations per seed; NO FLOP-matched comparison or exact hardware FLOP claim'},
         'literature':[
           {'paper':'Strictly Proper Scoring Rules, Prediction, and Estimation; Gneiting and Raftery2007','url':'https://doi.org/10.1198/016214506000001437','reuse':'CRPS and logarithmic scoring as proper distributional objectives'},
           {'paper':'DeepAR: Probabilistic forecasting with autoregressive recurrent networks,2020','url':'https://doi.org/10.1016/j.ijforecast.2019.07.001','reuse':'shared conditional probabilistic forecasting across sequences; not a new TAC discovery'},
           {'paper':'Mixture Density Networks, Bishop1994','url':'https://publications.aston.ac.uk/id/eprint/373/','reuse':'conditional mixture likelihood and distributional predictions'},
           {'paper':'Gated Delta Networks','url':'https://arxiv.org/abs/2412.06464','reuse':'gated error-correcting associative updates, not invented here'},
           {'paper':'R2D2, ICLR2019','url':'https://willdabney.com/publication/r2d2/','reuse':'explicit recurrent state and state staleness cautions; no claim this eight-frame study proves long retention'},
           {'paper':'Mamba-2/SSD, ICML2024','url':'https://proceedings.mlr.press/v235/dao24a.html','reuse':'hardware-aware stateful computation motivation; native nonlinear TAC is NOT a linear scan'}],
         'unresolved_capability':'Can native32frame state learning, a horizon-query decoder and proper CRPS distribution training jointly improve actual quiet/motion and multihorizon predictive skill?',
         'evaluation':{'exposure':'EXPOSED_DEV','queries':query_evidence,'old_frozen_full_models':['ME077','ME078','ME079','ME080'],
           'horizons':list(HORIZONS),'all_eligible_origins':True,'sealed_TEST_opened':False,'protected_tissue_box_opened':False,
           'primary':'same original8frame query H1MAE <=0.95*ME080, movingH1 <=1.05*ME078, stationaryH1 <=1.10*ME077; all simultaneous. H4 <=0.95*ME079 and H16<=1.05*ME080. Also score all causally eligible32frame original query prefixes separately; no differing-input old/new winner claim',
           'guardrails':'H4/H16see primary; secondary32frame H1/H4/H16mean MAE must beat persistence, movingH1<=velocity;  no task H1>1.10*persistence; nominal90%coordinate coverage >=.85 and proper interval score <=ME080 at each horizon',
           'decision':'DEV_GAIN only if every primary+guardrail passes and paired episode bootstrap95%upper(new-ME080 H1)<0, >=4/5seed improves ME080; otherwise honest MIXED_TRADEOFF/REGRESSION/INCONCLUSIVE. One recipe only; no additional fits under registration.',
           'statistics':'5000paired episode-cluster bootstrap, fixedseed81081, conditional on fixed fivefits; all40episode clusters; no independent-source promotion',
           'numeric':'float32 neural parameters, float64 mixture quantile inversion40bisections and ALL scalar scoring; raw rtol1e-5/atol1e-6 frozen prospectively; no old-study tolerance edits',
           'models_Git_frozen_before_REPORT':True,'runtime_scope':'functional step/chunk equivalence and actual stored cache; trained32frame full gradients; primary8frame and secondary32frame factual forecasts; arbitrary-history or memory superiority NOT established'},
         'source_qualification_sha256':digest(evidence),'schedules':schedules,'secondary32_queries_sha256':digest(long_queries(report,queries)),'secondary32_queries':len(long_queries(report,queries)),
         'capability_to_value':{'skill':'native longer-context multi-horizon state distributions with sharper proper uncertainty and actual-motion skill',
           'operator_question':'Could a read-only physical forecast help a human notice genuine movement without jitter?','external_pilot':'NOT_MEASURED',
           'customer_value':'UNKNOWN','business_stage':'research only; no actions, deployment or commercial qualification'}}
    check_manifest(reg,'preflight');write(MANIFEST,reg)
    write(ROOT/'SOURCE_QUALIFICATION.json',{'source':evidence,'queries':query_evidence,'schedules':schedules,'secondary32_queries_sha256':digest(long_queries(report,queries)),'secondary32_queries':len(long_queries(report,queries)),
       'actual_trainable_parameters':n,'actual_deployed_parameters':n,'horizon_available_queries':{str(h):sum(t+h<len(report[(task,ep)]['state']) and not bool(report[(task,ep)]['human'][t:t+h+1].any()) for task,ep,t in queries) for h in HORIZONS}})
    print('ME081_PREFIT_PASS',n,flush=True)


def fit_all(cache):
    integrity()
    if list(OUT.glob('seed-*/final.pt')):raise RuntimeError('Never overwrite original fits')
    _,fit,_,identified,_=load_original(cache);index=[]
    normalizer=float(np.mean(identified['delta_scale']*identified['state_scale']))
    for seed in SEEDS:
        torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
        m=HorizonQueryWorld(identified);folder=OUT/f'seed-{seed}'
        orders,weights,stats=schedule(fit,identified,seed)
        if stats!=json.loads(MANIFEST.read_text())['schedules'][str(seed)]:raise RuntimeError('FIT schedule changed')
        opt=budgeted_optimizer(m,torch.optim.AdamW,m.parameters(),lr=.0007,weight_decay=.0001,budget_manifest=MANIFEST,budget_architecture=m.architecture,budget_seed=seed,budget_receipt=folder/'live-budget.json')
        trace=[];start=time.process_time();valid_count=0
        for step,(picks,ww) in enumerate(zip(orders,weights),1):
            x,a,y,valid=batch(fit,picks);valid_count+=int(valid.sum())
            pred=m(x,a);loss=horizon_distribution_loss(pred,y,x,valid,torch.tensor(ww),m.state_scale,normalizer)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite authentic loss')
            opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
            if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradient')
            checked_optimizer_step(opt)
            if step==1 or step%256==0:
                trace.append({'step':step,'FIT_loss':float(loss.detach())});print(json.dumps({'seed':seed,**trace[-1]}),flush=True)
        checkpoint=folder/'final.pt'
        save_training_checkpoint({'weights':m.state_dict(),'optimizer':opt.state_dict(),'seed':seed,'architecture':m.architecture,
           'updates':STEPS,'torch_rng':torch.get_rng_state(),'numpy_rng':(np.random.get_state()[0],np.random.get_state()[1].tolist(),*np.random.get_state()[2:]),
           'python_rng':random.getstate(),'FIT_only_identified':{k:v.tolist() for k,v in identified.items()},'schedule':stats,
           'valid_supervised_targets':valid_count,'prefit_registration_sha256':sha(MANIFEST)},checkpoint)
        rec={'seed':seed,'parameters':sum(p.numel() for p in m.parameters()),'deployed_parameters':sum(p.numel() for p in m.parameters()),
             'actual_steps':STEPS,'actual_examples':STEPS*BATCH,'actual_observed_prefixes':STEPS*BATCH*32,
             'valid_dense_target_presentations':valid_count,'training_cpu_seconds':time.process_time()-start,'sampling':stats,
             'checkpoint_sha256':sha(checkpoint),'trace':trace,'environment':{'torch':torch.__version__,'python':platform.python_version(),'threads':torch.get_num_threads()}}
        write(folder/'FIT_RECEIPT.json',rec);index.append(rec);write(OUT/'FIT_INDEX.json',index)
    print('ALL5_ORIGINALS_COMPLETE_NO_REPORT_SCORED',flush=True)


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
        m=HorizonQueryWorld(checkpoint['FIT_only_identified']);m.load_state_dict(checkpoint['weights'],strict=True)
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
            picks=queries[i:i+24];x,a,y,valid=batch(rows,picks,history);raw=m(x,a)
            raw={k:v[:,-1].double() for k,v in raw.items()}
            pred={key:quantile(raw,tau) for key,tau in [('median',.5),('lower',.05),('upper',.95)]}
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
        m=HorizonQueryWorld(identified);m.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['weights'],strict=True)
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
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['qualify','fit','finalize','score','replay']);parser.add_argument('--cache',type=Path,default=Path('/tmp/me077-original-source'));args=parser.parse_args();torch.set_num_threads(1)
    if args.mode=='qualify':qualify(args.cache)
    elif args.mode=='fit':fit_all(args.cache)
    elif args.mode=='finalize':finalize()
    else:run_scores(args.cache,args.mode=='replay')
