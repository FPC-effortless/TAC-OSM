"""ME078 prospective integrated physical learning and controlled rivals.

Original authenticated source, dense causal supervision, no TEST tuning. Qualify
and freeze code/data/parameter/operation parity BEFORE any optimizer updates.
All 15 original models Git-frozen before reporting developmental predictions.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,math,platform,random,subprocess,time,resource
from pathlib import Path
import numpy as np
import torch
from scripts.me076_train_system1_real import load_original,evaluation_origins,digest,sha,write,build_batch
from scripts.validate_research_parity import check_manifest
from scripts.me078_operation_accounting import profile
from tac_osm.variants.kinematic_world import KinematicWorld,HORIZONS,dense_loss
from tac_osm.parameter_budget import budgeted_optimizer,checked_optimizer_step,save_training_checkpoint

ROOT=Path('research/mathematical-engineering/me078');OUT=Path('research/replay-bundles/ME078');MANIFEST=ROOT/'FAIR_COMPARISON.json'
ARMS=('native','gru','transformer');SEEDS=tuple(range(78001,78006));STEPS=2048;BATCH=24


def pools(rows,identified):
    result={};physical=np.asarray(identified['state_scale'])
    for key,row in sorted(rows.items()):
        x=row['state'].numpy();human=row['human'].numpy();cumulative=np.r_[0,np.cumsum(human)]
        times=np.arange(8,len(x)-1,dtype=np.int64)
        valid=(cumulative[times+2]-cumulative[times-8])==0
        times=times[valid]
        if len(times):
            moving=np.mean(np.abs(x[times+1]-x[times])*physical,-1)>=.025
            result[key]=(times,moving)
    return result


def schedule(source,identified,seed):
    pp=pools(source,identified);tasks=sorted({k[0] for k in pp});allrows={t:[] for t in tasks};moving={t:[] for t in tasks}
    moveflags=set()
    for (task,ep),(times,flags) in pp.items():
        for t,flag in zip(times,flags):
            entry=(task,ep,int(t));allrows[task].append(entry)
            if flag:moving[task].append(entry);moveflags.add(entry)
    rng=np.random.default_rng(seed);orders=[];weights=[]
    for step in range(STEPS):
        entries=[];ww=[]
        for b in range(BATCH):
            task=tasks[(step*BATCH+b)%len(tasks)]
            has_motion=bool(moving[task]);selected=moving[task] if b%2==0 and has_motion else allrows[task]
            entry=selected[int(rng.integers(len(selected)))];entries.append(entry)
            rate=len(moving[task])/len(allrows[task])
            ww.append(1/(.5+.5/rate) if entry in moveflags and has_motion else 2. if has_motion else 1.)
        orders.append(entries);weights.append(ww)
    return orders,weights,{'sample_order_sha256':digest(orders),'importance_sha256':digest(weights),
       'task_motion_prevalence':{t:len(moving[t])/len(allrows[t]) for t in tasks},
       'unique_FIT_eligible_origins':sum(len(v) for v in allrows.values()),
       'sampling':'50%uniform+50%moving, exact balance-heuristic importance to natural within-task distribution'}


def batch(rows,entries):
    x,a,_=build_batch(rows,entries);labels=[];valid=[]
    for task,ep,t in entries:
        row=rows[(task,ep)];ys=[];masks=[]
        for j in range(8):
            origin=t-7+j; yy=[];mm=[]
            for horizon in HORIZONS:
                dest=origin+horizon
                okay=dest<len(row['state']) and not bool(row['human'][origin:dest+1].any())
                yy.append(row['state'][min(dest,len(row['state'])-1)]);mm.append(okay)
            ys.append(torch.stack(yy));masks.append(mm)
        labels.append(torch.stack(ys));valid.append(masks)
    return x,a,torch.stack(labels),torch.tensor(valid,dtype=torch.bool)


def model(arm,identified):return KinematicWorld(arm,identified=identified)

def integrity():
    m=json.loads(MANIFEST.read_text())
    for path,d in m['implementation_sha256'].items():
        if sha(path)!=d:raise RuntimeError('frozen implementation changed: '+path)
    check_manifest(m,'preflight')
    return m


def qualify(cache):
    ROOT.mkdir(parents=True,exist_ok=True)
    _,fit,report,identified,evidence=load_original(cache)
    queries,query_evidence=evaluation_origins(report,identified)
    source={'FIT':evidence['FIT_rows_sha256'],'REPORT':evidence['REPORT_rows_sha256'],'query':query_evidence['REPORT_selected_query_sha256']}
    normalizer=float(np.mean(identified['delta_scale']*identified['state_scale']))
    orders={};weights={};stats={}
    for seed in SEEDS:orders[seed],weights[seed],stats[seed]=schedule(fit,identified,seed)
    x,a,y,valid=batch(fit,orders[SEEDS[0]][0]);profiles={}
    for arm in ARMS:
        torch.manual_seed(SEEDS[0]);m=model(arm,identified)
        profiles[arm]=profile(m,x,a,y,x,valid,torch.tensor(weights[SEEDS[0]][0]),normalizer)
    operation_gap=max(r['total_update_operations'] for r in profiles.values())/min(r['total_update_operations'] for r in profiles.values())-1
    parameter_gap=max(r['parameter_count'] for r in profiles.values())/min(r['parameter_count'] for r in profiles.values())-1
    write(ROOT/'RESOURCE_PREFLIGHT.json',{'profiles':profiles,'max_pairwise_operations_gap':operation_gap,'max_pairwise_parameter_gap':parameter_gap,
       'method':'Actual ATen forward/backward/gradient-clip scalar arithmetic ledger with ZERO unsupported floating operators + analytic AdamW16N+20tensor+16; elementary transcendental=one scalar operation; NOT hardware instructions/FLOPs/energy equivalence.'})
    if operation_gap>.01 or parameter_gap>.01:raise RuntimeError('Prospective matching FAILED; NO FITS AUTHORIZED: '+str((operation_gap,parameter_gap)))
    code=['src/tac_osm/variants/kinematic_world.py','scripts/me078_train_world.py','scripts/me078_operation_accounting.py','scripts/me076_train_system1_real.py','scripts/me064_mixed_real_fit.py','scripts/me057_train_fused_anchor_robot.py','src/tac_osm/parameter_budget.py','scripts/validate_research_parity.py']
    manifest={'schema_version':'tac-osm.parity.v2','study_id':'ME078_NATIVE_KINEMATIC_MULTI_HORIZON_COMPARISON','study_type':'architecture_comparison','registered_before_first_fit':True,
      'real_world_source':{'name':'BodenAI RW-RL-HIL original160FIT episodes, pinned9290f5a96dc91ea58256e81d9ce2dd9621890b51;120FIT/40previously exposed REPORT','license':'CC-BY-4.0, BodenAI and parent RW-RL attribution','observed_real_outcomes':True,'sha256':evidence['original_160_episode_source_sha256']},
      'literature_gap':{'unresolved_TAConly_question':'Does an integrated native fast/slow/8x8 associative physical model acquire meaningful motion and multihorizon forecasting beyond simple physical references and shared-interface GRU/causal attention under matched observed training budgets?',
        'what_is_not_being_rediscovered':'Gated delta correction, GRU, attention, dense causal multitask supervision, importance weighting and pinball quantiles are established methods, not TAC inventions.',
        'established_work':[{'paper':'Gated Delta Networks','url':'https://arxiv.org/abs/2412.06464','established_result':'Gated error-correcting fast-weight updates.'},{'paper':'Attention Is All You Need','url':'https://arxiv.org/abs/1706.03762','established_result':'Causal self-attention sequence prediction.'},{'paper':'Learning Phrase Representations using RNN Encoder-Decoder','url':'https://arxiv.org/abs/1406.1078','established_result':'Gated recurrent unit dynamics.'},{'paper':'Regression Quantiles','url':'https://doi.org/10.2307/1913643','established_result':'Pinball quantile estimation.'}]},
      'implementation_sha256':{p:sha(p) for p in code},'arms':[],
      'training_budget':{'updates':STEPS,'batch_size':BATCH,'history':8,'horizons':list(HORIZONS),'dense_supervision_prefixes':8,'optimizer':'AdamW lr0.0007 wd0.0001 clip1','selection':'final checkpoint only; no search, restarts or REPORT tuning'},
      'compute_scope':'Audited algorithmic scalar-operation estimates, including all observed ATen float operators and AdamW analytical charge. Hardware FLOPs, energy and foundation-scale superiority not established.',
      'evaluation':{'exposure':'EXPOSED_DEV','queries':query_evidence,'cohorts':'exact ME076/077960query origins; H1all; H4/H16only factual pre-takeover observed futures, denominator per horizon explicit',
        'primary_gate':'Native H1all MAE >=5%better than persistence AND old ME077; H1moving >=5%better than strongest physical persistence/servo/velocity reference; native aggregate scale-free horizon MAE >=2%better than BOTH new matched GRU and Transformer, adjusted paired episode bootstrap upper<0 vsboth; >=4/5seed native wins vs each; no >10%task regression vspersistence.',
        'confidence':'5000paired episode-cluster bootstrap of five-seed-mean query scores;97.5%intervals for two architecture comparisons (Bonferroni family95%); seed78078; exposed DEV only',
        'references':'persistence, clipped constant velocity, FIT lag1servo, frozen ME077H1same exact IDs; no oracle routing or hindsight prevalence as deployable prior',
        'replay_rtol':1e-5,'replay_atol':1e-6,'sealed_TEST_opened':False,'protected_tissue_box_opened':False,'original_all_models_Git_frozen_before_scoring':True}}
    commonbudget={'steps':STEPS,'examples_seen':STEPS*BATCH,'tokens_seen':STEPS*BATCH*8,'batch_size':BATCH,'history_events':8,
       'max_training_flops':max(r['total_update_operations'] for r in profiles.values())*STEPS*1.005,'tuning_trials':0,'tuning_flops':0,
       'prefit_profile_method':'ATen scalar-operation ledger, all float ops classified; analytical AdamW16N+20tensors+16; same fixed algorithmic convention, includes dense loss, backward and gradient clipping; hardware FLOPs unclaimed'}
    contracts={'feature_contract_sha256':digest('observed8qpos; lag1command mapped to qpos coordinates; observed velocity/acceleration asinh; nominal dt; native/GRU/attention SAME features'),
       'target_contract_sha256':digest('dense 8prefixes × observed next1/4/16 physical14D states + genuine motion .025; valid pre-takeover targets only'),
       'optimizer_schedule_sha256':digest(manifest['training_budget']),'tuning_policy_sha256':digest('zero tuning; no restarts; all5seedsfinal'),
       'selection_policy_sha256':digest('final2048 only'),'evaluation_contract_sha256':digest(manifest['evaluation'])}
    for seed in SEEDS:
        for arm in ARMS:
            r=profiles[arm];b=dict(commonbudget,prefit_profile_training_flops=r['total_update_operations'],planned_training_flops=r['total_update_operations']*STEPS)
            manifest['arms'].append({'architecture':arm,'seed':seed,'trainable_parameters':r['parameter_count'],'deployed_parameters':r['parameter_count'],
              'nontrainable_state_bytes':512 if arm=='native' else 224 if arm=='gru' else 4096,
              'dataset_sha256':evidence['original_160_episode_source_sha256'],'train_split_sha256':source['FIT'],'dev_split_sha256':source['REPORT'],'test_split_sha256':source['query'],
              'training_sample_order_sha256':stats[seed]['sample_order_sha256'],**contracts,'hardware_profile':platform.processor()+'CPU;torch'+torch.__version__+';threads1;same execution host','budget':b})
    check_manifest(manifest,'preflight');write(MANIFEST,manifest)
    write(ROOT/'SOURCE_QUALIFICATION.json',{'source':evidence,'queries':query_evidence,'FIT_only_loss_normalizer':normalizer,'schedules':stats,
       'horizon_available_queries':{str(h):sum(t+h<len(report[(task,ep)]['state']) and not bool(report[(task,ep)]['human'][t:t+h+1].any()) for task,ep,t in queries) for h in HORIZONS}})
    print('PREFLIGHT_PASS',operation_gap,parameter_gap,flush=True)


def fit_all(cache):
    reg=integrity()
    if list(OUT.glob('*/seed-*/final.pt')):raise RuntimeError('never overwrite original model fits')
    _,fit,report,identified,evidence=load_original(cache)
    normalizer=float(np.mean(identified['delta_scale']*identified['state_scale']))
    index=[]
    for seed in SEEDS:
        orders,weights,stats=schedule(fit,identified,seed)
        for arm in ARMS:
            torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
            m=model(arm,identified);folder=OUT/arm/f'seed-{seed}'
            optimizer=budgeted_optimizer(m,torch.optim.AdamW,m.parameters(),lr=.0007,weight_decay=.0001,budget_manifest=MANIFEST,budget_architecture=arm,budget_seed=seed,budget_receipt=folder/'live-budget.json')
            trace=[];start=time.process_time();actual_valid=0
            for step,(picks,ww) in enumerate(zip(orders,weights),1):
                x,a,y,valid=batch(fit,picks);actual_valid+=int(valid.sum())
                prediction=m(x,a);loss=dense_loss(prediction,y,x,valid,torch.tensor(ww),m.state_scale,normalizer)
                if not torch.isfinite(loss):raise RuntimeError('nonfinite authentic loss')
                optimizer.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
                if not torch.isfinite(norm):raise RuntimeError('nonfinite gradient')
                checked_optimizer_step(optimizer)
                if step==1 or step%256==0:
                    trace.append({'step':step,'FIT_loss':float(loss.detach())});print(json.dumps({'arm':arm,'seed':seed,**trace[-1]}),flush=True)
            checkpoint=folder/'final.pt'
            save_training_checkpoint({'weights':m.state_dict(),'optimizer':optimizer.state_dict(),'seed':seed,'architecture':arm,'updates':STEPS,'torch_rng':torch.get_rng_state(),'numpy_rng':(np.random.get_state()[0],np.random.get_state()[1].tolist(),*np.random.get_state()[2:]),'python_rng':random.getstate(),'FIT_only_identified':{k:v.tolist() for k,v in identified.items()},'schedule':stats,'valid_supervised_targets':actual_valid,'prefit_registration_sha256':sha(MANIFEST)},checkpoint)
            # Reprofile the fitted whole model using the same actual first FIT batch.
            x,a,y,valid=batch(fit,orders[0]);post=profile(m,x,a,y,x,valid,torch.tensor(weights[0]),normalizer)
            original=json.loads((ROOT/'RESOURCE_PREFLIGHT.json').read_text())['profiles'][arm]
            if post['total_update_operations']!=original['total_update_operations']:raise RuntimeError('fitted resource graph differs from prefit')
            entry={'seed':seed,'architecture':arm,'checkpoint_sha256':sha(checkpoint),'actual_steps':STEPS,'actual_examples':STEPS*BATCH,'actual_tokens':STEPS*BATCH*8,
              'valid_dense_future_target_presentations':actual_valid,'sampling':stats,'training_cpu_seconds':time.process_time()-start,'trace':trace,'postfit_profile':post,
              'peak_process_memory_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
            write(folder/'FIT_RECEIPT.json',entry);index.append(entry);write(OUT/'FIT_INDEX.json',index)
    print('ALL15_ORIGINAL_MODELS_COMPLETE_NO_REPORT_SCORED',flush=True)


def frozen():
    for arm in ARMS:
        for seed in SEEDS:
            for name in ('final.pt','live-budget.json','FIT_RECEIPT.json'):
                p=OUT/arm/f'seed-{seed}'/name
                if hashlib.sha256(subprocess.check_output(['git','show','HEAD:'+str(p)])).hexdigest()!=sha(p):raise RuntimeError('all original model bytes must be Git-frozen BEFORE scoring')


def finalize():
    reg=integrity();frozen();final=copy.deepcopy(reg);final['prefit_registration_sha256']=sha(MANIFEST)
    for row in final['arms']:
        folder=OUT/row['architecture']/f"seed-{row['seed']}";fit=json.loads((folder/'FIT_RECEIPT.json').read_text())
        row['measured']={'actual_steps':fit['actual_steps'],'actual_examples':fit['actual_examples'],'actual_tokens':fit['actual_tokens'],
          'training_flops':fit['postfit_profile']['total_update_operations']*STEPS,'optimizer_cpu_seconds':fit['training_cpu_seconds'],'peak_memory_bytes':fit['peak_process_memory_bytes'],
          'flops_profiler':row['budget']['prefit_profile_method'],'checkpoint_sha256':sha(folder/'final.pt'),
          'actual_sample_order_sha256':fit['sampling']['sample_order_sha256'],'evaluation_split_sha256':row['test_split_sha256'],
          'trainable_parameters':row['trainable_parameters'],'deployed_parameters':row['deployed_parameters'],
          'runtime_evidence':{'receipt_path':str(folder/'live-budget.json'),'receipt_sha256':sha(folder/'live-budget.json'),'checkpoint_path':str(folder/'final.pt'),'checkpoint_sha256':sha(folder/'final.pt')}}
    status=check_manifest(final,'final');write(ROOT/'FINAL_COMPARISON.json',final);print(status,flush=True)


def score_one(m,rows,identified,queries):
    physical=torch.as_tensor(identified['state_scale']);results=[];m.eval()
    with torch.no_grad():
        for i in range(0,len(queries),24):
            picks=queries[i:i+24];x,a,y,valid=batch(rows,picks);pred=m(x,a)
            pred={k:v[:,-1] for k,v in pred.items()};target=y[:,-1];available=valid[:,-1]
            velocity=x[:,-1]-x[:,-2]
            command=(a[:,-1]*m.action_scale+m.action_mean-m.state_mean)/m.state_scale-x[:,-1]
            servo=torch.clamp(m.servo_gain*command,-m.servo_step_limit,m.servo_step_limit)
            for j,(task,ep,t) in enumerate(picks):
                for hidx,horizon in enumerate(HORIZONS):
                    if not available[j,hidx]:continue
                    truth=target[j,hidx];center=pred['median'][j,hidx];lower=pred['lower'][j,hidx];upper=pred['upper'][j,hidx]
                    future=truth-x[j,-1];movement=float((future.abs()*physical).mean())
                    q=pred['motion_logit'][j,hidx].sigmoid();event=movement>=.025
                    refs={'persistence':x[j,-1],'velocity':x[j,-1]+horizon*torch.clamp(velocity[j],-m.servo_step_limit,m.servo_step_limit),'servo':x[j,-1]+horizon*servo[j]}
                    record={'task':task,'episode':ep,'frame':t,'horizon':horizon,'moving':event,'physical_movement':movement,
                      'mae':float(((center-truth).abs()*physical).mean()),'motion_probability':float(q),'motion_brier':float((q-float(event)).square()),
                      'coverage':float(((lower<=truth)&(truth<=upper)).float().mean()),'width':float(((upper-lower)*physical).mean()),
                      'interval_score':float(((upper-lower+20*(lower-truth).clamp_min(0)+20*(truth-upper).clamp_min(0))*physical).mean()),
                      'prediction':center.tolist(),'target':truth.tolist(),'lower':lower.tolist(),'upper':upper.tolist()}
                    record.update({name+'_mae':float(((value-truth).abs()*physical).mean()) for name,value in refs.items()});results.append(record)
    return results


def run_scores(cache,replay=False):
    integrity();frozen();_,_,report,identified,evidence=load_original(cache);queries,_=evaluation_origins(report,identified);index=[]
    for arm in ARMS:
        for seed in SEEDS:
            folder=OUT/arm/f'seed-{seed}';fit=json.loads((folder/'FIT_RECEIPT.json').read_text())
            if sha(folder/'final.pt')!=fit['checkpoint_sha256']:raise RuntimeError('checkpoint changed')
            m=model(arm,identified);state=torch.load(folder/'final.pt',map_location='cpu',weights_only=True);m.load_state_dict(state['weights'],strict=True)
            rows=score_one(m,report,identified,queries);path=folder/'ORIGINAL_PREDICTIONS.json'
            if replay:
                original=json.loads(path.read_text())
                if len(original)!=len(rows):raise RuntimeError('query denominator differs')
                for old,new in zip(original,rows):
                    for key in old:
                        if isinstance(old[key],(float,list)):
                            if not np.allclose(old[key],new[key],rtol=1e-5,atol=1e-6):raise RuntimeError(f'raw replay failed:{arm}/{seed}/{key}')
                        elif old[key]!=new[key]:raise RuntimeError('query identity/labels differ')
            else:
                if path.exists():raise RuntimeError('never overwrite original model scores')
                write(path,rows)
            index.append({'architecture':arm,'seed':seed,'checkpoint_sha256':sha(folder/'final.pt'),'predictions_sha256':sha(path),'forecasts':len(rows)})
    if replay:write(OUT/'INDEPENDENT_REPLAY.json',{'status':'PASS','original_models':15,'original_forecasts':sum(r['forecasts'] for r in index),'rtol':1e-5,'atol':1e-6,'models':index,'no_refit':True,'source_sha256':evidence['original_160_episode_source_sha256']})
    else:write(OUT/'ORIGINAL_SCORE_INDEX.json',index)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['qualify','fit','finalize','score','replay']);p.add_argument('--cache',type=Path,default=Path('/tmp/me077-original-source'));args=p.parse_args();torch.set_num_threads(1)
    if args.mode=='qualify':qualify(args.cache)
    elif args.mode=='fit':fit_all(args.cache)
    elif args.mode=='finalize':finalize()
    else:run_scores(args.cache,args.mode=='replay')
