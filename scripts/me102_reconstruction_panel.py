"""ME102 fixed three-recipe/five-seed panel; replay never fits models."""
from pathlib import Path
import sys,json,hashlib,time,random,argparse,subprocess,resource
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts import me100_sensor_data as data
from scripts import me102_gap_data as gap
from scripts import me101_belief_sensor as prior
from scripts.validate_research_parity import check_manifest
from tac_osm.variants.reconstruction_sensor_world import ReconstructionSensorWorld
from tac_osm.parameter_budget import budgeted_optimizer,checked_optimizer_step,save_training_checkpoint,validate_runtime_evidence
OUT=ROOT/'research/replay-bundles/ME102';DOC=ROOT/'research/mathematical-engineering/me102';REG=DOC/'FAIR_COMPARISON.json'
SEEDS=tuple(range(102001,102006));STEPS=1536;BATCH=16
RECIPES={'light_floor':dict(aux=.3,floor=.25,logscore=.1,rawscore=0.),'strong_reconstruction':dict(aux=1.,floor=1.,logscore=.1,rawscore=0.),'physical_score':dict(aux=.3,floor=1.,logscore=0.,rawscore=.1)}
SOURCES=['scripts/me102_reconstruction_panel.py','scripts/me102_graph_reference.py','scripts/me102_gap_data.py','scripts/me100_sensor_data.py','scripts/me101_graph_reference.py','scripts/me100_graph_reference.py','src/tac_osm/variants/reconstruction_sensor_world.py','src/tac_osm/variants/belief_sensor_world.py','src/tac_osm/variants/entity_sensor_world.py','src/tac_osm/variants/parallel_affine_memory.py','src/tac_osm/variants/retention_controlled_world.py','src/tac_osm/parameter_budget.py']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def tensor_batch(d,origins,length,rng):return [torch.from_numpy(x) for x in gap.augmentation(d,origins,length,rng)]
def initialize(d,recipe,seed):
    torch.manual_seed(seed);m=ReconstructionSensorWorld(torch.tensor(gap.floor_rates(d)['rates']*RECIPES[recipe]['floor'],dtype=torch.float32))
    m.target_mean.copy_(torch.from_numpy(d['mean'][:6]));m.target_scale.copy_(torch.from_numpy(d['std'][:6]));return m

def objective(m,p,tr,x,rawscale,recipe):
    c=RECIPES[recipe];mask=x[7];truth=x[6];z=(torch.log1p(truth)-m.target_mean)/m.target_scale
    def avg(v):return torch.where(mask,v,torch.zeros_like(v)).sum()/mask.sum().clamp_min(1)
    nll=m.nll(p,truth,mask);mse=avg((p.log_location-z).square());reconstruction=m.reconstruction_loss(tr,x[8],x[9])
    q=m.raw_quantiles(p);low,mid,high=q.unbind(-1)
    rawscore=avg(((high-low)+20*(low-truth).clamp_min(0)+20*(truth-high).clamp_min(0))/rawscale)
    zl=(torch.log1p(low)-m.target_mean)/m.target_scale;zh=(torch.log1p(high)-m.target_mean)/m.target_scale
    logscore=avg(zh-zl+20*(zl-z).clamp_min(0)+20*(z-zh).clamp_min(0))
    # Every recipe evaluates the same tensor graph, including both proper scores.
    loss=nll+.1*mse+c['aux']*reconstruction+c['logscore']*logscore+c['rawscore']*rawscore
    return loss,dict(nll=nll,mse=mse,reconstruction=reconstruction,logscore=logscore,rawscore=rawscore)

def profile(m,x,rawscale,recipe):
    m.zero_grad(set_to_none=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],with_flops=True,record_shapes=True) as prof:
        p,_,tr=m(*x[:6],return_trace=True);loss,_=objective(m,p,tr,x,rawscale,recipe);loss.backward()
    events=prof.key_averages(group_by_input_shape=True)
    ops=sorted((e.key,e.count,str(e.input_shapes)) for e in events if e.key.startswith('aten::'))
    return dict(dense_flops=sum(e.flops for e in events),aten_signature=digest(ops),aten_calls=sum(e.count for e in events if e.key.startswith('aten::')))

def prepare():
    if REG.exists():raise RuntimeError('registration exists; no overwrite')
    d=data.load('fit');DOC.mkdir(parents=True,exist_ok=True);plans={};order={}
    for seed in SEEDS:
        rng=np.random.default_rng(seed);rows=[]
        for _ in range(STEPS):rows.append([d['gap'][i] for i in rng.integers(len(d['gap']),size=8)]+[d['general'][i] for i in rng.integers(len(d['general']),size=8)])
        plans[str(seed)]=np.asarray(rows,dtype=np.int64);order[seed]=hashlib.sha256(plans[str(seed)].tobytes()).hexdigest()
    np.savez_compressed(DOC/'FIT_PLANS.npz',**plans)
    rawscale=torch.tensor(np.nanstd(d['raw'][...,:6].astype(np.float64),axis=(0,1)).clip(1e-4),dtype=torch.float32)
    profiles={}
    for recipe in RECIPES:
        profiles[recipe]={}
        for length in (8,64,256):
            x=tensor_batch(d,plans[str(SEEDS[0])][0],length,np.random.default_rng(1234))
            profiles[recipe][str(length)]=profile(initialize(d,recipe,SEEDS[0]),x,rawscale,recipe)
    for length in (8,64,256):
        rows=[profiles[r][str(length)] for r in RECIPES]
        if len({r['dense_flops'] for r in rows})!=1 or len({r['aten_signature'] for r in rows})!=1:raise RuntimeError('profiled graph parity failed before registration')
    rates=gap.floor_rates(d);write(DOC/'FIT_STATISTICS.json',dict(floor_rates=rates['rates'].tolist(),pair_counts=rates['counts'].tolist(),raw_scale=rawscale.tolist(),fit_only=True))
    write(DOC/'PREFIT_PROFILES.json',dict(profiles=profiles,coverage='torch.profiler dense forward/backward FLOP estimates; optimizer and unsupported pointwise FLOPs not fully counted; ATen operator/shape/count signature equality independently enforced',no_optimizer_updates=True))
    arms=[];common=dict(dataset_sha256=data.SHA,train_split_sha256=digest(['FIT2013-2014',data.SHA]),dev_split_sha256=digest(['EXPOSED2015',data.SHA]),test_split_sha256=digest(['EXPOSED2015daily-and-gap',data.SHA]),feature_contract_sha256=digest(['sameaugmented inputs',sha(ROOT/'scripts/me102_gap_data.py')]),target_contract_sha256=digest(['actualfuture6h','actualcurrenthiddenlabels']),optimizer_schedule_sha256=digest(['AdamW',.0006,.001,STEPS,16]),tuning_policy_sha256=digest(['threefixedrecipes','noadditionaltuning']),selection_policy_sha256=digest(['allrecipesretained','nodesignwinnerclaim']),evaluation_contract_sha256=digest(['same2015cohorts','fullgatesallseeds','test2016closed']),hardware_profile='same managed CPU,onePyTorchthread/model,max2workers')
    for seed in SEEDS:
        for recipe in RECIPES:
            prof=profiles[recipe];total=sum(prof[str(l)]['dense_flops'] for l in (8,64,256))*512
            arms.append(dict(architecture=recipe,seed=seed,trainable_parameters=98495,deployed_parameters=98495,nontrainable_state_bytes=736,training_sample_order_sha256=order[seed],**common,budget=dict(steps=STEPS,examples_seen=STEPS*16,tokens_seen=512*16*(8+64+256),batch_size=16,history_events=256,planned_training_flops=total,max_training_flops=total*1.01,tuning_trials=1,tuning_flops=0,prefit_profile_method='torch.profiler denseforward/backward estimate plus exact ATen operation/shape signature equality; unsupported scalar FLOPs not separately quantified',prefit_profile_training_flops=total)))
    registration=dict(schema_version='tac-osm.parity.v1',study_id='ME102-RECONSTRUCTION-PANEL',study_type='architecture_comparison',registered_before_first_fit=True,real_world_source=dict(name='UCI501 BeijingMultiSite',sha256=data.SHA,license='CC-BY-4.0',observed_real_outcomes=True),literature_gap=dict(unresolved_TAConly_question='Can directly supervised current-state belief, shared-key associative reads and noncancellable FIT-only age floors improve the integrated model under natural missing observations?',what_is_not_being_rediscovered='Denoising reconstruction, variance floors and proper interval scores are established; no SSM or attention novelty or safety claim.',established_work=[dict(paper='Vincent et al2008 denoising autoencoders',url='https://proceedings.mlr.press/v5/vincent08a.html',established_result='reconstruct authentic inputs from corrupted observations'),dict(paper='Gneiting and Raftery2007 proper scoring rules',url='https://doi.org/10.1198/016214506000001437',established_result='evaluate probability and interval forecasts with proper scores')]),arms=arms,recipes=RECIPES,protocol=dict(seeds=list(SEEDS),updates=STEPS,batch=16,contexts=[8,64,256],augmentation='50percent clean; remainingeligibleorigins .65all6/.35singlepollutant blocks4/8/16/32/64clippedhistory, genuinecurrentlabels only; causalages; naturalprimaryunchanged',gates='EVERYseed: gapPM90coverage[.90,.98];gapPM MAE<=1.05ME101same-indexparent;dailyPM MAE<=1.05LOCF;gapNLL<=ME101mean6.234504261016846;gapPMintervalscore<=1.05ME101same-indexparent',fit_attempts=1,max_concurrent_fits=2,cpu_threads=1,stop_rule='exact15fits;alloriginalscommittedbeforeREPORT;noextra fits/changedthresholds;allrecipespublished;2016-2017closed',claim_scope='matched complete-training-recipe development on exposed2015; no architectural superiority to outside models or independent validation',replay_rtol=1e-5,replay_atol=1e-6))
    write(REG,registration);check_manifest(registration,'preflight');print('ME102_PREFIT_PANEL_PASS')

def fit_one(task):
    recipe,seed=task;torch.set_num_threads(1);torch.use_deterministic_algorithms(True);check_manifest(json.loads(REG.read_text()),'preflight')
    d=data.load('fit');m=initialize(d,recipe,seed);stats=json.loads((DOC/'FIT_STATISTICS.json').read_text());rawscale=torch.tensor(stats['raw_scale'])
    plan=np.load(DOC/'FIT_PLANS.npz')[str(seed)];rng=np.random.default_rng(seed+10000);folder=OUT/recipe/f'seed-{seed}';folder.mkdir(parents=True)
    opt=budgeted_optimizer(m,torch.optim.AdamW,lr=.0006,weight_decay=.001,budget_manifest=REG,budget_receipt=folder/'live-budget.json',budget_seed=seed,budget_architecture=recipe)
    start=time.process_time();wall=time.monotonic();losses=[];input_digest=hashlib.sha256();gradients=None;post={}
    for step,origins in enumerate(plan):
        length=(8,64,256)[step%3];x=tensor_batch(d,origins,length,rng)
        for v in x[:4]:input_digest.update(v.numpy().tobytes())
        opt.zero_grad(set_to_none=True);p,_,tr=m(*x[:6],return_trace=True);loss,parts=objective(m,p,tr,x,rawscale,recipe)
        if not torch.isfinite(loss):raise RuntimeError('nonfinite loss; no retry')
        loss.backward()
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()):raise RuntimeError('invalidgradient')
        if step==1:gradients={n:float(p.grad.norm()) for n,p in m.named_parameters()}
        torch.nn.utils.clip_grad_norm_(m.parameters(),1);checked_optimizer_step(opt);losses.append({k:float(v.detach()) for k,v in parts.items()})
        if (step+1)%512==0:print(json.dumps(dict(recipe=recipe,seed=seed,step=step+1,loss=losses[-1],wall=time.monotonic()-wall)),flush=True)
    cpu=time.process_time()-start
    for length in (8,64,256):post[str(length)]=profile(m,tensor_batch(d,plan[0],length,np.random.default_rng(1234)),rawscale,recipe)
    path=folder/'final.pt';sources={p:sha(ROOT/p) for p in SOURCES}
    save_training_checkpoint(dict(model=m.state_dict(),optimizer=opt.state_dict(),seed=seed,recipe=recipe,steps=STEPS,plan=plan,augmentation_rng_initial_seed=seed+10000,augmentation_rng_final=rng.bit_generator.state,torch_rng=torch.get_rng_state(),losses=losses,preprocessing=dict(mean=d['mean'],std=d['std'],rawscale=rawscale),source_files=sources,source_sha256=data.SHA),path)
    row=dict(architecture=recipe,seed=seed,trainable_parameters=sum(p.numel() for p in m.parameters()),deployed_parameters=sum(p.numel() for p in m.parameters()),checkpoint_sha256=sha(path),actual_steps=STEPS,actual_examples=STEPS*16,actual_tokens=512*16*(8+64+256),actual_sample_order_sha256=hashlib.sha256(plan.tobytes()).hexdigest(),augmented_inputs_sha256=input_digest.hexdigest(),optimizer_cpu_seconds=cpu,wall_seconds=time.monotonic()-wall,peak_memory_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,training_flops=sum(post[str(l)]['dense_flops'] for l in (8,64,256))*512,flops_profiler='postfit originalweight torch.profiler estimates + exact ATen graph parity audit; partialhardwareFLOPcoverage',postfit_profiles=post,nonzero_gradient_tensors_step2=sum(v>0 for v in gradients.values()),parameter_tensors=len(gradients))
    write(folder/'FIT_RECEIPT.json',row);write(folder/'GRADIENTS.json',gradients);print(json.dumps({k:v for k,v in row.items() if k!='postfit_profiles'}),flush=True);return row

def fit():
    check_manifest(json.loads(REG.read_text()),'preflight')
    if OUT.exists():raise RuntimeError('originalpanel exists; nooverwrite')
    prior.frozen([REG,DOC/'FIT_PLANS.npz',DOC/'FIT_STATISTICS.json',DOC/'PREFIT_PROFILES.json',ROOT/'.github/workflows/me102-reconstruction-panel.yml',data.ARCHIVE]+[ROOT/p for p in SOURCES]);OUT.mkdir(parents=True)
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:rows=list(pool.map(fit_one,[(r,s) for s in SEEDS for r in RECIPES]))
    for seed in SEEDS:
        group=[r for r in rows if r['seed']==seed]
        if len({r['augmented_inputs_sha256'] for r in group})!=1:raise RuntimeError('actualinputexposureparityfailure')
        for length in (8,64,256):
            if len({r['postfit_profiles'][str(length)]['aten_signature'] for r in group})!=1:raise RuntimeError('postfitoperatorgraphparityfailure')
    write(OUT/'FIT_INDEX.json',dict(models=rows,source_files={p:sha(ROOT/p) for p in SOURCES}))

def budget_audit():
    idx=json.loads((OUT/'FIT_INDEX.json').read_text());prior.frozen([OUT/'FIT_INDEX.json']+[p for r in RECIPES for s in SEEDS for p in (OUT/r/f'seed-{s}').glob('*')])
    for p,h in idx['source_files'].items():
        if sha(ROOT/p)!=h:raise RuntimeError('frozentrainingsourcechanged')
    for row in idx['models']:
        f=OUT/row['architecture']/f"seed-{row['seed']}";validate_runtime_evidence(dict(row,_prefit_registration_sha256=sha(REG),_expected_optimizer_steps=STEPS,runtime_evidence=dict(receipt_path=str(f/'live-budget.json'),receipt_sha256=sha(f/'live-budget.json'),checkpoint_path=str(f/'final.pt'),checkpoint_sha256=sha(f/'final.pt'))))
    return idx

def score(replay=False):
    idx=budget_audit();d=data.load('report');parent_path=ROOT/'research/replay-bundles/ME101/FORECASTS.npz';prior.frozen([parent_path]);parent=np.load(parent_path)
    arrays={};cohorts={'gap':d['gap'],'general':d['general']};results={};rows=[]
    for name,origins in cohorts.items():
        for suffix in ('origins','truth','observed','locf','ridge'):arrays[name+'_'+suffix]=parent[name+'_'+suffix]
        if not np.array_equal(np.asarray(origins),arrays[name+'_origins']):raise RuntimeError('evaluationcohortchanged')
        results[name]=dict(count=len(origins),models=[])
    with torch.no_grad():
        for rec in idx['models']:
            r,s=rec['architecture'],rec['seed'];cp=torch.load(OUT/r/f'seed-{s}'/'final.pt',map_location='cpu',weights_only=False);m=ReconstructionSensorWorld();m.load_state_dict(cp['model']);m.eval();scores={}
            for name,origins in cohorts.items():
                qs=[];loss=0.;count=0
                for i in range(0,len(origins),32):
                    x=[torch.from_numpy(v) for v in data.batch(d,origins[i:i+32],256)];p,_=m(*x[:6]);qs.append(m.raw_quantiles(p).numpy());loss+=float(m.nll(p,x[6],x[7]))*int(x[7].sum());count+=int(x[7].sum())
                q=np.concatenate(qs);arrays[f'{name}_{r}_{s}']=q;v=prior.metrics(q,arrays[name+'_truth'],arrays[name+'_observed']);v['raw_density_nll']=loss/count;scores[name]=v;results[name]['models'].append(dict(recipe=r,seed=s,**v))
            old=prior.metrics(parent[f'gap_{s-1000}_256'],arrays['gap_truth'],arrays['gap_observed']);dailyref=float(np.abs(arrays['general_locf'][:,0]-arrays['general_truth'][:,0]).mean())
            gates=dict(coverage=.90<=scores['gap']['coverage_by_channel'][0]<=.98,gap_point=scores['gap']['pm_mae']<=1.05*old['pm_mae'],daily_point=scores['general']['pm_mae']<=1.05*dailyref,density=scores['gap']['raw_density_nll']<=6.234504261016846,interval_score=scores['gap']['interval_score_by_channel'][0]<=1.05*old['interval_score_by_channel'][0])
            rows.append(dict(recipe=r,seed=s,gates=gates,full_gate=all(gates.values())));print(json.dumps(rows[-1]),flush=True)
    result=dict(status='EXPOSED2015_MATCHED_TRAINING_RECIPE_PANEL',cohorts=results,models=rows,recipe_all_seed_gates={r:all(v['full_gate'] for v in rows if v['recipe']==r) for r in RECIPES},independent_qualification=False,protected_test_scored=False,synthetic_primary=False)
    if replay:
        orig=np.load(OUT/'FORECASTS.npz')
        for k,v in arrays.items():np.testing.assert_allclose(v,orig[k],rtol=1e-5,atol=1e-6)
        expected=json.loads((OUT/'RESULTS.json').read_text())
        for k in cohorts:
            for a,b in zip(result['cohorts'][k]['models'],expected['cohorts'][k]['models']):
                for key in ('pm_mae','mae_by_channel','coverage_by_channel','interval_score_by_channel','raw_density_nll'):np.testing.assert_allclose(a[key],b[key],rtol=1e-5,atol=1e-6)
        if rows!=expected['models'] or result['recipe_all_seed_gates']!=expected['recipe_all_seed_gates']:raise RuntimeError('gate replay mismatch')
        print('ME102_ORIGINAL15_MODEL_REPLAY=PASS')
    else:
        if (OUT/'RESULTS.json').exists():raise RuntimeError('originalscorescannotbeoverwritten')
        np.savez_compressed(OUT/'FORECASTS.npz',**arrays);write(OUT/'RESULTS.json',result)
        post=json.loads(REG.read_text())
        for arm in post['arms']:
            row=next(r for r in idx['models'] if r['architecture']==arm['architecture'] and r['seed']==arm['seed']);arm['measured']=dict(row,evaluation_split_sha256=arm['test_split_sha256'])
        write(DOC/'POSTFIT.json',post);check_manifest(post,'final')
    print(json.dumps(result['recipe_all_seed_gates']))

if __name__=='__main__':
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','fit','score','replay']);a=parser.parse_args();{'prepare':prepare,'fit':fit,'score':score,'replay':lambda:score(True)}[a.mode]()
