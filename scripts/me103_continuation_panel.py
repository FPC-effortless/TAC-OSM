"""ME103 fixed 1,536-update continuation of all ME102 lineages.

This runner never changes the ME102 graph/objective. prepare performs a no-step
profile and writes a new registration; fit continues all 15 lineages once.
"""
from pathlib import Path
import sys, json, hashlib, time, random, argparse, subprocess, resource, copy
from functools import lru_cache
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts import me100_sensor_data as data
from scripts import me102_gap_data as gap
from scripts import me101_belief_sensor as prior
from scripts.me102_reconstruction_panel import RECIPES, SOURCES as ME102_SOURCES, tensor_batch, initialize, objective, profile, digest, sha, write
from scripts.validate_research_parity import check_manifest
from tac_osm.variants.reconstruction_sensor_world import ReconstructionSensorWorld
from tac_osm.parameter_budget import budgeted_optimizer,checked_optimizer_step,save_training_checkpoint,validate_runtime_evidence

OUT=ROOT/'research/replay-bundles/ME103'; DOC=ROOT/'research/mathematical-engineering/me103'; REG=DOC/'FAIR_COMPARISON.json'
PARENT=ROOT/'research/replay-bundles/ME102'; PARENT_DOC=ROOT/'research/mathematical-engineering/me102'
SEEDS=tuple(range(102001,102006)); STEPS=1536; TOTAL=3072; BATCH=16
SOURCES=list(ME102_SOURCES)+['scripts/me103_continuation_panel.py']

@lru_cache(maxsize=1)
def fit_data():
    """Load immutable FIT records once; augmentation works on derived tensors."""
    return data.load('fit')

def next_plan(seed, d):
    """Exact next 1,536 ME102 plan draws, after replaying its first 1,536."""
    rng=np.random.default_rng(seed); rows=[]
    for _ in range(TOTAL):
        rows.append([d['gap'][i] for i in rng.integers(len(d['gap']),size=8)]+[d['general'][i] for i in rng.integers(len(d['general']),size=8)])
    full=np.asarray(rows,dtype=np.int64)
    old=np.load(PARENT_DOC/'FIT_PLANS.npz')[str(seed)]
    if not np.array_equal(full[:STEPS],old): raise RuntimeError(f'ME102 stored plan mismatch for seed {seed}')
    return full[STEPS:]

def parent_paths(recipe,seed):
    base=PARENT/recipe/f'seed-{seed}'
    return base/'final.pt',base/'live-budget.json',base/'FIT_RECEIPT.json'

def verify_parent(recipe,seed):
    cp,guard,receipt=parent_paths(recipe,seed)
    index=json.loads((PARENT/'FIT_INDEX.json').read_text())
    row=next(x for x in index['models'] if x['architecture']==recipe and x['seed']==seed)
    if sha(cp)!=row['checkpoint_sha256']: raise RuntimeError('parent checkpoint SHA mismatch')
    if sha(cp)!=json.loads(guard.read_text()).get('checkpoints',[{}])[-1].get('sha256'): raise RuntimeError('parent runtime receipt checkpoint binding mismatch')
    c=torch.load(cp,map_location='cpu',weights_only=False)
    if c['steps']!=STEPS or c['seed']!=seed or c['recipe']!=recipe: raise RuntimeError('unexpected ME102 lineage metadata')
    if c['source_sha256']!=data.SHA: raise RuntimeError('ME102 dataset source hash changed')
    if row.get('actual_steps')!=STEPS or hashlib.sha256(c['plan'].tobytes()).hexdigest()!=row.get('actual_sample_order_sha256'): raise RuntimeError('ME102 fit receipt/plan lineage mismatch')
    stats=json.loads((PARENT_DOC/'FIT_STATISTICS.json').read_text())
    for got,want in ((c['preprocessing']['mean'],fit_data()['mean']),(c['preprocessing']['std'],fit_data()['std']),(c['preprocessing']['rawscale'],stats['raw_scale'])):
        if not np.array_equal(np.asarray(got),np.asarray(want)): raise RuntimeError('ME102 preprocessing state mismatch')
    if c['source_files']!={p:sha(ROOT/p) for p in ME102_SOURCES}: raise RuntimeError('ME102 source bytes changed')
    if json.loads(guard.read_text()).get('successful_optimizer_steps')!=STEPS: raise RuntimeError('parent optimizer guard count is not 1536')
    steps={int(v['step']) for v in c['optimizer']['state'].values()}
    if steps!={STEPS} or len(c['optimizer']['state'])!=29: raise RuntimeError('parent optimizer state is not 29/29 at step 1536')
    return cp,c,row

def prepare():
    if REG.exists() or OUT.exists(): raise RuntimeError('ME103 artifacts already exist; no overwrite')
    d=fit_data(); plans={str(s):next_plan(s,d) for s in SEEDS}
    DOC.mkdir(parents=True,exist_ok=True); np.savez_compressed(DOC/'CONTINUATION_PLANS.npz',**plans)
    order={s:hashlib.sha256(plans[str(s)].tobytes()).hexdigest() for s in SEEDS}
    parent_rows=[]; parent_hashes={}
    for recipe in RECIPES:
        for seed in SEEDS:
            cp,c,row=verify_parent(recipe,seed);parent_rows.append((recipe,seed,row));parent_hashes[f'{recipe}/{seed}']={'checkpoint_sha256':sha(cp),'runtime_receipt_sha256':sha(parent_paths(recipe,seed)[1]),'fit_receipt_sha256':sha(parent_paths(recipe,seed)[2])}
    stats=json.loads((PARENT_DOC/'FIT_STATISTICS.json').read_text());rawscale=torch.tensor(stats['raw_scale'])
    restore_rows=[]; augmentation_by_seed={}
    profiles={}
    for recipe in RECIPES:
        profiles[recipe]={}
        cpdata=verify_parent(recipe,SEEDS[0])[1]
        m=initialize(d,recipe,SEEDS[0]);m.load_state_dict(cpdata['model'])
        for length in (8,64,256):
            x=tensor_batch(d,plans[str(SEEDS[0])][0],length,np.random.default_rng(1234))
            profiles[recipe][str(length)]=profile(m,x,rawscale,recipe)
    # Verify each exact restored update input twice without calling optimizer.step.
    # This authenticates model/RNG/sampler restoration before the new registration.
    for recipe in RECIPES:
        for seed in SEEDS:
            cp,parent,row=verify_parent(recipe,seed)
            state_digest=digest(parent['augmentation_rng_final'])
            prior_state=augmentation_by_seed.setdefault(seed,state_digest)
            if prior_state!=state_digest: raise RuntimeError('same-seed parent augmentation RNG states differ across recipes')
            model=initialize(d,recipe,seed); model.load_state_dict(parent['model'])
            snap=[]
            for _ in range(2):
                torch.set_rng_state(parent['torch_rng'].clone())
                rng=np.random.default_rng();rng.bit_generator.state=copy.deepcopy(parent['augmentation_rng_final'])
                x=tensor_batch(d,plans[str(seed)][0],(8,64,256)[STEPS%3],rng)
                model.zero_grad(set_to_none=True);p,_,tr=model(*x[:6],return_trace=True);loss,parts=objective(model,p,tr,x,rawscale,recipe);loss.backward()
                grads={n:hashlib.sha256(v.grad.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for n,v in model.named_parameters()}
                snap.append(dict(input_sha256=hashlib.sha256(b''.join(v.numpy().tobytes() for v in x[:4])).hexdigest(),loss=float(loss.detach()),parts={k:float(v.detach()) for k,v in parts.items()},gradient_sha256=digest(grads)))
            if snap[0]!=snap[1]: raise RuntimeError('restored update forward/loss/gradient replay mismatch')
            restore_rows.append(dict(recipe=recipe,seed=seed,checkpoint_sha256=sha(cp),optimizer_steps=STEPS,restored_update=snap[0],no_optimizer_step=True))
    write(DOC/'RESTORE_PREFIT.json',dict(models=restore_rows,all15=True,no_optimizer_updates=True))
    for length in (8,64,256):
        rows=[profiles[r][str(length)] for r in RECIPES]
        if len({x['aten_signature'] for x in rows})!=1 or len({x['dense_flops'] for x in rows})!=1: raise RuntimeError('continued graph/profile parity failed')
    write(DOC/'PREFIT_PROFILES.json',dict(profiles=profiles,no_optimizer_updates=True,weights='actual ME102 final checkpoints',method='same ME102 profile helper; CPU forward/backward only'))
    parent=json.loads((PARENT_DOC/'FAIR_COMPARISON.json').read_text())
    arms=[]
    for old in parent['arms']:
        arm=dict(old); arm['training_sample_order_sha256']=order[arm['seed']]
        budget=dict(arm['budget']);budget.update(steps=STEPS,examples_seen=STEPS*BATCH,tokens_seen=512*BATCH*(8+64+256),planned_training_flops=sum(profiles[arm['architecture']][str(l)]['dense_flops'] for l in (8,64,256))*512,max_training_flops=sum(profiles[arm['architecture']][str(l)]['dense_flops'] for l in (8,64,256))*512*1.01,prefit_profile_training_flops=sum(profiles[arm['architecture']][str(l)]['dense_flops'] for l in (8,64,256))*512)
        arm['budget']=budget; arms.append(arm)
    manifest=dict(parent);manifest.update(study_id='ME103-ME102-CONTINUATION',registered_before_first_fit=True,arms=arms)
    manifest['literature_gap']=dict(unresolved_TAConly_question='For these same 15 fixed ME102 lineages, does one preregistered equal 1536-update continuation change the unchanged all-seed full-gate outcomes, consistent with update-budget sensitivity?',what_is_not_being_rediscovered='Denoising reconstruction and proper scoring rules are established mechanisms reused unchanged. This exploratory continuation does not introduce a new architecture or establish undertraining generally.',established_work=[dict(paper='Vincent et al. 2008 denoising autoencoders',url='https://doi.org/10.1145/1390156.1390294',established_result='reconstruct authentic inputs from corrupted observations'),dict(paper='Gneiting and Raftery 2007 proper scoring rules',url='https://doi.org/10.1198/016214506000001437',established_result='evaluate probability and interval forecasts with proper scores')])
    manifest['protocol']=dict(parent['protocol'],updates=STEPS,cumulative_updates=TOTAL,continuation_of='ME102-RECONSTRUCTION-PANEL',lineage='restore all 15 exact parent weights, optimizer moments, augmentation RNG and torch RNG; one fixed 1536-step continuation for every lineage',no_architecture_changes=True,stop_rule='exactly 15 fixed continuations; no interim scoring, replacement, retry, tuning or further fit; cumulative 3072 optimizer updates per lineage',claim_scope='exploratory update-budget sensitivity for the unchanged ME102 graph on exposed 2015 cohorts; no novelty or general undertraining claim')
    manifest['parent_artifacts']=parent_hashes
    manifest['parent_registration_sha256']=sha(PARENT_DOC/'FAIR_COMPARISON.json')
    manifest['parent_fit_index_sha256']=sha(PARENT/'FIT_INDEX.json')
    manifest['restore_prefit_sha256']=sha(DOC/'RESTORE_PREFIT.json')
    manifest['source_files']={p:sha(ROOT/p) for p in SOURCES}
    write(REG,manifest);check_manifest(manifest,'preflight')
    print('ME103_PREFIT_PANEL_PASS')

def fit_one(task):
    recipe,seed=task;torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    manifest=json.loads(REG.read_text());check_manifest(manifest,'preflight')
    d=fit_data();cp,parent,_=verify_parent(recipe,seed);m=initialize(d,recipe,seed);m.load_state_dict(parent['model'])
    folder=OUT/recipe/f'seed-{seed}';folder.mkdir(parents=True)
    opt=budgeted_optimizer(m,torch.optim.AdamW,lr=.0006,weight_decay=.001,budget_manifest=REG,budget_receipt=folder/'live-budget.json',budget_seed=seed,budget_architecture=recipe)
    opt.load_state_dict(parent['optimizer'])
    # loading state_dict restores only native optimizer state; new live guard remains at zero steps.
    if opt._tac_parameter_budget_guard.steps!=0: raise RuntimeError('continuation guard inherited parent step count')
    rng=np.random.default_rng();rng.bit_generator.state=parent['augmentation_rng_final']; torch.set_rng_state(parent['torch_rng'])
    plan=np.load(DOC/'CONTINUATION_PLANS.npz')[str(seed)]
    if plan.shape!=(STEPS,BATCH,2): raise RuntimeError('continuation plan shape mismatch')
    rawscale=torch.tensor(parent['preprocessing']['rawscale']);input_digest=hashlib.sha256();losses=[];start=time.process_time();wall=time.monotonic()
    for step,origins in enumerate(plan):
        length=(8,64,256)[(STEPS+step)%3];x=tensor_batch(d,origins,length,rng)
        for v in x[:4]:input_digest.update(v.numpy().tobytes())
        opt.zero_grad(set_to_none=True);p,_,tr=m(*x[:6],return_trace=True);loss,parts=objective(m,p,tr,x,rawscale,recipe)
        if not torch.isfinite(loss): raise RuntimeError('nonfinite loss; no retry')
        loss.backward()
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()): raise RuntimeError('invalid gradient')
        torch.nn.utils.clip_grad_norm_(m.parameters(),1);checked_optimizer_step(opt);losses.append({k:float(v.detach()) for k,v in parts.items()})
        if (step+1)%512==0: print(json.dumps(dict(recipe=recipe,seed=seed,additional_step=step+1,cumulative_step=STEPS+step+1)),flush=True)
    cpu=time.process_time()-start; parent_optimizer_steps=STEPS
    if {int(v['step']) for v in opt.state.values()}!={TOTAL}: raise RuntimeError('optimizer moments did not advance from 1536 to 3072')
    post={}; profile_inputs={}
    for length in (8,64,256):
        px=tensor_batch(d,plan[0],length,np.random.default_rng(1234));profile_inputs[str(length)]=hashlib.sha256(b''.join(v.numpy().tobytes() for v in px[:4])).hexdigest();post[str(length)]=profile(m,px,rawscale,recipe)
    sources={p:sha(ROOT/p) for p in SOURCES}
    path=folder/'final.pt';save_training_checkpoint(dict(model=m.state_dict(),optimizer=opt.state_dict(),seed=seed,recipe=recipe,steps=TOTAL,additional_steps=STEPS,parent_checkpoint_sha256=sha(cp),parent_optimizer_steps=parent_optimizer_steps,continuation_plan=plan,augmentation_rng_initial_state=parent['augmentation_rng_final'],augmentation_rng_final=rng.bit_generator.state,torch_rng=torch.get_rng_state(),losses=losses,preprocessing=parent['preprocessing'],source_files=sources,source_sha256=data.SHA),path)
    row=dict(architecture=recipe,seed=seed,trainable_parameters=sum(p.numel() for p in m.parameters()),deployed_parameters=sum(p.numel() for p in m.parameters()),checkpoint_sha256=sha(path),actual_steps=STEPS,additional_optimizer_steps=STEPS,actual_examples=STEPS*BATCH,actual_tokens=512*BATCH*(8+64+256),actual_sample_order_sha256=hashlib.sha256(plan.tobytes()).hexdigest(),augmented_inputs_sha256=input_digest.hexdigest(),parent_checkpoint_sha256=sha(cp),optimizer_parent_steps=parent_optimizer_steps,optimizer_final_steps=TOTAL,cumulative_optimizer_steps=TOTAL,profile_inputs_sha256=profile_inputs,training_flops=sum(post[str(l)]['dense_flops'] for l in (8,64,256))*512,flops_profiler='ME102 torch.profiler dense forward/backward estimate; unsupported pointwise FLOPs are partial; exact ATen signature audited',optimizer_cpu_seconds=cpu,wall_seconds=time.monotonic()-wall,peak_memory_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,postfit_profiles=post)
    write(folder/'FIT_RECEIPT.json',row);return row

def fit():
    if OUT.exists():raise RuntimeError('ME103 output exists; no overwrite')
    registration=json.loads(REG.read_text());check_manifest(registration,'preflight')
    if registration.get('source_files')!={p:sha(ROOT/p) for p in SOURCES}: raise RuntimeError('ME103 registered source hashes changed before first update')
    # Match every original parent artifact to the exact hashes registered at prepare.
    if registration.get('parent_registration_sha256')!=sha(PARENT_DOC/'FAIR_COMPARISON.json') or registration.get('parent_fit_index_sha256')!=sha(PARENT/'FIT_INDEX.json'):
        raise RuntimeError('ME102 registration/index changed after ME103 prepare')
    if registration.get('restore_prefit_sha256')!=sha(DOC/'RESTORE_PREFIT.json'):
        raise RuntimeError('ME103 restored-state prefit receipt changed')
    for recipe in RECIPES:
        for seed in SEEDS:
            cp,guard,receipt=parent_paths(recipe,seed); key=f'{recipe}/{seed}'; expected=registration.get('parent_artifacts',{}).get(key,{})
            actual={'checkpoint_sha256':sha(cp),'runtime_receipt_sha256':sha(guard),'fit_receipt_sha256':sha(receipt)}
            if actual!=expected: raise RuntimeError(f'ME102 parent artifacts changed after prepare: {key}')
    # Hash the immutable full ME102 panel, including checkpoints and receipts, before first fit.
    frozen=[PARENT/'FIT_INDEX.json']+[p for r in RECIPES for s in SEEDS for p in (PARENT/r/f'seed-{s}').glob('*')]
    prior.frozen(frozen+[REG,ROOT/'.github/workflows/me102-reconstruction-panel.yml',ROOT/'.github/workflows/me103-continuation-panel.yml',DOC/'PROTOCOL.md',DOC/'MATHEMATICS.md',DOC/'CONTINUATION_PLANS.npz',DOC/'PREFIT_PROFILES.json',DOC/'RESTORE_PREFIT.json']+[ROOT/p for p in SOURCES])
    OUT.mkdir(parents=True)
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool: rows=list(pool.map(fit_one,[(r,s) for s in SEEDS for r in RECIPES]))
    for seed in SEEDS:
        group=[r for r in rows if r['seed']==seed]
        if len(group)!=len(RECIPES) or len({r['augmented_inputs_sha256'] for r in group})!=1: raise RuntimeError('continued actual exposure parity failure')
        for length in ('8','64','256'):
            profs=[next(r for r in group if r['architecture']==recipe)['postfit_profiles'][length] for recipe in RECIPES]
            if len({p['aten_signature'] for p in profs})!=1 or len({p['dense_flops'] for p in profs})!=1: raise RuntimeError('postfit graph/profile parity failure')
            inputs=[next(r for r in group if r['architecture']==recipe)['profile_inputs_sha256'][length] for recipe in RECIPES]
            if len(set(inputs))!=1: raise RuntimeError('postfit profile input parity failure')
    write(OUT/'FIT_INDEX.json',dict(models=rows,source_files={p:sha(ROOT/p) for p in SOURCES},parent_artifacts={f'{r}/{s}':sha(parent_paths(r,s)[0]) for r in RECIPES for s in SEEDS}))

def budget_audit():
    index_path=OUT/'FIT_INDEX.json';idx=json.loads(index_path.read_text())
    expected={(r,s) for r in RECIPES for s in SEEDS}
    actual={(row.get('architecture'),row.get('seed')) for row in idx.get('models',[])}
    if len(idx.get('models',[]))!=15 or actual!=expected: raise RuntimeError('ME103 index must contain exactly one row for each of all 15 lineages')
    artifacts=[index_path]
    for recipe in RECIPES:
        for seed in SEEDS:
            folder=OUT/recipe/f'seed-{seed}';artifacts.extend([folder/p for p in ('final.pt','live-budget.json','FIT_RECEIPT.json')])
    prior.frozen(artifacts)
    for p,h in idx['source_files'].items():
        if sha(ROOT/p)!=h:raise RuntimeError('ME103 source changed')
    for row in idx['models']:
        f=OUT/row['architecture']/f"seed-{row['seed']}"; cp=f/'final.pt'
        validate_runtime_evidence(dict(row,_prefit_registration_sha256=sha(REG),_expected_optimizer_steps=STEPS,runtime_evidence=dict(receipt_path=str(f/'live-budget.json'),receipt_sha256=sha(f/'live-budget.json'),checkpoint_path=str(cp),checkpoint_sha256=sha(cp))))
        saved=torch.load(cp,map_location='cpu',weights_only=False)
        if saved.get('steps')!=TOTAL or saved.get('additional_steps')!=STEPS or saved.get('parent_optimizer_steps')!=STEPS: raise RuntimeError('continued checkpoint lineage step counts differ')
        steps={int(v['step']) for v in saved['optimizer']['state'].values()}
        if steps!={TOTAL}: raise RuntimeError('continued checkpoint optimizer states are not all at 3072')
        if sha(parent_paths(row['architecture'],row['seed'])[0])!=saved.get('parent_checkpoint_sha256'): raise RuntimeError('continued checkpoint does not name exact parent')
    return idx

def score(replay=False):
    idx=budget_audit();d=data.load('report');parent_path=ROOT/'research/replay-bundles/ME101/FORECASTS.npz';prior.frozen([parent_path]);parent=np.load(parent_path)
    arrays={};cohorts={'gap':d['gap'],'general':d['general']};results={}
    for name,origins in cohorts.items():
        for suffix in ('origins','truth','observed','locf','ridge'):arrays[name+'_'+suffix]=parent[name+'_'+suffix]
        if not np.array_equal(np.asarray(origins),arrays[name+'_origins']):raise RuntimeError('evaluation cohort changed')
        results[name]=dict(count=len(origins),models=[])
    rows=[]
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
            rows.append(dict(recipe=r,seed=s,gates=gates,full_gate=all(gates.values())))
    result=dict(status='ME103_FIXED_CONTINUATION_EXPOSED2015',cohorts=results,models=rows,recipe_all_seed_gates={r:all(v['full_gate'] for v in rows if v['recipe']==r) for r in RECIPES},independent_qualification=False,protected_test_scored=False,synthetic_primary=False)
    if replay:
        orig=np.load(OUT/'FORECASTS.npz')
        for k,v in arrays.items():np.testing.assert_allclose(v,orig[k],rtol=1e-5,atol=1e-6)
        expected=json.loads((OUT/'RESULTS.json').read_text())
        if rows!=expected['models'] or result['recipe_all_seed_gates']!=expected['recipe_all_seed_gates']:raise RuntimeError('ME103 gate replay mismatch')
        for cohort in cohorts:
            got=result['cohorts'][cohort]['models'];want=expected['cohorts'][cohort]['models']
            if len(got)!=len(want): raise RuntimeError('ME103 score replay model count mismatch')
            for a,b in zip(got,want):
                if (a['recipe'],a['seed'])!=(b['recipe'],b['seed']): raise RuntimeError('ME103 score replay identity mismatch')
                for key,value in a.items():
                    if key in ('recipe','seed'): continue
                    left=np.asarray(value);right=np.asarray(b[key])
                    if left.dtype.kind in 'biuf' and right.dtype.kind in 'biuf': np.testing.assert_allclose(left,right,rtol=1e-5,atol=1e-6)
                    elif value!=b[key]: raise RuntimeError(f'ME103 score replay mismatch for {key}')
    else:
        post_path=DOC/'POSTFIT.json'
        if (OUT/'RESULTS.json').exists() or (OUT/'FORECASTS.npz').exists() or post_path.exists():raise RuntimeError('ME103 score artifacts cannot be overwritten')
        post=json.loads(REG.read_text())
        for arm in post['arms']:
            row=next(r for r in idx['models'] if r['architecture']==arm['architecture'] and r['seed']==arm['seed'])
            arm['measured']=dict(row,evaluation_split_sha256=arm['test_split_sha256'])
        check_manifest(post,'final')
        np.savez_compressed(OUT/'FORECASTS.npz',**arrays);write(OUT/'RESULTS.json',result);write(post_path,post)
    print(json.dumps(result['recipe_all_seed_gates']))

if __name__=='__main__':
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','fit','score','replay']);a=parser.parse_args();{'prepare':prepare,'fit':fit,'score':score,'replay':lambda:score(True)}[a.mode]()
