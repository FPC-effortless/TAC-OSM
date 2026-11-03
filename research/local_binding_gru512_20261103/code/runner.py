"""Scientific runner. Requires a separately reviewed bounded launcher and grant.

Importing this module never imports Torch or constructs a model. No automatic
execution entry point is supplied. Root must release one stage/regime matrix.
"""
import dataclasses,hashlib,json,math,time
from collections import defaultdict
from pathlib import Path
import core,fixtures,admission
COUNTS={}
UPDATES={}
ATTEMPTED_UPDATES={}

def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path,value):
    with path.open('x') as f:f.write(json.dumps(value,indent=2,allow_nan=False)+'\n')

def clone_state(torch,v):
    if isinstance(v,torch.Tensor):return v.detach().clone()
    if dataclasses.is_dataclass(v):return type(v)(**{f.name:clone_state(torch,getattr(v,f.name)) for f in dataclasses.fields(v)})
    if isinstance(v,tuple):return tuple(clone_state(torch,x) for x in v)
    if isinstance(v,list):return [clone_state(torch,x) for x in v]
    if isinstance(v,dict):return {k:clone_state(torch,x) for k,x in v.items()}
    return v

def state_hash(torch,v):
    def pack(x):
        if isinstance(x,torch.Tensor):return {'tensor':core.tensor_hash(torch,x)}
        if dataclasses.is_dataclass(x):return {'type':type(x).__name__,'fields':{f.name:pack(getattr(x,f.name)) for f in dataclasses.fields(x)}}
        if isinstance(x,(list,tuple)):return [pack(y) for y in x]
        if isinstance(x,dict):return {k:pack(y) for k,y in x.items()}
        return x
    return hashlib.sha256(canonical(pack(v))).hexdigest()

def batch(torch,cases):
    slabs=[fixtures.NATURAL]+[bytes.fromhex(c['slab_hex']) for c in cases]
    v=torch.tensor([list(x) for x in slabs],dtype=torch.long);mask=torch.zeros(4,128,dtype=torch.bool);first=mask.clone()
    for i,c in enumerate(cases,1):p=c['first_position'];mask[i,p:p+4]=True;first[i,p]=True
    return v[:,:-1],v[:,1:],mask,first

def behavioral_summary(stage,spec,evaluations,held,families):
    result={}
    for family in families:
        late=[x for x in evaluations if x['family']==family and x['label']=='training_probe' and x['step'] in spec['late_optimization_gate_steps']]
        assert len(late)==2
        data=held[family];cohorts={}
        for name in sorted({x['cohort'] for x in data}):
            rows=[x for x in data if x['cohort']==name];present=[x for x in rows if x['target_hex']!='3f3f3f3f'];absent=[x for x in rows if x['target_hex']=='3f3f3f3f']
            cohorts[name]={'queries':len(rows),'present_queries':len(present),'absent_queries':len(absent),'present_first_correct':sum(x['first_correct'] for x in present),'present_exact4':sum(x['exact4'] for x in present),'absent_exact4':sum(x['exact4'] for x in absent),'all_present_first':all(x['first_correct'] for x in present) if present else None,'all_present_exact4':all(x['exact4'] for x in present) if present else None,'all_absent_exact4':all(x['exact4'] for x in absent) if absent else None}
        maps=defaultdict(list)
        for row in data:
            if row['target_hex']!='3f3f3f3f':maps[row['group']].append(row)
        pairs={};by={(x['cohort'],x['pair_id'],x['key']):x for x in data}
        for other in ('swapped_values','record_order','query_order'):
            a=[x for x in data if x['cohort']=='base' and x['target_hex']!='3f3f3f3f'];matched=[]
            for row in a:
                right=by[(other,row['pair_id'],row['key'])];should_change=other=='swapped_values'
                assert (row['target_hex']!=right['target_hex'])==should_change
                matched.append(row['exact4'] and right['exact4'] and ((row['prediction_hex']!=right['prediction_hex'])==should_change))
            pairs[other]={'pairs':len(matched),'both_factual_answers_correct_and_expected_change':sum(matched),'all_pairs':all(matched) if matched else None}
        result[family]={'fixed_budget_training_first_gate':all(x['first_correct']==x['count'] for x in late),'fixed_budget_training_exact4_gate':all(x['exact4']==x['count'] for x in late),'late_steps':spec['late_optimization_gate_steps'],'probe_scope':spec['probe_scope'],'cohorts':cohorts,'complete_present_maps':sum(all(x['exact4'] for x in rows) for rows in maps.values()),'present_maps':len(maps),'counterfactual_pairs':pairs,'empty_history_original_first_target_retention':[x.get('original_first_target_retention_not_factual_correctness') for x in data if x['cohort']=='empty_history'],'interpretation':'Finite constructed optimization and compositional held-out gates only. Never pool these into learned reliability.'}
    return result

def run(root,protocol,grant,out,check):
    """Only a source-reviewed resource supervisor may call this function."""
    assert grant['authorized'] is True and grant['stage'] in ('fixed','varying') and grant['regime'] in ('default','larger_epsilon')
    assert grant['source_manifest_sha256']==sha(root/'SOURCE_MANIFEST.json')
    assert grant['code_manifest_sha256']==sha(root/'CODE_MANIFEST.json') and grant['protocol_sha256']==sha(root/'PROTOCOL.json')
    assert grant['bounded_launcher_review_sha256'] and grant['retry_authorized'] is False
    for name,row in json.loads((root/'CODE_MANIFEST.json').read_bytes())['files'].items():assert sha(root/name)==row['sha256']
    admission_receipt=admission.enforce(root,protocol,grant,out)
    import torch
    assert torch.__version__=='2.13.0+cpu' and torch.version.cuda is None
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    global COUNTS,UPDATES,ATTEMPTED_UPDATES
    stage=grant['stage'];regime=grant['regime'];spec=protocol['stages'][stage];eps=protocol['regimes'][regime]['epsilon']
    counts=defaultdict(int,{k:0 for k in spec['expected_counts']});COUNTS=counts
    sm=json.loads((root/'SOURCE_MANIFEST.json').read_bytes())
    for name,row in sm['files'].items():assert sha(root/'source'/name)==row['sha256']
    finder=core.PinnedFinder(root/'source',sm['files']);__import__('sys').meta_path.insert(0,finder)
    from collect018.common import build
    from collect018.model import run_chunks,tensors
    from collect018.reference import whole_reference
    from collect018.train import objective
    def load_fixture(name):
        p=root/'fixtures'/name;assert sha(p)==protocol['fixture_files'][name]['sha256'];return json.loads(p.read_bytes())
    if stage=='fixed':
        cases=load_fixture('fixed_train_cases.json');byid={c['id']:c for c in cases};schedule=load_fixture('fixed_schedule.json');batches=[[byid[i] for i in row] for row in schedule];probe=cases
    else:batches=load_fixture('varying_train_batches.json');probe=sum(batches[:4],[])
    assert len(batches)==spec['updates_per_arm']
    models={};refs={};opts={};initial_hashes={};gate_rows=[];evaluations=[];updates={f:0 for f in protocol['families']};sealed=False;UPDATES=updates;attempted={f:0 for f in protocol['families']};ATTEMPTED_UPDATES=attempted
    def compare(a,b,label,tol):
        row={'label':label,'atol':tol,'rtol':tol,'pass':False}
        if a is None or b is None:row['error']='MISSING_TENSOR'
        else:
            maximum=float((a-b).detach().abs().max()) if a.numel() else 0.
            row['max_abs']=maximum if math.isfinite(maximum) else None;row['nonfinite_difference']=not math.isfinite(maximum)
            row['finite_inputs']=bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())
            if not row['finite_inputs']:row['error']='NONFINITE_INPUT'
            else:
                try:torch.testing.assert_close(a,b,atol=tol,rtol=tol);row['pass']=True
                except AssertionError as e:row['error']=str(e)
        gate_rows.append(row)
        if not row['pass']:
            with (out/'RETAINED_NUMERICAL_FAILURES.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
    def source_forward(m,tokens,**kw):
        z,st=m(tokens,**kw);counts['candidate_forward_calls']+=1;return z,st
    def audit(family,step):
        check(f'audit {family} {step}');m,ref=models[family],refs[family];ref.load_state_dict(m.state_dict());assert core.parameter_hash(torch,m)==core.parameter_hash(torch,ref)
        m.train();ref.train();x,y,mask,first=batch(torch,batches[0]);owners=tuple(f'audit-slot-{i}' for i in range(4))
        with core.capture(torch,m) as (ar,ae):z,st,bal,_=run_chunks(m,x,chunk=128,owners=owners);counts['candidate_forward_calls']+=1
        with core.capture(torch,ref) as (br,be):zz,ss=whole_reference(ref,x,owners=owners);counts['reference_forward_calls']+=1
        bbal=z.new_zeros(()) if family=='gru' else core.reference_balance(torch,br)[0]
        compare(z,zz,f'{family} {step} audit logits',4e-5)
        for n,t in tensors(st).items():compare(t,tensors(ss)[n],f'{family} {step} audit state {n}',4e-5)
        losses={'first_ce':torch.nn.functional.cross_entropy(z[first],y[first]),'joint':objective(z,y,mask,first,bal,False)[0],'first':objective(z,y,mask,first,bal,True)[0]};rl=core.reference_losses(torch,zz,y,mask,first,bbal)
        for k in ('first_ce','joint','first'):
            m.zero_grad(set_to_none=True);ref.zero_grad(set_to_none=True);compare(losses[k],rl[k],f'{family} {step} audit loss {k}',4e-5)
            losses[k].backward(retain_graph=k!='first');counts['candidate_backward_calls']+=1
            rl[k].backward(retain_graph=k!='first');counts['reference_backward_calls']+=1
            for (n,p),(nn,q) in zip(m.named_parameters(),ref.named_parameters()):assert n==nn;compare(p.grad,q.grad,f'{family} {step} audit {k} gradient {n}',4e-4)
        if family!='gru':
            for i in range(4):compare(ar[i].softmax(-1).argmax(-1),br[i].softmax(-1).argmax(-1),f'{family} {step} audit phase route {i}',0.)
        if step==0:
            poison=x.clone()
            for row,c in enumerate(batches[0],1):
                position=c['first_position'];poison[row,position+1:]=(poison[row,position+1:]+137)%256
            # Same training-mode, grad-enabled source path. No backward or update.
            poisoned,_=source_forward(m,poison,owners=owners);counts['future_poison_forward_calls']+=1
            compare(z[0],poisoned[0],f'{family} initial future poison natural row',0.)
            for row,c in enumerate(batches[0],1):
                end=c['first_position']+1;compare(z[row,:end],poisoned[row,:end],f'{family} initial future poison prefix {row}',0.)
        m.zero_grad(set_to_none=True);ref.zero_grad(set_to_none=True)
    def save(family,step):
        path=out/f'{family}_checkpoint_{step:04}.pt';assert not path.exists()
        torch.save({'model':clone_state(torch,models[family].state_dict()),'optimizer':clone_state(torch,opts[family].state_dict()),'rng':torch.get_rng_state().clone(),'step':step,'family':family,'stage':stage,'regime':regime,'source_manifest_sha256':grant['source_manifest_sha256'],'protocol_sha256':grant['protocol_sha256'],'initializer_hash':initial_hashes[family]},path)
        dump(path.with_suffix('.sha256.json'),{'sha256':sha(path)})
        ph=core.parameter_hash(torch,models[family])
        with (out/'CHECKPOINT_PARAMETER_HASHES.jsonl').open('a') as stream:stream.write(json.dumps({'family':family,'step':step,'parameter_sha256':ph})+'\n')
        if step==0:assert ph==protocol['baseline_parameter_hashes']['initial'],'INITIALIZATION_DIFFERS_FROM_APPROVED_BASELINE'
    def decode_batch(m,rows,label):
        """Six forward calls per distinct prefix length, no answer-fed free run."""
        bylength=defaultdict(list);answers={}
        for c in rows:bylength[len(bytes.fromhex(c['prefix_hex']))].append(c)
        with torch.no_grad():
            for length,items in sorted(bylength.items()):
                owners=tuple(f'probe-slot-{i}' for i in range(len(items)));p=torch.tensor([list(bytes.fromhex(c['prefix_hex'])) for c in items]);q=torch.tensor([list(bytes.fromhex(c['query_hex'])) for c in items]);targets=torch.tensor([list(bytes.fromhex(c['target_hex'])) for c in items])
                _,st=source_forward(m,p,owners=owners);original=state_hash(torch,st);z,ss=source_forward(m,q,state=clone_state(torch,st),owners=owners);pred=[];first_logits=z[:,-1].clone()
                for j in range(4):
                    token=z[:,-1].argmax(-1);pred.append(token)
                    if j<3:z,ss=source_forward(m,token[:,None],state=ss,owners=owners)
                teacher,_=source_forward(m,torch.cat((q,targets[:,:3]),1),state=clone_state(torch,st),owners=owners)
                assert state_hash(torch,st)==original
                nll=torch.nn.functional.cross_entropy(teacher[:,-4:].flatten(0,1),targets.flatten(),reduction='none').reshape(len(items),4);pred=torch.stack(pred,1)
                for i,c in enumerate(items):answers[c['id']]={'prediction_hex':bytes(pred[i].tolist()).hex(),'target_hex':c['target_hex'],'first_correct':bool(pred[i,0]==targets[i,0]),'exact4':bool(torch.equal(pred[i],targets[i])),'teacher_forced_mean_nll':float(nll[i].mean()),'first_logits':first_logits[i].tolist()}
        return {'label':label,'rows':answers,'first_correct':sum(x['first_correct'] for x in answers.values()),'exact4':sum(x['exact4'] for x in answers.values()),'count':len(answers)}
    def heldout(m,groups):
        rows=[]
        with torch.no_grad():
            for g in groups:
                check('heldout '+g['id'])
                owners=('heldout-owner',);prefix=bytes.fromhex(g['prefix_hex']);st=None
                if prefix:_,st=source_forward(m,torch.tensor([list(prefix)]),owners=owners)
                original=state_hash(torch,st)
                for q in g['queries']:
                    query=bytes.fromhex(q['query_hex']);target=fixtures.factual_answer(prefix,query);assert target.hex()==q['target_hex'];tokens=torch.tensor([list(query)]);z,ss=source_forward(m,tokens,state=clone_state(torch,st),owners=owners);first=z[0,-1].clone();pred=[]
                    for j in range(4):
                        token=int(z[0,-1].argmax());pred.append(token)
                        if j<3:z,ss=source_forward(m,torch.tensor([[token]]),state=ss,owners=owners)
                    teacher,_=source_forward(m,torch.tensor([list(query+target[:3])]),state=clone_state(torch,st),owners=owners)
                    nll=float(torch.nn.functional.cross_entropy(teacher[0,-4:],torch.tensor(list(target))))
                    original_target=g.get('original_targets',{}).get(q['key'])
                    rows.append({'group':g['id'],'pair_id':g['pair_id'],'cohort':g['cohort'],'key':q['key'],'target_hex':target.hex(),'prediction_hex':bytes(pred).hex(),'first_correct':pred[0]==target[0],'exact4':bytes(pred)==target,'teacher_forced_nll':nll,'first_target_log_probability':float(first.log_softmax(-1)[target[0]]),'original_target_retention_not_factual_correctness':None if original_target is None else bytes(pred).hex()==original_target,'original_first_target_retention_not_factual_correctness':None if original_target is None else pred[0]==bytes.fromhex(original_target)[0]})
                    assert state_hash(torch,st)==original,'QUERY_MUTATED_SHARED_PREFIX'
        return rows
    # Constructors are fixed before optimization. References never get optimizers.
    for family in protocol['families']:
        models[family]=build(family+'_first',protocol['seed']);refs[family]=build(family+'_first',protocol['seed']);counts['model_constructors']+=2
        assert sum(p.numel() for p in models[family].parameters())==protocol['parameter_counts'][family]
        initial_hashes[family]=core.parameter_hash(torch,models[family]);opts[family]=torch.optim.AdamW(models[family].parameters(),lr=.003,betas=(.9,.95),weight_decay=.1,eps=eps,foreach=False);counts['optimizer_constructors']+=1
    if regime=='larger_epsilon':
        # Read only pairing/resource metadata. Do not expose the default run's
        # held-out labels or predictions to this optimizer-regime fit.
        p=Path(grant['paired_default_receipt_path']);assert sha(p)==grant['paired_default_receipt_sha256'];prior=json.loads(p.read_bytes())
        accounting=Path(grant['paired_default_accounting_path']);assert sha(accounting)==grant['paired_default_accounting_sha256'];receipt=json.loads(accounting.read_bytes());assert receipt['status']=='COMPLETE_WITH_CAPTURED_RESOURCE_ACCEPTANCE' and receipt['owned_group_children_reaped']
        assert prior['protocol_sha256']==grant['protocol_sha256'] and prior['code_manifest_sha256']==grant['code_manifest_sha256']
        assert prior['status']=='COMPLETE_FIXED_BUDGET_MATRIX' and prior['stage']==stage and prior['regime']=='default' and prior['initial_hashes']==initial_hashes
    dump(out/'INITIAL_HASHES.json',initial_hashes)
    for family in protocol['families']:audit(family,0);save(family,0)
    assert all(x['pass'] for x in gate_rows),'INITIAL_IMPLEMENTATION_GATE_HOLD'
    import source_probes
    probe_receipt=source_probes.run(torch,models['gru'],out,protocol['source_probes'])
    for family in protocol['families']:
        models[family].eval();evaluations.append({'family':family,'step':0,**decode_batch(models[family],probe,'training_probe')})
    for step,cases in enumerate(batches,1):
        assert not sealed;check(f'train round {step}');x,y,mask,first=batch(torch,cases);owners=tuple(f'exposure-{step}-slot-{i}' for i in range(4));slab_hash=hashlib.sha256(bytes(fixtures.NATURAL)+b''.join(bytes.fromhex(c['slab_hex']) for c in cases)).hexdigest()
        order=protocol['families'][step%3:]+protocol['families'][:step%3]
        for family in order:
            check(f'train {step} {family}')
            m,opt=models[family],opts[family];m.train();opt.zero_grad(set_to_none=True);start=time.process_time();z,_,bal,_=run_chunks(m,x,chunk=128,owners=owners);counts['candidate_forward_calls']+=1
            loss,metrics=objective(z,y,mask,first,bal,True);assert bool(torch.isfinite(loss));loss.backward();counts['candidate_backward_calls']+=1
            assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in m.parameters())
            raw_coordinates={name:float(dict(m.named_parameters())[name].grad[tuple(ix)]) for name,ix in protocol['observed_coordinates'][family].items()}
            gn=torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);assert bool(torch.isfinite(gn))
            n=sum(p.numel() for p in m.parameters());near_common=sum(int((p.grad.abs()<=1e-8).sum()) for p in m.parameters());near_regime=sum(int((p.grad.abs()<=eps).sum()) for p in m.parameters())
            sampled={}
            for name,index in protocol['observed_coordinates'][family].items():
                p=dict(m.named_parameters())[name];ix=tuple(index);sampled[name]={'index':index,'raw_focused_gradient':raw_coordinates[name],'before':float(p[ix].detach()),'clipped_gradient':float(p.grad[ix])}
            attempted[family]+=1;counts['optimizer_attempts']+=1
            (out/'UPDATE_ATTEMPT.json').write_text(json.dumps({'family':family,'round':step,'attempted_updates':attempted,'completed_updates':updates,'outcome_unresolved':True},indent=2)+'\n')
            opt.step();updates[family]+=1;counts['optimizer_updates']+=1;assert all(bool(torch.isfinite(p).all()) for p in m.parameters())
            for name,row in sampled.items():
                p=dict(m.named_parameters())[name];ix=tuple(row['index']);row.update(after=float(p[ix].detach()),exp_avg=float(opt.state[p]['exp_avg'][ix]),exp_avg_sq=float(opt.state[p]['exp_avg_sq'][ix]),optimizer_step=float(opt.state[p]['step']))
            (out/'UPDATE_ATTEMPT.json').write_text(json.dumps({'family':family,'round':step,'attempted_updates':attempted,'completed_updates':updates,'outcome_unresolved':False},indent=2)+'\n')
            row={'family':family,'step':step,'slab_sha256':slab_hash,'owners':owners,'loss':float(loss.detach()),'gradient_norm_before_clip':float(gn),'fraction_clipped_abs_le_1e_minus8':near_common/n,'fraction_clipped_abs_le_regime_epsilon':near_regime/n,'coordinates':sampled,'CPU':time.process_time()-start,**metrics}
            with (out/'TRAINING_JOURNAL.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        if step in spec['checkpoints'][1:]:
            for family in protocol['families']:
                models[family].eval();save(family,step);evaluations.append({'family':family,'step':step,**decode_batch(models[family],probe,'training_probe')})
        (out/'PROGRESS_COUNTS.json').write_text(json.dumps({'round':step,'updates':updates,'counts':dict(counts)},indent=2)+'\n')
    assert all(n==spec['updates_per_arm'] for n in updates.values()) and attempted==updates
    sealed=True;final_hashes={f:core.parameter_hash(torch,m) for f,m in models.items()};dump(out/'TRAINING_SEALED.json',{'updates':updates,'final_hashes':final_hashes,'no_further_optimizer_updates':True})
    for family in protocol['families']:
        audit(family,spec['updates_per_arm']);models[family].eval();permuted=decode_batch(models[family],list(reversed(probe)),'permuted_training_probe');original=next(x for x in evaluations if x['family']==family and x['step']==spec['updates_per_arm'])
        for name,row in permuted['rows'].items():
            compare(torch.tensor(row['first_logits']),torch.tensor(original['rows'][name]['first_logits']),f'{family} final batch permutation logits {name}',4e-5)
            compare(torch.tensor(list(bytes.fromhex(row['prediction_hex']))),torch.tensor(list(bytes.fromhex(original['rows'][name]['prediction_hex']))),f'{family} final batch permutation prediction {name}',0.)
        evaluations.append({'family':family,'step':spec['updates_per_arm'],**permuted})
    # Verify the actual trained objects and newly written final artifacts.
    artifacts={}
    def exact_tree(a,b):
        if isinstance(a,torch.Tensor):assert isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.device==b.device and torch.equal(a,b)
        elif isinstance(a,dict):
            assert isinstance(b,dict) and a.keys()==b.keys()
            for k in a:exact_tree(a[k],b[k])
        elif isinstance(a,(list,tuple)):
            assert type(a)==type(b) and len(a)==len(b)
            for x,y in zip(a,b):exact_tree(x,y)
        else:assert type(a)==type(b) and a==b
    for family in protocol['families']:
        check('final artifact verification '+family);m,opt,restored=models[family],opts[family],refs[family]
        parameters=list(m.parameters());deployed=sum(p.numel() for p in parameters);active=sum(p.numel() for p in parameters if p.requires_grad)
        assert active==deployed==protocol['parameter_counts'][family] and final_hashes[family]!=initial_hashes[family]
        assert set(opt.state)==set(parameters)
        for parameter in parameters:
            state=opt.state[parameter];assert float(state['step'])==spec['updates_per_arm']
            assert state['exp_avg'].shape==parameter.shape and state['exp_avg_sq'].shape==parameter.shape
            assert bool(torch.isfinite(state['exp_avg']).all()) and bool(torch.isfinite(state['exp_avg_sq']).all()) and bool((state['exp_avg_sq']>=0).all())
        path=out/f'{family}_checkpoint_{spec["updates_per_arm"]:04}.pt';checksum=json.loads(path.with_suffix('.sha256.json').read_bytes())['sha256'];assert sha(path)==checksum
        saved=torch.load(path,map_location='cpu',weights_only=True)
        assert saved['step']==spec['updates_per_arm'] and saved['family']==family and saved['stage']==stage and saved['regime']==regime and saved['protocol_sha256']==grant['protocol_sha256'] and saved['source_manifest_sha256']==grant['source_manifest_sha256']
        exact_tree(saved['model'],m.state_dict());exact_tree(saved['optimizer'],opt.state_dict());assert torch.equal(saved['rng'],torch.get_rng_state())
        restored.load_state_dict(saved['model']);restored.eval()
        restored_opt=torch.optim.AdamW(restored.parameters(),lr=.003,betas=(.9,.95),weight_decay=.1,eps=eps,foreach=False);counts['optimizer_constructors']+=1
        restored_opt.load_state_dict(saved['optimizer']);exact_tree(restored_opt.state_dict(),opt.state_dict())
        x,_,_,_=batch(torch,batches[0]);owners=tuple(f'serialization-slot-{i}' for i in range(4))
        with torch.no_grad():
            live,ls=source_forward(m,x,owners=owners);replay,rs=source_forward(restored,x,owners=owners)
        assert torch.equal(live,replay) and state_hash(torch,ls)==state_hash(torch,rs)
        artifacts[family]={'active_parameters':active,'deployed_parameters':deployed,'AdamW_parameter_states':len(opt.state),'every_AdamW_step':spec['updates_per_arm'],'finite_moments':True,'checkpoint_sha256':checksum,'model_optimizer_rng_payload_exact':True,'restored_optimizer_state_exact':True,'restored_optimizer_updates_executed':0,'serialization_logits_state_bitwise_exact':True}
    dump(out/'FINAL_ARTIFACT_VERIFICATION.json',artifacts)
    # This is the first read of held-out byte/label files, after all fits are sealed.
    groups=load_fixture(spec['heldout_file']);held={}
    for family in protocol['families']:
        check(f'heldout {family}');held[family]=heldout(models[family],groups);assert core.parameter_hash(torch,models[family])==final_hashes[family]
    assert dict(counts)==spec['expected_counts'],(dict(counts),spec['expected_counts'])
    for name,row in sm['files'].items():assert sha(root/'source'/name)==row['sha256']
    result={'status':'COMPLETE_FIXED_BUDGET_MATRIX','stage':stage,'regime':regime,'epsilon':eps,'seed':protocol['seed'],'counts':dict(counts),'updates':updates,'attempted_updates':attempted,'artifact_verification':artifacts,'initial_hashes':initial_hashes,'final_hashes':final_hashes,'admission_receipt':admission_receipt,'implementation_gate_pass':all(x['pass'] for x in gate_rows),'comparison_rows':gate_rows,'training_probe':evaluations,'heldout':held,'behavioral_gates':behavioral_summary(stage,spec,evaluations,held,protocol['families']),'source_imports':finder.loaded,'qualification':'Fixed-case optimization and finite synthetic held-out behavior are separate. Single seed, no architecture-superiority or reliable-binding claim. Epsilon is a changed optimizer regime, not a precision fix.'}
    hashes=[json.loads(row) for row in (out/'CHECKPOINT_PARAMETER_HASHES.jsonl').read_text().splitlines()]
    result['source_probe_receipt']=probe_receipt
    result['baseline_128_parameter_hash_matches']=next(row['parameter_sha256'] for row in hashes if row['step']==128)==protocol['baseline_parameter_hashes']['step128']
    result['budget_causal_interpretation_allowed']=result['baseline_128_parameter_hash_matches']
    result['heldout_opened']=False
    result['allocation_id']=protocol['allocation_id']
    dump(out/'PAIRING_RECEIPT.json',{'status':result['status'],'stage':stage,'regime':regime,'initial_hashes':initial_hashes,'updates':updates,'protocol_sha256':grant['protocol_sha256'],'code_manifest_sha256':grant['code_manifest_sha256']})
    return result
