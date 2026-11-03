import copy,json,subprocess,sys
from .common import ROOT,C16,ARMS,SEEDS,EXPECTED,build,digest,source_paths,verify_data,internal_gate,require
def main():
    verify_data();source={k:digest(v) for k,v in source_paths().items()};base=json.loads((C16/'registrations/full_bptt/FAIR_COMPARISON.json').read_text());plans=[]
    for arm in ARMS:
        family=arm.split('_')[0];count=sum(p.numel() for p in build(arm,18000).parameters());require(count==EXPECTED[family],'count differs')
        m=copy.deepcopy(base);m.update(study_id='COLLECT018-'+arm.upper(),scientific_status='INTERNAL_LOCAL_ACQUISITION_PILOT_NO_ARCHITECTURE_SUPERIORITY',source_hashes=source,
            model_config=dict(family=family,objective=arm.split('_')[1],parameters=count,context_source='existing native pathways before phase' if family=='context' else 'original byte phase' if family=='phase' else 'standard GRU32/125'),
            models=[dict(architecture=arm,seed=s,trainable_parameters=count,deployed_parameters=count) for s in SEEDS],
            causal_protocol='Observed byte-prefix only. Original TRAIN source quotes and explicitly constructed copy framing. Full32-byte chunk credit across128 bytes. No role masks or true answer inputs before prediction.',
            comparison_gate='No architecture-superiority eligibility. Source exposure/parameters/optimizer budget matched within this factorial pilot; complete FLOPs/independent qualification OPEN.',
            evaluation_protocol_sha256=digest(ROOT/'EVALUATION_PROTOCOL.json'),experiment_protocol_sha256=digest(ROOT/'EXPERIMENT_PROTOCOL.json'))
        m['real_world_source']['augmentation']='Constructed local first-answer/4-byte copy episodes use verbatim original quotations; mechanism supervision only. Ordinary original-text DEV is separate.'
        m['fit'].update(steps=1024,batch_size=4,sequence_length=128,chunk_size=128,learning_rate=.003,betas=[.9,.95],checkpoint_steps=[0,256,512,1024],seed_ids=list(SEEDS),worker_seconds_cap=2400,
            natural_coefficient=1.,task_lm_coefficient=.25,answer_coefficient=1.,first_extra_coefficient=3. if arm.endswith('_first') else 0.,balance_coefficient=0. if family=='gru' else .01)
        dest=ROOT/'registrations'/arm;dest.mkdir(parents=True,exist_ok=True);path=dest/'FAIR_COMPARISON.json';path.write_text(json.dumps(m,indent=2));gate=internal_gate(m)
        alias='research/mathematical-engineering/collect018_'+arm+'/FAIR_COMPARISON.json';target=ROOT/'policy'/alias;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(path.read_text())
        prop=json.loads((C16/'registrations/full_bptt/PHONE_PROPOSAL.json').read_text());prop.update(run_id='COLLECT018-'+arm.upper(),study_manifest=alias,seeds=list(SEEDS),estimated_cpu_minutes=40)
        p=ROOT/'policy/research/phone_queue'/('collect018_'+arm+'.json');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(prop,indent=2));(dest/'PHONE_PROPOSAL.json').write_text(p.read_text())
        plan=subprocess.run([sys.executable,'scripts/phone_lab.py','plan',str(p.relative_to(ROOT/'policy'))],cwd=ROOT/'policy',capture_output=True,text=True,check=True);decision=json.loads(plan.stdout)
        require(decision['state']=='READY_INTERNAL_RESEARCH_PLAN' and decision['backend']=='github_actions','compute plan blocked');plans.append(dict(arm=arm,gate=gate,plan=decision))
        smoke=copy.deepcopy(m);smoke['study_id']='COLLECT018-SMOKE-'+arm.upper();smoke['models']=[dict(architecture=arm,seed=18000,trainable_parameters=count,deployed_parameters=count)];smoke['fit'].update(steps=8,seed_ids=[18000],checkpoint_steps=[0,8],worker_seconds_cap=300)
        sp=ROOT/'smoke_registrations'/arm;sp.mkdir(parents=True,exist_ok=True);(sp/'FAIR_COMPARISON.json').write_text(json.dumps(smoke,indent=2));internal_gate(smoke)
    require(max(EXPECTED.values())/min(EXPECTED.values())-1<=.01,'cross-arm count parity fails')
    (ROOT/'results/PREFLIGHT.json').write_text(json.dumps(dict(status='PASS_REGISTERED_INTERNAL_PREFLIGHT',plans=plans,optimizer_updates=0),indent=2));print(json.dumps(dict(arms=6,models=18,counts=EXPECTED,plan='READY_INTERNAL_RESEARCH_PLAN'),indent=2))
if __name__=='__main__':main()
