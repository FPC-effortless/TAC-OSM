"""Freeze all recipes, prediction boundaries and independent evidence limits."""
import copy,json,subprocess,sys
from .common import ROOT,OLD,ARMS,SEEDS,EXPECTED,build,digest,source_paths,verify_data,internal_gate,require
def main():
    data,split=verify_data();base=json.loads((OLD/'registrations/combined/FAIR_COMPARISON.json').read_text())
    sources={p:digest(v) for p,v in source_paths().items()};gates={};plans={}
    for arm in ARMS:
        m=copy.deepcopy(base)
        for field in ('reused_controls','repair','training_source_hashes'):m.pop(field,None)
        m.update(study_id='COLLECT016-'+arm.upper(),source_hashes=sources,
            model_config=dict(width=16,ffn_dim=78 if arm=='current_only' else 57,depth=4,phase='current_value_only' if arm=='current_only' else 'read_after',balance=.01),
            models=[dict(architecture=arm,seed=s,trainable_parameters=EXPECTED[arm],deployed_parameters=EXPECTED[arm]) for s in SEEDS],
            causal_protocol='Observed byte prefix only; full state carried across 128-byte chunks of a 384-byte episode. No detach except declared detach_128 arm. Source-copy answers are supervised targets, never write gates/read inputs before their prediction.',
            scientific_status='INTERNAL_SOURCE_GROUNDED_CREDIT_ASSIGNMENT_PILOT_NO_ARCHITECTURE_SUPERIORITY',
            comparison_gate='OPEN full FLOPs, independent qualification and larger resource-matched competitors; all new arms match examples/optimizer budget and active parameters within 1%.',
            evaluation_protocol_sha256=digest(ROOT/'EVALUATION_PROTOCOL.json'),
            experiment_protocol_sha256=digest(ROOT/'EXPERIMENT_PROTOCOL.json'),
            mathematical_status='Python whole graph numerical/surrogate-gradient fixtures; Lean definitions partial, full Lean/IEEE/AD OPEN')
        m['real_world_source']['name']='Pinned Tiny Shakespeare; verbatim source quotations plus explicitly generated copy-query framing'
        m['real_world_source']['augmentation']='All arms receive identical constructed temporary-key copy tasks; these are mechanism supervision, not real-world facts or qualified primary intelligence evidence.'
        m['fit']=dict(steps=768,batch_size=2,sequence_length=384,chunk_size=128,full_bptt=arm!='detach_128',
            learning_rate=.0003,betas=[.9,.95],weight_decay=.1,clip_norm=1.,optimizer='AdamW',precision='float32',schedule='constant',
            seed_ids=list(SEEDS),checkpoint_steps=[0,256,512,768],worker_seconds_cap=2400,execution_backend='github_actions_cpu',allocation_gpu_minutes=0,
            natural_coefficient=1.,task_lm_coefficient=.25,answer_coefficient=1.,balance_coefficient=.01,answer_length=4,dev_windows=64)
        folder=ROOT/'registrations'/arm;folder.mkdir(parents=True,exist_ok=True)
        (folder/'FAIR_COMPARISON.json').write_text(json.dumps(m,indent=2));gates[arm]=internal_gate(m)
        require(sum(p.numel() for p in build(arm,SEEDS[0]).parameters())==EXPECTED[arm],'actual count differs')
        alias=f'research/mathematical-engineering/collect016_{arm}/FAIR_COMPARISON.json';target=ROOT/'policy'/alias;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(m,indent=2))
        prop=json.loads((OLD/'registrations/combined/PHONE_PROPOSAL.json').read_text());prop.update(run_id='COLLECT016-'+arm.upper(),seeds=list(SEEDS),study_manifest=alias)
        dest=ROOT/'policy/research/phone_queue'/f'collect016_{arm}.json';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(prop,indent=2));(folder/'PHONE_PROPOSAL.json').write_text(json.dumps(prop,indent=2))
        result=subprocess.run([sys.executable,str(ROOT/'policy/scripts/phone_lab.py'),'plan',str(dest.relative_to(ROOT/'policy'))],cwd=ROOT/'policy',capture_output=True,text=True,check=True)
        plans[arm]=json.loads(result.stdout);require(plans[arm]['state']=='READY_INTERNAL_RESEARCH_PLAN' and plans[arm]['backend']=='github_actions','CPU plan rejected')
    smoke=copy.deepcopy(m);smoke.update(study_id='COLLECT016-OPTIMIZER-SMOKE',models=[dict(architecture=a,seed=16000,trainable_parameters=EXPECTED[a],deployed_parameters=EXPECTED[a]) for a in ARMS]);smoke['fit'].update(steps=8,checkpoint_steps=[0,8],seed_ids=[16000],worker_seconds_cap=300)
    (ROOT/'SMOKE_REGISTRATION.json').write_text(json.dumps(smoke,indent=2));gates['smoke']=internal_gate(smoke)
    (ROOT/'results/PREFLIGHT.json').write_text(json.dumps(dict(gates=gates,plans=plans),indent=2));print(json.dumps(dict(gates=gates,plans=plans),indent=2))
if __name__=='__main__':main()
