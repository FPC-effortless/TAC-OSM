import copy,json,subprocess,sys
from .common import ROOT,C16,SEEDS,build,digest,source_paths,verify_data,internal_gate,require
def main():
    verify_data();m=json.loads((C16/'registrations/full_bptt/FAIR_COMPARISON.json').read_text());m.update(study_id='COLLECT017-ATTENTION-SANITY',source_hashes={k:digest(v) for k,v in source_paths().items()},
        model_config=dict(width=16,ffn_dim=38,depth=4,window=1536,attention='causal complete registered prefix; native fast/slow/rotary/SSA retained'),
        models=[dict(architecture='attention',seed=s,trainable_parameters=99875,deployed_parameters=99875) for s in SEEDS],
        scientific_status='ADDITIONAL_POST_NULL_LEARNABILITY_CONTROL_NOT_ORIGINAL_PRIMARY_COMPARISON',
        causal_protocol='Byte-prefix causal attention; connected 384-byte BPTT; KV window 1536 covers all registered histories and generations. Exact same source copy/LM training batches as COLLECT016. No phase routing/balance loss exists.',
        evaluation_protocol_sha256=digest(ROOT/'EVALUATION_PROTOCOL.json'),experiment_protocol_sha256=digest(ROOT/'EXPERIMENT_PROTOCOL.json'),
        comparison_gate='OPEN full FLOPs and independently qualified architecture comparison; additional baseline is explicitly selected after initial COLLECT016 null results.')
    m['fit'].update(full_bptt=True,balance_coefficient=0.)
    dest=ROOT/'registrations/attention';dest.mkdir(parents=True,exist_ok=True);(dest/'FAIR_COMPARISON.json').write_text(json.dumps(m,indent=2));gate=internal_gate(m)
    alias='research/mathematical-engineering/collect017_attention/FAIR_COMPARISON.json';target=ROOT/'policy'/alias;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(m,indent=2))
    prop=json.loads((C16/'registrations/full_bptt/PHONE_PROPOSAL.json').read_text());prop.update(run_id='COLLECT017-ATTENTION-SANITY',study_manifest=alias);p=ROOT/'policy/research/phone_queue/collect017_attention.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(prop,indent=2));(dest/'PHONE_PROPOSAL.json').write_text(json.dumps(prop,indent=2))
    plan=subprocess.run([sys.executable,'scripts/phone_lab.py','plan',str(p.relative_to(ROOT/'policy'))],cwd=ROOT/'policy',capture_output=True,text=True,check=True);decision=json.loads(plan.stdout);require(decision['state']=='READY_INTERNAL_RESEARCH_PLAN','plan blocked')
    smoke=copy.deepcopy(m);smoke.update(study_id='COLLECT017-OPTIMIZER-SMOKE',models=[dict(architecture='attention',seed=16000,trainable_parameters=99875,deployed_parameters=99875)]);smoke['fit'].update(steps=8,seed_ids=[16000],checkpoint_steps=[0,8],worker_seconds_cap=300)
    (ROOT/'SMOKE_REGISTRATION.json').write_text(json.dumps(smoke,indent=2));sg=internal_gate(smoke)
    (ROOT/'results/PREFLIGHT.json').write_text(json.dumps(dict(gate=gate,smoke_gate=sg,plan=decision),indent=2));print(json.dumps(dict(gate=gate,plan=decision),indent=2))
if __name__=='__main__':main()
