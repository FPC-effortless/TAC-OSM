"""Register three repairs and immutable reused controls before any new fitting."""
import copy,json,sys
from .common import ROOT,OLD,SPECS,build,digest,source_paths,internal_gate,verify_data

def main():
    data,split=verify_data();original=json.loads((OLD/'registrations/hybrid_hard_p/FAIR_COMPARISON.json').read_text())
    original['real_world_source']['source_receipt']='../collect012/data/SOURCE_QUALIFICATION.json'
    sources={p:digest(f) for p,f in source_paths().items()};gates={}
    controls=json.loads((ROOT/'REUSED_CONTROLS.json').read_text())
    for arm,spec in SPECS.items():
        n=sum(p.numel() for p in build(arm,12001).parameters());manifest=copy.deepcopy(original)
        manifest.update(study_id='COLLECT014-'+arm.upper(),model_config=dict(width=16,ffn_dim=57,depth=4,**spec),
            models=[dict(architecture=arm,seed=s,trainable_parameters=n,deployed_parameters=n) for s in (12001,12002,12003)],
            source_hashes=sources,training_source_hashes={f'collect014/{p}.py':digest(ROOT/f'collect014/{p}.py') for p in ('train','common','register')},
            causal_protocol='Observed byte x[t] predicts x[t+1], fresh state per 128-byte TRAIN window. Router balance is a training loss; forward routing is owner-local and causal. No synthetic probe labels or future bytes enter FIT.',
            reused_controls=controls,repair=spec,
            scientific_status='INTERNAL_FACTORIAL_REPAIR_PILOT_NOT_FLOPS_MATCHED_ARCHITECTURE_SUPERIORITY',
            comparison_gate='OPEN: full training FLOPs and independent qualification remain unmeasured; descriptive same-graph repair ablations only')
        manifest['fit'].update(sample_seeds=[140100],generation_rng=140100,worker_seconds_cap=2400,
            retrieval_protocol='EVALUATION_PROTOCOL.json',balance_coefficient=spec['balance'],read_after=spec['read_after'])
        dest=ROOT/'registrations'/arm;dest.mkdir(parents=True,exist_ok=True)
        (dest/'FAIR_COMPARISON.json').write_text(json.dumps(manifest,indent=2));gates[arm]=internal_gate(manifest)
        proposal=json.loads((OLD/'policy/research/phone_queue/collect012_hybrid_hard_p.json').read_text())
        alias=f'research/mathematical-engineering/collect014_{arm}/FAIR_COMPARISON.json'
        target=ROOT/'policy'/alias;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text((dest/'FAIR_COMPARISON.json').read_text())
        proposal.update(run_id='COLLECT014-'+arm.upper(),study_manifest=alias)
        (dest/'PHONE_PROPOSAL.json').write_text(json.dumps(proposal,indent=2))
    smoke=copy.deepcopy(manifest);smoke.update(study_id='COLLECT014-OPTIMIZER-SMOKE',
        models=[dict(architecture=arm,seed=12000,trainable_parameters=99971,deployed_parameters=99971) for arm in SPECS])
    smoke['fit'].update(steps=8,checkpoint_steps=[0,8],seed_ids=[12000],worker_seconds_cap=300)
    smoke['repair']=SPECS
    smoke['model_config']=dict(width=16,ffn_dim=57,depth=4,variants=SPECS)
    (ROOT/'SMOKE_REGISTRATION.json').write_text(json.dumps(smoke,indent=2));gates['smoke']=internal_gate(smoke)
    (ROOT/'results/INTERNAL_PREFLIGHT.json').write_text(json.dumps(gates,indent=2));print(json.dumps(gates,indent=2))

if __name__=='__main__':main()
