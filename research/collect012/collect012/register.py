"""Freeze genuine text/size/source prerequisites before any optimizer call.

Each single-graph registration asks whether that graph learns ordinary text;
it does not use single-model gates to qualify a cross-architecture comparison.
The separate comparison gate stays OPEN until full FLOPs parity is established.
"""
from pathlib import Path
import hashlib,json,sys
import torch
from .common import ROOT,SPECS,EXPECTED,build,verify_data,source_paths,digest,canonical,internal_gate

PUBLISHER='https://github.com/karpathy/char-rnn/blob/6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e/data/tinyshakespeare/input.txt'
LICENSE='https://github.com/karpathy/char-rnn/blob/6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e/Readme.md#license'

def main():
    torch.set_num_threads(1);data,split=verify_data()
    qualification=json.loads((ROOT/'data/SOURCE_QUALIFICATION.json').read_text())
    assert qualification['publisher_blob']=='7dcb3a2d4cc3b48b6283dd46870bfeb78f88aac9' and qualification['license']=='MIT'
    # A fixed complete budget; truncation is partial, never successful fitting.
    fit=dict(steps=1024,batch_size=4,sequence_length=128,learning_rate=.0003,betas=[.9,.95],
        weight_decay=.1,clip_norm=1.,precision='float32',optimizer='AdamW',
        schedule='constant; no tuning',fresh_state_each_window=True,checkpoint_steps=[0,256,512,768,1024],
        dev_windows=64,seed_ids=[12001,12002,12003],sample_seeds=[12011,12012,12013],
        allocation_gpu_minutes=0,worker_seconds_cap=2400,execution_backend='github_actions_cpu',
        planned_dev_interventions=['normal','memory_zero','memory_batch_shuffle'],
        generation_tokens=512,generation_temperatures=[.8,1.2])
    cohort=hashlib.sha256(canonical(split)).hexdigest()
    shared=dict(schema_version='tac-osm.parity.v1',study_type='single_model_training',
        execution_level='internal_research',registered_before_first_fit=True,
        research_scope=dict(publicly_licensed=True,partitions=['FIT','DEV'],protected_evaluation=False,
            reward_integrity_certification=False,safety_claim=False,checkpoint_promotion=False),
        real_world_source=dict(name='Pinned Tiny Shakespeare publisher text',sha256=split['source_sha256'],
            observed_real_outcomes=True,publisher_url=PUBLISHER,license='MIT',license_evidence_url=LICENSE,
            source_receipt='data/SOURCE_QUALIFICATION.json'),
        split_policy=dict(frozen=True,cohort_sha256=cohort,rule='original contiguous 90/5/5; TEST bytes absent'),
        causal_protocol='Input bytes x[t] predict x[t+1]; fresh state per original FIT window; no future/grammar/role labels. FIT-only optimization, fixed postfit DEV windows, no DEV selection.',
        checkpoint_policy='Retain initialization and immutable 256-step model/AdamW/RNG snapshots; exact source/data/protocol hashes and input replay. Fixed final update, never best DEV.',
        scientific_status='INTERNAL_LEARNABILITY_NOT_RESOURCE_MATCHED_ARCHITECTURE_COMPARISON',
        comparison_gate='OPEN: measured complete charged FLOPs and independent qualification not established; do not rank architectures from these fits.')
    manifests={};gates={}
    for arm in SPECS:
        m=build(arm,fit['seed_ids'][0]);n=sum(p.numel() for p in m.parameters())
        manifest=dict(shared,study_id='COLLECT012-'+arm.upper(),model_config=SPECS[arm],fit=fit,
            models=[dict(architecture=arm,seed=s,trainable_parameters=n,deployed_parameters=n) for s in fit['seed_ids']],
            source_hashes={p:digest(f) for p,f in source_paths().items()},
            training_source_hashes={p:digest(ROOT/p) for p in ('collect012/train.py','collect012/common.py','collect012/register.py')})
        target=ROOT/'registrations'/arm;target.mkdir(parents=True,exist_ok=True)
        (target/'FAIR_COMPARISON.json').write_text(json.dumps(manifest,indent=2))
        manifests[arm]=manifest;gates[arm]=internal_gate(manifest)
    # The short, separately registered optimizer diagnostic is NOT a full fit.
    smoke=dict(shared,study_id='COLLECT012-OPTIMIZER-SMOKE',
        fit=dict(fit,steps=8,checkpoint_steps=[0,8],seed_ids=[12000],sample_seeds=[],worker_seconds_cap=300),
        models=[dict(architecture=a,seed=12000,trainable_parameters=EXPECTED[a],deployed_parameters=EXPECTED[a]) for a in SPECS],
        purpose='8 actual optimizer updates per graph; source/checkpoint/AdamW path verification, not language capability')
    (ROOT/'SMOKE_REGISTRATION.json').write_text(json.dumps(smoke,indent=2));gates['smoke']=internal_gate(smoke)
    (ROOT/'results/INTERNAL_PREFLIGHT.json').write_text(json.dumps(gates,indent=2))
    print(json.dumps(gates,indent=2))

if __name__=='__main__':main()
