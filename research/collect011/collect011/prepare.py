"""Honest no-fit inventory, draft fit registration and canonical preflight.

Unknown resource/provenance evidence stays null. This cannot authorize a fit.
"""
import hashlib,json,subprocess,sys
from pathlib import Path
import torch
from .model import CollectiveTAC,FFN_DIM as FFN

def main():
    torch.set_num_threads(1);root=Path(__file__).resolve().parents[1]
    split=json.loads((root/'data/SPLIT_MANIFEST.json').read_text())
    models=[]
    for arm in CollectiveTAC.ARMS:
        m=CollectiveTAC(arm,ffn_dim=FFN[arm]);n=sum(p.numel() for p in m.parameters())
        models.append(dict(architecture=arm,trainable_parameters=n,deployed_parameters=n,
            ffn_dim=FFN[arm],state_bytes_per_stream_at_start=m.start(('count',)).state_bytes()))
    inventory=dict(models=models,relative_parameter_gap=(max(m['trainable_parameters'] for m in models)/min(m['trainable_parameters'] for m in models)-1),optimizer_updates=0)
    (root/'results/PARAMETER_INVENTORY.json').write_text(json.dumps(inventory,indent=2))
    plan=dict(schema_version='tac-osm.parity.v1',study_id='COLLECT011-LM',study_type='architecture_comparison',
        status='DRAFT_NOT_QUALIFIED_NO_FITS',registered_before_first_fit=True,model_scale_policy='declared-target-v2',
        parameter_scale=dict(target_trainable_parameters=208000,target_deployed_parameters=208000,relative_tolerance=.05,
            registered_before_first_fit=True,rationale='Four physical TAC layers retain native fast/slow/rotating-field and SSA modules; smallest selected width 32. All used parameters counted. FFNs adjusted for <=1% gap.'),
        real_world_source=dict(name='Pinned Tiny Shakespeare TRAIN/DEV from TACOPT114 artifact',sha256=split['source_sha256'],
            license='OPEN: original publisher/license authority must be attached to signed source evidence',observed_real_outcomes=None),
        arms=[dict(**m,seed=seed) for seed in (1,2,3) for m in models],
        proposed_fit=dict(steps=1024,batch_size=4,sequence_length=128,optimizer='AdamW common recipe',
            learning_rate=.001,weight_decay=.1,clip_norm=1.,warmup=64,selection='fixed final update',
            tuning_trials=0,training_sample_order='per-seed fixed precomputed window indices shared across arms',
            train_split_sha256=split['parts']['train']['sha256'],dev_split_sha256=split['parts']['dev']['sha256'],
            test_access=False,planned_training_flops=None,max_training_flops=None,execution_allocation=None),
        unresolved=['genuine externally signed source/outcome/access evidence','full source-bound PNDS graph formulation and verification',
            'qualified matched full-update FLOP budgets','approved compute execution allocation','sealed TEST access remains disabled'],
        authorization='User explicitly requested integration and testing. Repository signed evidence and compute qualification remain unresolved; no keys/attestations are fabricated.')
    (root/'FAIR_COMPARISON.DRAFT.json').write_text(json.dumps(plan,indent=2))
    proc=subprocess.run([sys.executable,str(root/'governance/scripts/validate_research_parity_v2.py'),
        '--manifest',str(root/'FAIR_COMPARISON.DRAFT.json'),'--phase','preflight'],capture_output=True,text=True)
    result=dict(command='validate_research_parity_v2.py --manifest FAIR_COMPARISON.DRAFT.json --phase preflight',
        exit_code=proc.returncode,stdout=proc.stdout,stderr=proc.stderr,approved_to_train=False,
        unresolved=plan['unresolved'],optimizer_updates=0)
    (root/'results/TRAINING_PREFLIGHT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    proposal=dict(schema='tac-osm.phone-research.v1',run_id='COLLECT011-NOFIT',variant='System 1',kind='no_fit',
        issue_number=686,estimated_cpu_minutes=30,requires_cuda=False,gpu_speedup_expected=False,
        estimated_gpu_minutes=0,gpu_strategy=None,seeds=[1,2,3],real_source_sha256=split['source_sha256'],
        study_manifest=None,workload=None,diagnostic='observatory',dataset_sources=[],budget_gpu_minutes=0)
    path=root/'governance/research/phone_queue/collect011_no_fit.json';path.write_text(json.dumps(proposal,indent=2))
    proc=subprocess.run([sys.executable,'scripts/phone_lab.py','plan','research/phone_queue/collect011_no_fit.json'],cwd=root/'governance',capture_output=True,text=True)
    (root/'results/PHONE_PLAN.json').write_text(json.dumps(dict(exit_code=proc.returncode,stdout=proc.stdout,stderr=proc.stderr,
        qualification='Routing only; this proposal was materialized after the initial engineering checks, not a retrospective training preregistration.'),indent=2))
    print(proc.stdout,proc.stderr)
    # Direct canonical signed verifier: independently record the missing-evidence
    # gate rather than masking it behind the earlier source-authenticity failure.
    sys.path.insert(0,str(root/'governance/src'))
    from tac_osm.research_reward_runner import verify_reward_attestation,RewardEvidenceError
    try:
        verify_reward_attestation(plan,'preflight',None,trusted_public_key_hex=None)
    except RewardEvidenceError as e:
        (root/'results/SIGNED_GATE.json').write_text(json.dumps(dict(status='BLOCKED',reason=str(e),approved_to_train=False),indent=2))

if __name__=='__main__':main()
