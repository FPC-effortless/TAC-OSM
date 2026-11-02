"""Read-only collection of all frozen fits; incomplete cohorts cannot pass.

This records descriptive exposed-DEV measurements. It does not turn aligned
parameter counts and token budgets into a compute-qualified comparison.
"""
import argparse, copy, hashlib, json, statistics
from pathlib import Path
import torch
from .common import ROOT, SPECS, EXPECTED, build, digest, internal_gate, require

SEEDS=(12001,12002,12003)

def collect(downloads):
    torch.set_num_threads(1)
    rows=[];gates={}
    for arm in SPECS:
        manifest=json.loads((ROOT/'registrations'/arm/'FAIR_COMPARISON.json').read_text())
        frozen_sha=hashlib.sha256((ROOT/'registrations'/arm/'FAIR_COMPARISON.json').read_bytes()).hexdigest()
        final=copy.deepcopy(manifest)
        for seed in SEEDS:
            folder=downloads/f'{arm}_{seed}'
            r=json.loads((folder/'FIT_RECEIPT.json').read_text())
            e=json.loads((folder/'EVALUATION.json').read_text())
            a=json.loads((ROOT/'results'/f'audit_{arm}_{seed}.json').read_text())
            require(r['status']=='FULL_FIT_COMPLETE_INTERNAL_UNQUALIFIED' and r['optimizer_updates']==r['attempted_updates']==1024,'incomplete original fit')
            require(r['protocol_sha256']==frozen_sha and r['source_hashes']==manifest['source_hashes'],'registration/source identity mismatch')
            require(a['status']=='PASS_READ_ONLY_TRAINED_SOURCE_AUDIT' and a['original_checkpoint_sha256']==r['checkpoint_sha256'],'trained audit absent or mismatched')
            require(e['checkpoint_sha256']==r['checkpoint_sha256'] and e['cold_replay_exact'] and e['weights_unchanged'],'original evaluator mismatch')
            for step in (0,256,512,768,1024):
                p=folder/f'checkpoint_{step:04}.pt'
                require(p.is_file() and digest(p)==p.with_suffix('.sha256').read_text().strip(),'checkpoint closure/hash mismatch')
                saved=torch.load(p,map_location='cpu',weights_only=True)
                require(saved['steps']==step and saved['protocol_sha256']==frozen_sha,'snapshot progress/protocol mismatch')
            model=build(arm,seed);model.load_state_dict(saved['model'],strict=True)
            count=sum(p.numel() for p in model.parameters())
            require(count==r['trainable_parameters']==EXPECTED[arm],'actual parameter count changed')
            initial=torch.load(folder/'checkpoint_0000.pt',map_location='cpu',weights_only=True)
            require(any(not torch.equal(initial['model'][k],v) for k,v in saved['model'].items()),'unchanged model')
            measured=dict(trainable_parameters=count,deployed_parameters=count,checkpoint_sha256=digest(folder/'checkpoint_1024.pt'))
            next(m for m in final['models'] if m['seed']==seed)['measured']=measured
            for gen in e['generation']:
                raw=(folder/f"generation_temperature_{gen['temperature']}.bin").read_bytes()
                require(len(raw)==gen['bytes']==512 and hashlib.sha256(raw).hexdigest()==gen['sha256'],'generation payload changed')
                require(gen['state_tensor_bytes'][0]==gen['state_tensor_bytes'][1],'generation state payload changed between observations')
            rows.append(dict(arm=arm,seed=seed,parameters=count,
                initial_dev_nll=e['initial_dev']['nll'],final_dev_nll=e['final_dev']['normal']['nll'],
                memory_zero_delta=None if arm=='attention' else e['final_dev']['memory_zero']['nll']-e['final_dev']['normal']['nll'],
                memory_shuffle_delta=None if arm=='attention' else e['final_dev']['memory_batch_shuffle']['nll']-e['final_dev']['normal']['nll'],
                optimizer_updates=r['optimizer_updates'],training_byte_presentations=r['actual_tokens'],
                optimizer_seconds=r['total_recorded_optimizer_seconds'],fit_elapsed_seconds=r['elapsed_seconds'],
                mean_optimizer_step_seconds=r['mean_step_seconds'],
                generation_state_tensor_bytes=e['generation'][0]['state_tensor_bytes'][0],
                checkpoint_sha256=r['checkpoint_sha256'],hard_bank_usage=a['hard_bank_usage']))
        gates[arm]=internal_gate(final,phase='final')
        (ROOT/'results'/f'FINAL_MANIFEST_{arm}.json').write_text(json.dumps(final,indent=2))
    aggregates=[]
    for arm in SPECS:
        group=[r for r in rows if r['arm']==arm]
        result=dict(arm=arm,seeds=list(SEEDS),parameters=EXPECTED[arm])
        for key in ('initial_dev_nll','final_dev_nll','memory_zero_delta','memory_shuffle_delta','mean_optimizer_step_seconds','generation_state_tensor_bytes'):
            values=[r[key] for r in group if r[key] is not None]
            result[key]=None if not values else dict(mean=statistics.mean(values),minimum=min(values),maximum=max(values),sample_sd=statistics.stdev(values) if len(values)>1 else None)
        aggregates.append(result)
    c=json.loads((ROOT/'results/CURVATURE_DIAGNOSTIC.json').read_text())
    require(len(c['rows'])==36,'curvature grid incomplete')
    curvature=[]
    for row in c['rows']:
        require(row['finite'] and row['optimizer_updates']==0,'invalid curvature diagnostic')
        for label,metric in row['metrics'].items():
            flags=torch.tensor(metric['predictions'])==torch.tensor(metric['targets'])
            require(abs(float(flags.double().mean())-metric['accuracy'])<1e-12,'curvature query summary wrong')
            require(abs(float(flags.reshape(-1,16).all(0).double().mean())-metric['complete_map'])<1e-12,'curvature map summary wrong')
    for mode in c['protocol']['modes']:
        group=[r for r in c['rows'] if r['mode']==mode and (r['N'],r['K'])==(256,32)]
        curvature.append(dict(mode=mode,N=256,K=32,seeds=[r['seed'] for r in group],
            present_query_accuracy_mean=statistics.mean(r['metrics']['present']['accuracy'] for r in group),
            complete_map_mean=statistics.mean(r['metrics']['present']['complete_map'] for r in group),
            curvature_state_bytes_per_owner=group[0]['curvature_state_bytes_per_owner'],
            other_state_bytes_per_owner=group[0]['memory_state_bytes_per_owner']+group[0]['support_bytes_per_owner']))
    spectral=json.loads((ROOT/'results/SPECTRAL_DIAGNOSTIC.json').read_text())
    require(len(spectral['rows'])==12,'spectral diagnostic incomplete')
    spectral_summary=[]
    for mode in spectral['protocol']['modes']:
        group=[r for r in spectral['rows'] if r['mode']==mode]
        for row in group:
            require(row['finite'] and row['optimizer_updates']==0,'invalid spectral diagnostic')
            for metric in row['metrics'].values():
                flags=torch.tensor(metric['predictions'])==torch.tensor(metric['targets'])
                require(abs(float(flags.double().mean())-metric['accuracy'])<1e-12,'spectral query summary wrong')
                require(abs(float(flags.reshape(-1,16).all(0).double().mean())-metric['complete_map'])<1e-12,'spectral map summary wrong')
        spectral_summary.append(dict(mode=mode,rank=group[0]['rank'],N=256,K=32,
            present_query_accuracy_mean=statistics.mean(r['metrics']['present']['accuracy'] for r in group),
            complete_map_mean=statistics.mean(r['metrics']['present']['complete_map'] for r in group),
            complete_map_each_seed=[r['metrics']['present']['complete_map'] for r in group],
            curvature_state_bytes_per_owner=group[0]['curvature_state_bytes_per_owner'],
            other_state_bytes_per_owner=group[0]['other_state_bytes_per_owner'],
            operator_seconds_mean=statistics.mean(r['operator_seconds'] for r in group)))
    result=dict(study='COLLECT012',status='ALL_NINE_FIXED_BUDGET_INTERNAL_FITS_COMPLETE',
        frozen_fit_commit='036d778255a416d3aae886acdbfeef8fb1b6a1ab',
        workflow_run='https://github.com/FPC-effortless/TAC-OSM/actions/runs/41513912016',
        metrics_unit='nats per next-byte target; lower is better',rows=rows,aggregates=aggregates,
        full_fit_optimizer_updates=sum(r['optimizer_updates'] for r in rows),
        separate_local_optimizer_smoke_updates=32,
        training_byte_presentations=sum(r['training_byte_presentations'] for r in rows),
        unique_exposed_dev_targets=8192,dev_cohort_shared_across_all_models=True,
        immutable_full_fit_checkpoints=45,trained_generation_samples=18,
        summed_optimizer_seconds=sum(r['optimizer_seconds'] for r in rows),
        summed_fit_elapsed_seconds=sum(r['fit_elapsed_seconds'] for r in rows),
        curvature_diagnostic_highest_load=curvature,spectral_diagnostic_highest_load=spectral_summary,final_internal_gates=gates,
        protected_test_read=False,independently_qualified=False,eligible_for_architecture_superiority=False,
        full_training_flops='UNMEASURED',
        protocol_notes=['The manifest lists sample_seeds 12011/12012/12013, but the frozen evaluator uses sampling RNG 120100 for each temperature and model. Generation is a fixed execution diagnostic, not a three-seed sampling study.',
          'Postfit zero/shuffle interventions remove only the added phase-memory output; native TAC carried banks remain.',
          'Each fit presents 524288 TRAIN bytes with replacement, about 0.522 times the TRAIN byte count; this is a learnability pilot, not a converged language model.',
          'Timing describes observed CPU executions on separate workers. Hardware-controlled throughput, peak memory and full FLOPs parity are not measured.',
          'Finite reference/gradient checks and exact same-runtime replay are executed evidence, not full formal or independent qualification.'])
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--downloads',type=Path,required=True);a=p.parse_args()
    result=collect(a.downloads);(ROOT/'results/SUMMARY.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','final_internal_gates')},indent=2))

if __name__=='__main__':main()
