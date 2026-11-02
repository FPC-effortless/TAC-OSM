"""Collect every fixed-budget result, preserving control/cohort distinctions."""
import argparse,collections,hashlib,json,math,statistics
from pathlib import Path
import torch
from .common import ROOT,SPECS,build,digest,internal_gate,require
from collect012.train import fingerprint

SEEDS=(12001,12002,12003)
VARIANTS=('control','balance','write_first','combined')

def read(p):return json.loads(p.read_text())
def description(values):
    return dict(mean=statistics.mean(values),minimum=min(values),maximum=max(values),sample_sd=statistics.stdev(values) if len(values)>1 else None)

def verify_queries(obj):
    rows=obj['rows']
    for summary in obj['summaries']:
        sub=[r for r in rows if r['mode']==summary['mode'] and r['delay']==summary['delay']]
        present=[r for r in sub if r['kind']!='absent'];absent=[r for r in sub if r['kind']=='absent'];over=[r for r in present if r['kind']=='overwrite']
        acc=lambda rr:sum(r['prediction']==r['target'] for r in rr)/len(rr)
        require(abs(acc(present)-summary['present_accuracy'])<1e-12,'present score mismatch')
        require(abs(acc(absent)-summary['absent_marker_accuracy'])<1e-12,'absent score mismatch')
        require(abs(acc(over)-summary['overwrite_accuracy'])<1e-12,'overwrite score mismatch')
        maps=sum(all(r['prediction']==r['target'] for r in present if r['owner']==i) for i in range(12))/12
        require(abs(maps-summary['complete_present_map'])<1e-12,'map score mismatch')
        for r in sub:
            require(all(math.isfinite(x) for x in r['candidate_logits']) and math.isfinite(r['target_log_probability']),'nonfinite query score')
            require(r['prediction']=='abcdefgh?'[max(range(9),key=lambda k:r['candidate_logits'][k])],'argmax differs from retained logits')
    return True

def collect(downloads,old_downloads):
    torch.set_num_threads(1);rows=[];gates={};old_registry=read(ROOT/'REUSED_CONTROLS.json')
    old_receipts={s:read(old_downloads/f'hybrid_hard_p_{s}'/'FIT_RECEIPT.json') for s in SEEDS}
    for variant in VARIANTS:
        manifest=None if variant=='control' else read(ROOT/'registrations'/variant/'FAIR_COMPARISON.json')
        for seed in SEEDS:
            is_control=variant=='control';folder=old_downloads/f'hybrid_hard_p_{seed}' if is_control else downloads/f'{variant}_{seed}'
            receipt=read(folder/'FIT_RECEIPT.json');checkpoint=folder/receipt['checkpoint']
            require(receipt['status']=='FULL_FIT_COMPLETE_INTERNAL_UNQUALIFIED' and receipt['optimizer_updates']==receipt['attempted_updates']==1024,'incomplete original fit')
            require(digest(checkpoint)==receipt['checkpoint_sha256'],'final checkpoint changed')
            initial=torch.load(folder/'checkpoint_0000.pt',weights_only=True,map_location='cpu')
            m=build('balance' if is_control else variant,seed);m.load_state_dict(initial['model'],strict=True)
            require(fingerprint(m)==receipt['initial_parameter_sha256'],'actual initializer fingerprint differs')
            require(sum(p.numel() for p in m.parameters())==99971,'actual count mismatch')
            # Actual initializer identity must hold for every arm and seed.
            require(receipt['initial_parameter_sha256']==old_receipts[seed]['initial_parameter_sha256'],'initializer differs from reused control')
            for step in (0,256,512,768,1024):
                p=folder/f'checkpoint_{step:04}.pt';require(p.is_file() and digest(p)==p.with_suffix('.sha256').read_text().strip(),'checkpoint closure/hash mismatch')
                state=torch.load(p,weights_only=True,map_location='cpu');require(state['steps']==step,'snapshot progress differs')
            lines=[json.loads(x) for x in (folder/'journal.jsonl').read_text().splitlines()]
            original=[json.loads(x) for x in (old_downloads/f'hybrid_hard_p_{seed}'/'journal.jsonl').read_text().splitlines()]
            require(len(lines)==len(original)==1024,'journal budget incomplete')
            require(all(a['indices']==b['indices'] for a,b in zip(lines,original)),'actual minibatch exposure differs')
            if not is_control:
                audit=read(ROOT/'results'/f'audit_{variant}_{seed}.json')
                require(audit['status']=='PASS_READ_ONLY_TRAINED_SOURCE_AUDIT' and audit['original_checkpoint_sha256']==receipt['checkpoint_sha256'],'trained source audit absent/mismatched')
                require(receipt['source_hashes']==manifest['source_hashes'],'fit source differs')
                require(receipt['protocol_sha256']==digest(ROOT/'registrations'/variant/'FAIR_COMPARISON.json'),'registration differs')
                for line in lines:
                    require(abs(line['objective']-line['nll']-SPECS[variant]['balance']*line['balance_loss'])<2e-6,'charged objective differs')
                    require(len(line['bank_counts'])==4 and all(sum(c)==512 for c in line['bank_counts']),'per-step route counts differ')
                next(m for m in manifest['models'] if m['seed']==seed)['measured']=dict(trainable_parameters=99971,deployed_parameters=99971,checkpoint_sha256=receipt['checkpoint_sha256'])
            results=ROOT/'results'/f'{variant}_{seed}';common=read(results/'COMMON_EVALUATION.json')
            require(common['checkpoint_sha256']==receipt['checkpoint_sha256'] and common['weights_unchanged'],'common runtime identity differs')
            primary=read(results/'COMMON_RETRIEVAL.json');balanced=read(results/'SUPPLEMENTAL_BALANCED.json')
            verify_queries(primary);verify_queries(balanced)
            for cohort in (primary,balanced):
                for summary in cohort['summaries']:
                    rr=[r for r in cohort['rows'] if r['mode']==summary['mode'] and r['delay']==summary['delay'] and r['kind']!='absent']
                    summary['full_vocab_present_accuracy']=sum(r['full_vocab_prediction']==ord(r['target']) for r in rr)/len(rr)
            counts=collections.Counter(r['target'] for r in balanced['rows'] if r['mode']=='normal' and r['delay']==64 and r['kind']!='absent')
            require(counts==dict.fromkeys('abcdefgh',6),'supplemental label balance differs')
            carry=read(results/'COMMON_CONTINUATION.json')
            carried=next(r['chunk_nll'] for r in carry['rows'] if r['mode']=='carry');reset=next(r['chunk_nll'] for r in carry['rows'] if r['mode']=='reset')
            wrong=next(r['chunk_nll'] for r in carry['rows'] if r['mode']=='wrong_history')
            original_eval=read(folder/'EVALUATION.json')
            cohort_eval=read((ROOT/'results'/f'control_{seed}' if is_control else folder)/'EVALUATION.json')
            for gen in cohort_eval['generation']:
                require(gen['state_tensor_bytes'][0]==gen['state_tensor_bytes'][1],'generation tensor payload changed between observations')
                gp=(ROOT/'results'/f'control_{seed}' if is_control else folder)/f"generation_temperature_{gen['temperature']}.bin"
                require(digest(gp)==gen['sha256'] and gp.stat().st_size==512,'generation changed')
            split=None
            if variant in ('write_first','combined'):
                split=read(results/'READ_SPLIT.json');require(split['checkpoint_sha256']==receipt['checkpoint_sha256'],'read split checkpoint differs')
            layers=common['routing'];fractions=[max(r['counts'])/sum(r['counts']) for r in layers]
            rows.append(dict(variant=variant,seed=seed,parameters=99971,checkpoint_sha256=receipt['checkpoint_sha256'],
                initial_parameter_sha256=receipt['initial_parameter_sha256'],reused_control=is_control,
                common_torch=common['torch'],common_dev_nll=common['final_dev']['normal']['nll'],
                original_evaluation_dev_nll=original_eval['final_dev']['normal']['nll'],
                phase_zero_delta=common['final_dev']['memory_zero']['nll']-common['final_dev']['normal']['nll'],
                phase_shuffle_delta=common['final_dev']['memory_batch_shuffle']['nll']-common['final_dev']['normal']['nll'],
                largest_bank_mean=statistics.mean(fractions),largest_bank_maximum=max(fractions),routing=layers,
                primary_retrieval=primary['summaries'],balanced_retrieval=balanced['summaries'],
                continuation=carry['rows'],continuation_carry_gain=statistics.mean(reset[1:])-statistics.mean(carried[1:]),
                continuation_wrong_history_cost=statistics.mean(wrong[1:])-statistics.mean(carried[1:]),
                read_split=split,optimizer_steps=1024,fit_seconds=receipt['elapsed_seconds'],optimizer_seconds=receipt['total_recorded_optimizer_seconds'],
                mean_step_seconds=receipt['mean_step_seconds'],byte_presentations=receipt['actual_tokens'],
                generation_state_tensor_bytes=cohort_eval['generation'][0]['state_tensor_bytes'][0]))
        if manifest:
            gates[variant]=internal_gate(manifest,phase='final')
            (ROOT/'results'/f'FINAL_MANIFEST_{variant}.json').write_text(json.dumps(manifest,indent=2))
    baseline={r['seed']:r['common_dev_nll'] for r in rows if r['variant']=='control'}
    aggregate=[]
    for variant in VARIANTS:
        group=[r for r in rows if r['variant']==variant]
        a=dict(variant=variant,parameters=99971,seeds=list(SEEDS))
        for metric in ('common_dev_nll','largest_bank_mean','largest_bank_maximum','phase_zero_delta','phase_shuffle_delta','continuation_carry_gain','continuation_wrong_history_cost','mean_step_seconds'):
            a[metric]=description([r[metric] for r in group])
        a['paired_dev_delta_vs_control']=description([r['common_dev_nll']-baseline[r['seed']] for r in group])
        retrieval=[]
        for cohort in ('primary_retrieval','balanced_retrieval'):
            for delay in (64,256,1024):
                for mode in ('normal','wrong_history','reset'):
                    rr=[next(x for x in r[cohort] if x['mode']==mode and x['delay']==delay) for r in group]
                    retrieval.append(dict(cohort=cohort,delay=delay,mode=mode,**{k:description([x[k] for x in rr]) for k in ('present_accuracy','absent_marker_accuracy','overwrite_accuracy','complete_present_map','absent_false_accept','full_vocab_present_accuracy')}))
        a['retrieval']=retrieval;aggregate.append(a)
    require(len({r['common_torch'] for r in rows})==1,'comparison runtime differs')
    return dict(study='COLLECT014',status='ALL_NINE_NEW_FIXED_BUDGET_FITS_AND_THREE_REUSED_CONTROLS_COLLECTED',
        frozen_fit_commit='6e19362ab0028b7e857e4506657f2703403b451f',workflow_run=41532283413,
        rows=rows,aggregates=aggregate,final_internal_gates=gates,
        new_full_fit_optimizer_updates=9216,reused_control_optimizer_updates=3072,separate_new_smoke_updates=24,
        new_full_fit_checkpoints=45,reused_control_checkpoints=15,
        new_byte_presentations=sum(r['byte_presentations'] for r in rows if not r['reused_control']),
        new_optimizer_seconds=sum(r['optimizer_seconds'] for r in rows if not r['reused_control']),
        new_fit_seconds=sum(r['fit_seconds'] for r in rows if not r['reused_control']),
        shared_original_dev_targets=8192,unique_primary_histories=12,unique_balanced_histories=12,
        generation_samples_in_this_cohort=24,
        primary_probe_constant_present_baseline=.25,supplemental_probe_constant_present_baseline=.125,
        comparison_eligibility=False,full_training_flops='UNMEASURED',protected_test=False,independently_qualified=False,
        limitations=['Single-graph internal repairs, not a qualified architecture ranking or frontier scale result.',
          'Zero-shot generated retrieval grammar was not trained; floor performance does not isolate physical storage capacity.',
          'Supplemental balanced cohort repairs an original label imbalance; primary probe retained and both disclosed.',
          'Read-split mechanism diagnostic was declared after launch before inspecting new full-fit outcomes; not a trained alternative.',
          'All seeds reuse the same evaluation histories/windows; repetitions are not additional independent observations.'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--downloads',type=Path,required=True);p.add_argument('--old-downloads',type=Path,required=True);a=p.parse_args()
    result=collect(a.downloads,a.old_downloads);(ROOT/'results/SUMMARY.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','aggregates','final_internal_gates')},indent=2))

if __name__=='__main__':main()
