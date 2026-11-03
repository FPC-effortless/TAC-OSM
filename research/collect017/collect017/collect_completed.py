"""Require all three original attention artifacts, source audits and raw-score recomputation."""
import argparse,collections,copy,hashlib,json,math,statistics
from pathlib import Path
import torch
from .common import ROOT,ARMS,SEEDS,EXPECTED,canonical,digest,require,internal_gate
def recompute(probe):
    rows=probe['rows'];out=[]
    groups=sorted(set((r['keys'],r['gap'],r['condition']) for r in rows))
    for n,gap,condition in groups:
        cell=[r for r in rows if (r['keys'],r['gap'],r['condition'])==(n,gap,condition)]
        p=[r for r in cell if not r['absent']];a=[r for r in cell if r['absent']]
        require(len(p)==24*n and len(a)==48,'incomplete cohort')
        for row in cell:
            answer=bytes.fromhex(row['answer_hex']);pred=bytes.fromhex(row['prediction_hex'])
            require(row['first_top1']==pred[0] and row['first_correct']==int(pred[0]==answer[0]) and row['exact4']==int(pred==answer),'raw predictions disagree')
            require(row['closed_first_correct']==int(row['closed_first']==answer[0]),'closed score disagrees')
            if 'first_logits' in row:
                v=row['first_logits'];require(len(v)==256 and all(math.isfinite(x) for x in v),'missing/nonfinite logit vector')
                require(max(range(256),key=lambda i:v[i])==row['first_top1'],'full vocabulary argmax differs')
                candidates=list(b'abcdefgh')+[63];require(max(candidates,key=lambda i:v[i])==row['closed_first'],'candidate argmax differs')
        m=lambda z,k:sum(r[k] for r in z)/len(z)
        maps=sum(all(r['exact4'] for r in p if r['history']==i) for i in range(24))/24
        firstmaps=sum(all(r['first_correct'] for r in p if r['history']==i) for i in range(24))/24
        entry=dict(keys=n,gap=gap,condition=condition,present_queries=len(p),absent_queries=len(a),first_accuracy=m(p,'first_correct'),
            closed_first_accuracy=m(p,'closed_first_correct'),exact4=m(p,'exact4'),complete_maps=maps,first_complete_maps=firstmaps,
            absent_first=m(a,'first_correct'),absent_exact4=m(a,'exact4'),present_mean_log_probability=m(p,'target_log_probability'))
        previous=next(q for q in probe['summary'] if (q['keys'],q['gap'],q['condition'])==(n,gap,condition))
        for k,v in entry.items():
            require(v==previous[k] if isinstance(v,(int,str)) else abs(v-previous[k])<1e-12,'summary differs from raw '+k)
        out.append(entry)
    return out
def main():
    p=argparse.ArgumentParser();p.add_argument('--runs-root',type=Path,required=True);a=p.parse_args();records=[];actual={}
    protocol=json.loads((ROOT/'EVALUATION_PROTOCOL.json').read_text());g=protocol['gates']
    for arm in ARMS:
      for seed in SEEDS:
        folder=a.runs_root/f'{arm}_{seed}';extra=ROOT/'results'/f'{arm}_{seed}'
        receipt=json.loads((folder/'FIT_RECEIPT.json').read_text());require(receipt['arm']==arm and receipt['seed']==seed and receipt['optimizer_updates']==512,'wrong original')
        audit=json.loads((ROOT/'results'/f'audit_{arm}_{seed}.json').read_text());require(audit['status']=='PASS_TRAINED_SOURCE_AUDIT','trained audit failed')
        e=json.loads((folder/'EVALUATION.json').read_text());raw=json.loads((folder/'RETRIEVAL.json').read_text());summary=recompute(raw)
        require(e['checkpoint_sha256']==receipt['checkpoint_sha256']==digest(folder/'checkpoint_0512.pt'),'checkpoint identity differs')
        require(e['weights_unchanged'] and e['optimizer_updates']==0,'evaluation was not read-only')
        require(e['retrieval_summary']==raw['summary'],'evaluation/probe summary differs')
        sup=json.loads((extra/'SUPPLEMENTAL.json').read_text());require(sup['checkpoint_sha256']==receipt['checkpoint_sha256'] and sup['weights_unchanged'],'supplemental model differs')
        for stratum in sup['results']:recompute(stratum)
        generation=json.loads((extra/'GENERATION.json').read_text());require(generation['checkpoint_sha256']==receipt['checkpoint_sha256'] and generation['weights_unchanged'],'generation model differs')
        for sample in generation['rows']:
            require(len(bytes.fromhex(sample['hex']))==sample['generated_bytes'],'generation bytes differ')
            require(sample['status']=='COMPLETE' and sample['generated_bytes']==512,'generation failed or incomplete')
            require(sample['payload'][0]['state_tensor_bytes']<=sample['payload'][1]['state_tensor_bytes'],'attention cache shrank unexpectedly')
        pick=lambda c:next(v for v in summary if v['keys']==4 and v['gap']==256 and v['condition']==c)
        normal=pick('normal');wrong=pick('wrong_history');reset=pick('reset_all');swap=pick('query_swap')
        gates=dict(present=normal['first_accuracy']>=g['each_seed_present_first_accuracy_min'],
            wrong_history_gap=normal['first_accuracy']-wrong['first_accuracy']>=g['each_seed_normal_minus_wrong_min'],
            reset_gap=normal['first_accuracy']-reset['first_accuracy']>=g['each_seed_normal_minus_reset_min'],
            absence=normal['absent_first']>=g['each_seed_absent_first_min'])
        actual[f'{arm}_{seed}']=audit['exposure_sequence_sha256']
        supplemental={q['stratum']['name']:q['summary'] for q in sup['results']}
        gradients=json.loads((extra/'GRADIENT_PROFILE.json').read_text())
        records.append(dict(arm=arm,seed=seed,parameters=EXPECTED[arm],checkpoint_sha256=receipt['checkpoint_sha256'],
            dev_nll=e['ordinary_dev']['nll'],primary_normal=normal,primary_wrong=wrong,primary_reset=reset,primary_swap=swap,
            retrieval_summary=summary,supplemental=supplemental,gates=gates,all_primary_gates=all(gates.values()),
            optimizer_seconds=receipt['total_recorded_optimizer_seconds'],mean_step_seconds=receipt['mean_step_seconds'],
            continuation=e['continuation'],cross_boundary_answer_credit=audit['cross_boundary_answer_credit'],first_answer_gradient_profile=gradients['rows'][0],
            bank_counts=[json.loads(line)['bank_counts'] for line in (folder/'journal.jsonl').read_text().splitlines()],
            generation_statuses=[v['status'] for v in generation['rows']]))
    for seed in SEEDS:require(len({actual[f'{arm}_{seed}'] for arm in ARMS})==1,'paired original input exposure differs')
    aggregates=[]
    for arm in ARMS:
        z=[r for r in records if r['arm']==arm];avg=lambda f:statistics.mean(f(r) for r in z)
        aggregates.append(dict(arm=arm,parameters=EXPECTED[arm],dev_nll_mean=avg(lambda r:r['dev_nll']),dev_nll_sd=statistics.stdev(r['dev_nll'] for r in z),
            first_accuracy=avg(lambda r:r['primary_normal']['first_accuracy']),closed_first_accuracy=avg(lambda r:r['primary_normal']['closed_first_accuracy']),
            exact4=avg(lambda r:r['primary_normal']['exact4']),complete_maps=avg(lambda r:r['primary_normal']['complete_maps']),
            absent_first=avg(lambda r:r['primary_normal']['absent_first']),wrong_first=avg(lambda r:r['primary_wrong']['first_accuracy']),
            reset_first=avg(lambda r:r['primary_reset']['first_accuracy']),swap_first=avg(lambda r:r['primary_swap']['first_accuracy']),
            seeds_passing=sum(r['all_primary_gates'] for r in z),optimizer_seconds=sum(r['optimizer_seconds'] for r in z)))
        manifest=json.loads((ROOT/'registrations'/arm/'FAIR_COMPARISON.json').read_text())
        for row in manifest['models']:
            r=next(q for q in z if q['seed']==row['seed']);row['measured']=dict(trainable_parameters=r['parameters'],deployed_parameters=r['parameters'],checkpoint_sha256=r['checkpoint_sha256'])
        gate=internal_gate(manifest,phase='final');require(not gate['eligible_for_architecture_superiority'],'internal study promoted')
        (ROOT/'results'/f'FINAL_MANIFEST_{arm}.json').write_text(json.dumps(manifest,indent=2));(ROOT/'results'/f'FINAL_GATE_{arm}.json').write_text(json.dumps(gate,indent=2))
    result=dict(status='COMPLETE_INTERNAL_POST_NULL_ATTENTION_CONTROL',frozen_fit_source='0da7f1b24354aedf3183131fba987c72d1a94618',graph_source='e5813459c9337af058cdfb2d3e864c767c457053',aggregates=aggregates,records=records,
        updates=1536,smoke_updates=8,presented_training_bytes=1179648,trained_checkpoints=12,smoke_checkpoints=2,generation_samples=6,
        gpu_minutes=0,paired_exposure_verified=True,full_training_flops='OPEN',independent_qualification=False,global_attention_replacement=False,
        caution='Post-C16-null control. Not pure Transformer; native other memories retained. 24 base identities; delays reused; no TEST. Window1536 covers registered prefixes only.')
    (ROOT/'results/SUMMARY.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
if __name__=='__main__':main()
