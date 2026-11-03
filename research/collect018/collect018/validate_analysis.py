"""Verify derived analysis coverage and unchanged original predictions, no fits."""
import json,math,statistics
from pathlib import Path
from .common import ROOT,ARMS,SEEDS,require,digest
from .collect_results import audit_predictions
from .common import verify_data

def main():
    data,_=verify_data();downloads=ROOT.parent/'collect018_downloads';spec=json.loads((ROOT/'EVALUATION_PROTOCOL.json').read_text());summary=json.loads((ROOT/'results/SUMMARY.json').read_text());require(len(summary['models'])==18 and not summary['missing'],'full coverage absent')
    references=[];interventions=[];localizations=[];all_hashes=[]
    for arm in ARMS:
      for seed in SEEDS:
        name=f'{arm}_{seed}';folder=downloads/name;receipt=json.loads((folder/'FIT_RECEIPT.json').read_text());p=ROOT/'results/audits'/(name+'.json');a=json.loads(p.read_text())
        require(a['status'] in ('PASS_TRAINED_SOURCE_AUDIT','RETAINED_SOURCE_PARITY_FAILURE_OTHER_CHECKS_PASS'),'unresolved audit mechanics')
        require(a['optimizer_updates_in_audit']==0 and a['original_thresholds_unchanged'] and a['source_exposure_verified']==1024,'audit scope differs')
        failures=a['retained_parity_failures'];require((not failures)==(a['status']=='PASS_TRAINED_SOURCE_AUDIT'),'failure status hidden')
        references.append(dict(model=name,status=a['status'],failure_labels=[f['label'] for f in failures],replay_max_abs=a['replay_max_abs'],references=a['trained_reference']))
        all_hashes.append(dict(path=str(p.relative_to(ROOT)),sha256=digest(p)))
        p=ROOT/'results/ablations'/(name+'.json');ab=json.loads(p.read_text());require(ab['optimizer_updates']==0,'ablation optimized')
        if arm.startswith('gru_'):require(ab['status']=='INCONCLUSIVE_NOT_APPLICABLE_NO_PHASE_STATE','GRU fake phase state');continue
        require(ab['checkpoint_sha256']==receipt['checkpoint_sha256'] and ab['weights_unchanged'] and not ab['primary_replacement'],'state intervention model differs')
        require(len(ab['rows'])==24*4*(3+6+6),'state intervention coverage differs')
        original=json.loads((folder/'RETRIEVAL.json').read_text());index={(r['keys'],r['gap'],r['history'],r['condition'],r['key']):r for r in original['rows']};maxerr=0.;argmax_changes=0
        for r in ab['rows']:
            y=bytes.fromhex(r['target_hex']);z=bytes.fromhex(r['prediction_hex']);logits=r['first_logits'];require(len(logits)==256 and all(math.isfinite(v) for v in logits),'ablation nonfinite logits')
            require(z[0]==max(range(256),key=logits.__getitem__) and r['first_correct']==int(y[0]==z[0]) and r['exact4']==int(y==z),'ablation raw score differs')
            if r['condition']=='normal':
                s=index[(r['keys'],r['gap'],r['history'],'normal',r['key'])];maxerr=max(maxerr,max(abs(x-y) for x,y in zip(s['first_logits'],logits)));argmax_changes+=r['prediction_hex']!=s['prediction_hex']
        for cell in ab['summary']:
            rs=[r for r in ab['rows'] if (r['keys'],r['gap'],r['condition'])==(cell['keys'],cell['gap'],cell['condition'])];pr=[r for r in rs if not r['absent']];aa=[r for r in rs if r['absent']]
            for k,v in dict(present_first=statistics.mean(r['first_correct'] for r in pr),present_exact4=statistics.mean(r['exact4'] for r in pr),absent_first=statistics.mean(r['first_correct'] for r in aa)).items():require(abs(v-cell[k])<1e-12,'ablation summary differs')
        interventions.append(dict(model=name,status='PASS_RAW_INTERVENTION_RECOMPUTATION',normal_logits_cross_runtime_max_abs=maxerr,normal_generated4_changed_count=argmax_changes,normal_logits_within_original_replay_tolerance=maxerr<=4e-5,scope='Read-only OOD state dependence, no architecture superiority'))
        all_hashes.append(dict(path=str(p.relative_to(ROOT)),sha256=digest(p)))
    for m in summary['models']:
        loc=m.get('post_null_key_family_localization');require(loc is not None,'localization missing');localizations.append(dict(model=f"{m['arm']}_{m['seed']}",first=[{k:r[k] for k in ('keys','condition','first_accuracy','exact4','absent_first')} for r in loc['summary']]))
    out=dict(status='PASS_ANALYSIS_COVERAGE_AND_RAW_SCORES_WITH_RETAINED_NUMERICAL_FAILURES',models=18,trained_references=references,phase_interventions=interventions,key_family_localizations=localizations,analysis_hashes=all_hashes,optimizer_updates=0,mathematical_source_IEEE_AD='OPEN',qualified=False)
    (ROOT/'results/ANALYSIS_VALIDATION.json').write_text(json.dumps(out,indent=2));print(json.dumps(dict(status=out['status'],models=18,interventions=len(interventions),normal_generation_changes=sum(r['normal_generated4_changed_count'] for r in interventions))))
if __name__=='__main__':main()
