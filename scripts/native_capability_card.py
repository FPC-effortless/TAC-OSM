"""Derive a truthful capability card/report from frozen original study scores.

Formatting only: numerical gates remain owned by each registered report script.
Missing independent original replay is BLOCKED, never a provisional PASS.
"""
import sys,json,copy,hashlib
from pathlib import Path
study=sys.argv[1];d=Path('research/mathematical-engineering')/study
r=json.loads((d/'CAPABILITY_RESULTS.json').read_text());reg=json.loads((d/'FAIR_COMPARISON.json').read_text());m=r['native_candidate'];seed=reg['models'][0]['seed'];out=Path('research/replay-bundles')/study.upper()
c=json.loads(Path('research/mathematical-engineering/me081/CAPABILITY_SCORECARD.json').read_text())
c['study_id']=reg['study_id'];c['tier']='BLOCKED';c['numeric_result_tier']=r['tier'];c['primary_comparison'].update(candidate=m['1']['all']['mae'],paired_case_cluster_95ci_new_minus_old=r['paired_episode95CI_H1new_minus_ME080'],beneficial_seed_count=r['paired_seed_H1wins_vs_ME080'])
c['guardrails']=[{'metric':k,'status':'PASS' if v else 'FAIL'} for k,v in r['gates'].items()]
c['model'].update(candidate_sha256=hashlib.sha256((out/f'seed-{seed}/final.pt').read_bytes()).hexdigest(),trainable_parameters=reg["models"][0]["trainable_parameters"],deployed_parameters=reg["models"][0]["deployed_parameters"],training_exposure_and_compute_note=reg['training_budget']['compute'])
c['integrity']['raw_case_score_artifact_path']=str(out)
p=d/'EXTERNAL_ORIGINAL_REPLAY.json'
if p.exists():
 a=json.loads(p.read_text());c['tier']=r['tier'] if a['strict_original_contract']=='PASS' else 'BLOCKED';c['integrity'].update(independent_raw_case_audit=a['strict_original_contract']=='PASS',external_replay_status=a['strict_original_contract'],independent_clean_replay=a['strict_original_contract']=='PASS',independent_audit_run_url=a['github_run'])
else:c['integrity'].update(external_replay_status='PENDING',independent_clean_replay=False,independent_raw_case_audit=False,independent_audit_run_url='NOT_YET_RECORDED')
c['what_improved']='Actual primary8frame and secondary32frame frozen whole-model forecasts are fully retained. H1/H4/H16mean MAE '+ '/'.join(f"{m[h]['all']['mae']:.8f}" for h in ['1','4','16'])+'. See proper uncertainty, motion and quiet scores; no general-intelligence implication.'
c['what_regressed']='Fixed full predictive gate '+r['full_registered_gate']+'; failed conditions: '+','.join(k for k,v in r['gates'].items() if not v)
c['what_remains_unknown']='Independent new-source qualification, arbitrary-history accuracy, causal action intelligence, architecture superiority, enterprise/operator/customer value, general intelligence and accelerated variable-shape inference.'
(d/'CAPABILITY_SCORECARD.json').write_text(json.dumps(c,indent=2,sort_keys=True)+'\n')
failed=[k for k,v in r['gates'].items() if not v]
s=f'''# {study.upper()} — frozen complete native capability result

**Full predictive gate {r['full_registered_gate']} / {r['tier']}.** Exposed development only.
No extra fits under this completed registration. Numerical gates and independent replay are separate conditions. All five originals were Git-frozen
before REPORT scoring; no best-checkpoint or ensemble selection.

| Same original8frame task | MAE | Coverage | Interval score | Physical marginal NLL |
|---|---:|---:|---:|---:|
'''
for h in ['1','4','16']:
 v=m[h]['all'];s+=f"| {h} retained frames | {v['mae']:.8f} | {v['coverage']:.6f} | {v['interval_score']:.8f} | {v['physical_marginal_nll']:.8f} |\n"
s+=f"\nH1actual-moving MAE{m['1']['moving']['mae']:.8f}; H1near-stationary MAE{m['1']['stationary']['mae']:.8f}. Forty exposed recording episodes,960primary H1origins,2844total primary future pairs; both history readouts are preserved. Five seeds do not add independent episodes. Original95%paired episode CI(new−ME080): {r['paired_episode95CI_H1new_minus_ME080']}.\n\nFailed gates: "+(', '.join(failed) or 'none in numerical predictive scorecard; independent original replay remains separately compulsory')+'.\n'
s+='\nIndependent original replay: '+c['integrity']['external_replay_status']+'. Delivered capability tier: '+c['tier']+'.\n'
s+='''
[Original complete scores](CAPABILITY_RESULTS.json), [actual guarded optimizer/parameter audit](FINAL_COMPARISON.json), [frozen registration](FAIR_COMPARISON.json), [source](SOURCE_QUALIFICATION.json). Original weights, optimizer/RNG, receipts and all point/distribution forecasts are tracked in the study replay folder. Same authenticated real recordings and past/current information only; future commands/capture times and protected tissue-box/CAL/TEST remain closed.

Reference inference uses the whole forward graph in float64, promoted from exact original float32 trained values. Independent restoration must retain original rtol1e-5/atol1e-6. Compiled fixed-profile engineering evidence, when present, is separate from full original forecast replay and predictive performance. No architecture-winner, policy control, general intelligence, independent site proof, deployed product or customer-value claim is unlocked.
'''
(d/'RESULTS.md').write_text(s)
