"""Continue into registered external data after identity/belief research audits."""
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]


def call(script,*args):
    print('START',script,flush=True);subprocess.run([sys.executable,str(ROOT/'scripts'/script),*args],cwd=ROOT,check=True);print('DONE',script,flush=True)


def main():
    deadline=time.monotonic()+3*3600
    while not (ROOT/'research/dev062/RESULTS.md').exists():
        if time.monotonic()>deadline:raise RuntimeError('Previous registered research did not finish; no silent extension')
        time.sleep(10)
    base='/workspace/scratch/maintenance-source-audit';call('run_real_maintenance_001.py','--directory','results/real-maint-001-main','--csv',base+'/metropt3/MetroPT3(AirCompressor).csv','--source-archive',base+'/metropt3-complete.zip')
    summary=json.loads((ROOT/'results/real-maint-001-main/summary.json').read_text())
    if summary['status']!='COMPLETE':raise RuntimeError('External-study matrix incomplete; no test promotion')
    call('analyze_real_maintenance_001.py','--directory','results/real-maint-001-main','--output','research/real-maint-001/statistics.json')
    call('audit_real_maintenance_001.py','--directory','results/real-maint-001-main','--output','research/real-maint-001/audit.json','--exports','results/real-maint-001-agents')
    call('package_research_dev062.py','--study','real-maint-001','--run','results/real-maint-001-main','--exports','results/real-maint-001-agents','--smoke','results/real-maint-001-smoke-v2')
    r=json.loads((ROOT/'research/real-maint-001/statistics.json').read_text());lines=['# REAL-MAINT-001 — external operational sensor forecasts','','**Real data evaluated; commercial repair-policy validation remains unavailable.**','',f"Fitted {r['run']['fits']} fresh models and {r['run']['pilot_fits']} validation-only pilot models. All twenty training seeds retained per arm. CPU fitting/preparation: {r['run']['fitting_cpu_seconds']/60:.2f} minutes.",'','| Arm | Mean normalized MSE | Mean Gaussian NLL |','|---|---:|---:|']
    for a,v in r['arms'].items():lines.append(f"| {a} | {v['mean_normalized_MSE']:.6f} | {v['mean_Gaussian_NLL']:.6f} |")
    lines+=['',f"Persistence normalized MSE: {r['baselines']['persistence_normalized_MSE']:.6f}. Training-mean MSE: {r['baselines']['train_mean_normalized_MSE']:.6f}.",'','## Registered gates','']
    for k,v in r['gates'].items():lines.append(f'- {k}: **{v}**')
    lines+=['','One real compressor and four reported fault windows do not provide twenty independent fleets or repair trials. Outside reported fault windows, health is unknown. A May maintenance-date inconsistency is preserved. Models train only on next observed sensors; no simulator teacher or fault labels are used for fitting. Gaussian marginal assumptions include binary sensors.','', 'The exported seed92701 was selected before outcomes. The archive includes a PyTorch-only runtime with train-only normalization and next-sensor mean/std output. It does not issue maintenance actions or certify repair success. No cost saving is inferred from unmeasured inspection prices or counterfactual interventions.','', 'See [registration](PROTOCOL.md), [complete statistics](statistics.json), [strict replay and CPU measurements](audit.json), [source download hash](source-download.json), and [complete evidence index](evidence-index.json). Source: [MetroPT-3, UCI](https://archive.ics.uci.edu/dataset/791/metropt+3+dataset), DOI10.24432/C5VW3R, CC BY4.0.']
    (ROOT/'research/real-maint-001/RESULTS.md').write_text('\n'.join(lines)+'\n');print('EXTERNAL RESEARCH COMPLETE; commercial causal trial remains a separate gate.',flush=True)


if __name__=='__main__':main()
