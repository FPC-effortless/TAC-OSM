"""Conditional fresh-data follow-up; never extend or tune a completed matrix."""
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
    while not (ROOT/'research/real-maint-001/RESULTS.md').exists():
        if time.monotonic()>deadline:raise RuntimeError('Previous registered studies incomplete; no silent extension')
        time.sleep(10)
    current=json.loads((ROOT/'research/dev062/statistics.json').read_text());names=('18_of20_95_completion','median_KL_002','median_Brier_excess_002','zero_primary_faults');trigger=not all(current['gates'][k] is True for k in names);decision={'run_followup':trigger,'original_functional_gates':{k:current['gates'][k] for k in names},'rule':'Registered conditional follow-up, fresh namespaces; no effect on original evidence'};(ROOT/'research/dev062-r1/trigger.json').write_text(json.dumps(decision,indent=2)+'\n')
    if not trigger:print('Original functional gates pass; extra fitting SKIPPED.',flush=True);return
    call('run_belief_dev062_r1.py','--directory','results/dev062-r1-main');summary=json.loads((ROOT/'results/dev062-r1-main/summary.json').read_text())
    if summary['status']!='COMPLETE':raise RuntimeError('Follow-up matrix incomplete; no test promotion')
    call('analyze_belief_dev062_r1.py','--directory','results/dev062-r1-main','--output','research/dev062-r1/statistics.json')
    call('audit_belief_dev062_r1.py','--directory','results/dev062-r1-main','--output','research/dev062-r1/audit.json','--exports','results/dev062-r1-agents','--veritas','/workspace/veritas')
    call('benchmark_agents_dev062.py','--directory','results/dev062-r1-main','--family','belief','--output','research/dev062-r1/hardware.json')
    call('package_research_dev062.py','--study','dev062-r1','--run','results/dev062-r1-main','--exports','results/dev062-r1-agents','--smoke','results/dev062-r1-smoke')
    call('report_agents_dev062.py','--study','dev062-r1','--directory','results/dev062-r1-main')
    print('CONDITIONAL FOLLOW-UP COMPLETE; no further automatic fitting.',flush=True)


if __name__=='__main__':main()
