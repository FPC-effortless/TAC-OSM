"""Run registered downstream audits and next study; no gate or budget relaxation."""
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]


def call(script,*args):
    print('START',script,' '.join(args),flush=True);subprocess.run([sys.executable,str(ROOT/'scripts'/script),*args],cwd=ROOT,check=True);print('DONE',script,flush=True)


def finish(study):
    suffix='identity_dev0615' if study=='dev0615' else 'belief_dev062';root='results/'+study+'-main';exports='results/'+study+'-agents'
    call('analyze_'+suffix+'.py','--directory',root,'--output','research/'+study+'/statistics.json')
    call('audit_'+suffix+'.py','--directory',root,'--output','research/'+study+'/audit.json','--veritas','/workspace/veritas','--exports',exports)
    call('benchmark_agents_dev062.py','--directory',root,'--family','identity' if study=='dev0615' else 'belief','--output','research/'+study+'/hardware.json')
    call('package_research_dev062.py','--study',study,'--run',root,'--exports',exports,'--smoke','results/'+study+'-smoke')
    call('report_agents_dev062.py','--study',study,'--directory',root)


def main():
    root=ROOT/'results/dev0615-main';deadline=time.monotonic()+3*3600
    while True:
        path=root/'summary.json';summary=json.loads(path.read_text()) if path.exists() else {}
        if summary.get('status')=='INCOMPLETE':raise RuntimeError('DEV061.5 matrix incomplete: heldout evaluation and promotion prohibited')
        if summary.get('status')=='COMPLETE' and (root/'manifest.json').exists():break
        if time.monotonic()>deadline:raise RuntimeError('Research coordinator timeout; do not extend registered fitting cap')
        time.sleep(10)
    finish('dev0615')
    call('run_belief_dev062.py','--directory','results/dev062-main')
    summary=json.loads((ROOT/'results/dev062-main/summary.json').read_text())
    if summary['status']!='COMPLETE':raise RuntimeError('DEV062A matrix incomplete; no test promotion')
    finish('dev062')
    print('REGISTERED RESEARCH AND AUDITS COMPLETE; commercial and cross-host gates remain separate.',flush=True)


if __name__=='__main__':main()
