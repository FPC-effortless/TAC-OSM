"""Read-only processing of all native-attention original artifacts."""
import argparse,json,subprocess,sys
from pathlib import Path
from .common import ROOT,C16,require,internal_gate
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    internal_gate(json.loads((ROOT/'READONLY_DIAGNOSTIC_MANIFEST.json').read_text()))
    plan=json.loads((ROOT/'results/READONLY_PLAN.json').read_text())['plan']
    require(plan['state']=='READY_INTERNAL_RESEARCH_PLAN' and plan['backend']=='github_actions','read-only CPU plan failed')
    r=json.loads((a.run/'FIT_RECEIPT.json').read_text());name=f"{r['arm']}_{r['seed']}";dest=ROOT/'results'/name;dest.mkdir(exist_ok=True)
    jobs=[('audit_completed',['--output',str(ROOT/'results'/('audit_'+name+'.json'))]),('supplemental',['--output',str(dest/'SUPPLEMENTAL.json')]),('generate',['--output',str(dest/'GENERATION.json')]),('gradient_profile',['--output',str(dest/'GRADIENT_PROFILE.json')])]
    for module,extra in jobs:
        with (dest/(module+'.log')).open('w') as log:
            result=subprocess.run([sys.executable,'-m','collect017.'+module,'--run',str(a.run.resolve()),*extra],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        require(result.returncode==0,'read-only stage failed: '+module+'; original evidence retained')
        print(json.dumps(dict(run=name,stage=module,status='COMPLETE_READ_ONLY')),flush=True)
    with (dest/'position_and_train.log').open('w') as log:
        result=subprocess.run([sys.executable,'-m','collect016.position_and_train','--attention','--run',str(a.run.resolve()),'--output',str(dest/'POSITION_AND_TRAIN.json')],cwd=C16,stdout=log,stderr=subprocess.STDOUT)
    require(result.returncode==0,'post-null position check failed; original evidence retained')
    (dest/'PROCESS_RECEIPT.json').write_text(json.dumps(dict(status='COMPLETE_READ_ONLY_ORIGINAL_AUDIT',arm=r['arm'],seed=r['seed'],optimizer_updates=0),indent=2))
if __name__=='__main__':main()
