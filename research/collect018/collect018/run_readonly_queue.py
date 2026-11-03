"""Bounded local analysis queue; waits for verified originals, never starts fits."""
import concurrent.futures,json,subprocess,sys,time
from pathlib import Path
from .common import ROOT,ARMS,SEEDS

def execute(folder):
    outcomes=[]
    for module,subdir in [('audit_exhaustive','audits'),('phase_ablation','ablations'),('localize_key_family','localization')]:
        dest=ROOT/'results'/subdir;dest.mkdir(exist_ok=True);out=dest/(folder.name+'.json')
        if out.exists():continue
        # These two already had the intervention running in a separate process.
        if module=='phase_ablation' and folder.name in ('phase_joint_18002','context_joint_18003'):continue
        with (dest/(folder.name+'.log')).open('w') as log:
            r=subprocess.run([sys.executable,'-m','collect018.'+module,'--run',str(folder),'--output',str(out)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        outcomes.append(dict(module=module,returncode=r.returncode))
        if r.returncode:
            (dest/(folder.name+'.retained_failure.json')).write_text(json.dumps(dict(status='FAILED_READONLY_RETAINED',module=module,model=folder.name,returncode=r.returncode,optimizer_updates=0),indent=2))
    return dict(model=folder.name,outcomes=outcomes)

def main():
    downloads=ROOT.parent/'collect018_downloads';expected={f'{a}_{s}' for a in ARMS for s in SEEDS};submitted=set();futures={};start=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
      while submitted!=expected or futures:
        if time.monotonic()-start>2100:raise TimeoutError('Original downloads/analysis queue exceeded35minute bound; originals and failures retained')
        registry=json.loads((downloads/'ARTIFACT_REGISTRY.json').read_text());available={r['folder'] for r in registry if r['crc_verified']}
        for name in sorted((available&expected)-submitted):futures[pool.submit(execute,downloads/name)]=name;submitted.add(name)
        for f in list(futures):
            if f.done():print(json.dumps(f.result()),flush=True);del futures[f]
        time.sleep(3)
    print(json.dumps(dict(status='READONLY_QUEUE_COMPLETE',models=len(submitted),optimizer_updates=0)),flush=True)
if __name__=='__main__':main()
