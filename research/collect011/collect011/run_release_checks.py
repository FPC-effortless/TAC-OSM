"""Sequential release checks; measured CPU loops never overlap our other checks."""
import datetime,hashlib,json,subprocess,sys,time
from pathlib import Path

def main():
    root=Path(__file__).resolve().parents[1]
    sources={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
        for dirname in ('collect011','vendor','governance','math') for p in (root/dirname).rglob('*.py') if '__pycache__' not in p.parts}
    journal=dict(started=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_hashes=sources,optimizer_updates=0,test_access=False,steps=[])
    jobs=[('FINAL_GRAPH_BENCHMARK',['-m','collect011.benchmark','--lengths','256','--output','FINAL_GRAPH_BENCHMARK']),
          ('FINAL_HARD_SHORT_BENCHMARK',['-m','collect011.benchmark','--lengths','32','128','--arms','hybrid_hard_p','--output','FINAL_HARD_SHORT_BENCHMARK']),
          ('FINAL_GENERATION_HARD',['-m','collect011.generation_probe','--arms','hybrid_hard_p','--output','FINAL_GENERATION_HARD']),
          ('FINAL_PROBE',['-m','collect011.probe']),('FINAL_PREFLIGHT',['-m','collect011.prepare']),
          ('FINAL_MATH',['math/reference.py']),('FINAL_TESTS',['-m','unittest','collect011.test_integration','-v'])]
    for name,args in jobs:
        started=time.monotonic();command=[sys.executable,*args]
        with (root/f'results/{name}.log').open('w') as log:
            p=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
        row=dict(name=name,command=command,exit_code=p.returncode,seconds=time.monotonic()-started)
        journal['steps'].append(row)
        (root/'results/RELEASE_JOURNAL.json').write_text(json.dumps(journal,indent=2))
        print(json.dumps(row),flush=True)
        if p.returncode:raise SystemExit(p.returncode)
    assert all(hashlib.sha256((root/p).read_bytes()).hexdigest()==h for p,h in sources.items()),'Executable source changed during release checks'
    journal.update(finished=datetime.datetime.now(datetime.timezone.utc).isoformat(),executable_sources_unchanged=True)
    (root/'results/RELEASE_JOURNAL.json').write_text(json.dumps(journal,indent=2))

if __name__=='__main__':main()
