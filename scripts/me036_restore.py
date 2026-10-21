"""Verify/extract ME036 archive then replay checkpoint probes without refitting."""
import argparse, hashlib, json, os, subprocess, sys, tarfile, time
from pathlib import Path


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def run(archive,destination):
    start=time.process_time()
    if destination.exists():raise FileExistsError(destination)
    destination.mkdir(parents=True)
    with tarfile.open(archive,'r:gz') as tar:tar.extractall(destination,filter='data')
    index=json.loads((destination/'ARCHIVE-INDEX.json').read_text())
    for row in index['payloads']:
        p=destination/row['path'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
    env=dict(os.environ);env.update({'PYTHONPATH':str(destination/'src')+':'+str(destination),
        'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
    with open(destination/'offline-restoration.log','w') as log:
        for script in ('me036_research.py','me036b_continue.py'):
            subprocess.run([sys.executable,str(destination/'scripts'/script),'audit'],cwd=destination,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    for row in index['payloads']:
        # Replayed receipts must reproduce exactly, as must every original payload.
        assert sha(destination/row['path'])==row['sha256'],row['path']
    a=json.loads((destination/'results/me036/replay.json').read_text())
    b=json.loads((destination/'results/me036b/replay.json').read_text())
    result={'status':'OFFLINE_RESTORE_PASS','archive_sha256':sha(archive),'indexed_payloads':len(index['payloads']),
        'original_checkpoints_replayed':b['checkpoints'],
        'original_scalar_validation_forecasts_replayed':b['bit_exact_scalar_forecasts'],
        'initial_stage_replayed_again':a['checkpoints'],'no_model_refit':True,
        'test_metrics_recomputed':False,'same_CPU_environment':True,'destination':str(destination)}
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('destination',type=Path)
    args=p.parse_args();run(args.archive,args.destination)
