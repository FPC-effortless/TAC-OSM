"""Fixed two-process seed execution; no model/score selection or extra fits."""
import argparse,subprocess,sys,time,json,os
from pathlib import Path
import scripts.me093_train_world as R


def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,default=Path('/tmp/me077-original-source'));args=p.parse_args()
    R.configure('me093');R.integrity()
    if list(R.OUT.glob('seed-*/final.pt')):raise RuntimeError('Never restart original seed portfolio')
    pending=list(R.SEEDS);active=[];completed=[]
    while pending or active:
        while pending and len(active)<2:
            seed=pending.pop(0);folder=R.OUT/f'seed-{seed}';folder.mkdir(parents=True,exist_ok=True)
            stream=(folder/'ORIGINAL_FIT_LOG.jsonl').open('w')
            proc=subprocess.Popen([sys.executable,'scripts/me093_train_world.py','fit','--study','me093','--seed',str(seed),'--cache',str(args.cache)],stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONPATH='src:.'))
            active.append((seed,proc,stream));print(json.dumps({'started_seed':seed,'maximum_workers':2}),flush=True)
        for item in list(active):
            seed,proc,stream=item
            if proc.poll() is None:continue
            stream.close();active.remove(item)
            if proc.returncode:
                # No retry; finish already running originals and retain their logs.
                for _,other,other_stream in active:other.wait();other_stream.close()
                raise RuntimeError(f'Original seed failed; no retry: {seed}')
            completed.append(seed);print(json.dumps({'completed_seed':seed,'completed':len(completed)}),flush=True)
        if active:time.sleep(1)
    if sorted(completed)!=list(R.SEEDS):raise RuntimeError('Incomplete original portfolio')
    index=[]
    for seed in R.SEEDS:
        folder=R.OUT/f'seed-{seed}';receipt=json.loads((folder/'FIT_RECEIPT.json').read_text())
        if receipt['actual_steps']!=3072 or receipt['checkpoint_sha256']!=R.sha(folder/'final.pt'):raise RuntimeError('Actual original receipt differs')
        index.append(receipt)
    R.write(R.OUT/'FIT_INDEX.json',index)
    print('ALL5_ORIGINALS_COMPLETE_NO_REPORT_SCORED',flush=True)

if __name__=='__main__':main()
