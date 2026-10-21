"""Restore committed engineering records and recompute numerical agreement."""
import json,os,subprocess,sys,tarfile
from pathlib import Path
import numpy as np
import torch
import scripts.me039_research as b


def run():
    torch.set_num_threads(1);destination=Path('/workspace/me039-records-restored')
    if destination.exists():raise FileExistsError(destination)
    destination.mkdir();receipts=json.loads((b.ROOT/'research/model-artifacts/me039-archives.json').read_text())['archives'];payloads=0
    for row in receipts:
        archive=b.ROOT/row['path'];assert b.sha(archive)==row['sha256']
        with tarfile.open(archive) as tar:
            index=json.load(tar.extractfile('INDEX.json'));tar.extractall(destination,filter='data')
        for p in index['payloads']:
            path=destination/p['path'];assert path.stat().st_size==p['bytes'] and b.sha(path)==p['sha256']
        payloads+=len(index['payloads'])
    result=json.loads((b.OUT/'training.json').read_text());records={r['key']:r for r in result['records']}
    optimizer_states=0;comparisons=0
    for r in records.values():
        state=torch.load(destination/r['final_checkpoint'],weights_only=False)
        assert state['engineering_only_continuation'] and state['step']==64
        assert all(float(v['step'])==64 for v in state['optimizer']['state'].values())
        snapshots={s['step']:s for s in r['snapshots']}
        flat=np.concatenate([state['state_dict'][d['name']].numpy().ravel() for d in r['parameter_offsets']])
        assert np.array_equal(flat,np.load(destination/snapshots[64]['path'])['parameters'])
        optimizer_states+=1
    for pair in result['pairs']:
        ra=records[pair['eager']];rb=records[pair['scripted']]
        mask=np.concatenate([np.full(v['end']-v['start'],v['name'].startswith('memory.')) for v in ra['parameter_offsets']])
        for snapshot in pair['deviations']:
            step=snapshot['step'];a=np.load(destination/next(s['path'] for s in ra['snapshots'] if s['step']==step))
            c=np.load(destination/next(s['path'] for s in rb['snapshots'] if s['step']==step))
            ga=a['gradient'];gc=c['gradient'];error=ga.astype(np.float64)-gc
            relative=np.linalg.norm(error)/max(np.linalg.norm(ga.astype(np.float64)),1e-12)
            core=np.linalg.norm(error[mask])/max(np.linalg.norm(ga[mask].astype(np.float64)),1e-12)
            assert relative==snapshot['gradient_relative_L2'] and core==snapshot['memory_gradient_relative_L2']
            assert relative<=result['roundoff_tolerance'] and core<=result['roundoff_tolerance']
            comparisons+=1
    code='''import json,numpy as np,torch,sys
from pathlib import Path
torch.set_num_threads(1);r=Path(sys.argv[1]);meta=json.loads(Path(sys.argv[2]).read_text());count=0
for row in meta['models']:
 p=r/row['path'];m=torch.jit.load(str(p)).eval();folder=p.parent
 with torch.no_grad():y=m(torch.from_numpy(np.load(folder/'input.npy'))).numpy()
 assert np.array_equal(y,np.load(folder/'output.npy')),row['key'];count+=y.size
print('PASS',len(meta['models']),'experimental exports;',count,'physical forecast values exact; Torch-only')
'''
    env=dict(os.environ);env.pop('PYTHONPATH',None)
    out=subprocess.check_output([sys.executable,'-c',code,str(destination),str(b.OUT/'frozen-audit.json')],cwd='/tmp',env=env,text=True)
    receipt={'status':'RESTORED_AND_AUDITED','indexed_payloads':payloads,'original_optimizer_continuations':optimizer_states,
        'gradient_snapshot_pairs_recomputed':comparisons,'standalone_export_replay':out.strip(),'no_refit_or_test_scoring':True,
        'same_CPU_environment':True,'destination':str(destination)}
    b.dump(b.REPORT/'restoration.json',receipt);print(json.dumps(receipt,indent=2))

if __name__=='__main__':run()
