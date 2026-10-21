"""Deliver original ME038 weights and verify isolated restoration without fitting."""
import argparse,io,json,os,subprocess,sys,tarfile
from pathlib import Path
import scripts.me038_research as b


def compact():
    runtime=json.loads((b.OUT/'runtime.json').read_text());frozen=b.verify()
    fits={r['key']:r for r in frozen['fits']};members=[];models=[]
    for r in runtime['models']:
        key=r['key'];folder=b.OUT/key;alias='models/'+key
        for path,target in [(b.ROOT/r['physical_path'],alias+'.jit.pt'),
            (folder/'serving-input.npy',alias+'-input.npy'),(folder/'serving-output.npy',alias+'-expected.npy')]:
            members.append((path,target))
        models.append({'key':key,'model':alias+'.jit.pt','input':alias+'-input.npy','expected':alias+'-expected.npy',
            'selected_checkpoint':fits[key]['selected'],'dataset':r['dataset'],'horizon':r['horizon'],'seed':r['seed']})
    for path in ('data.json','PROTOCOL.md','MATHEMATICS.md','RESULTS.md','sources.json'):
        members.append((b.REPORT/path,path))
    members.extend([(b.old.DATA/'ETT-LICENSE','ETT-LICENSE'),(b.old.DATA/'TSL-README.md','TSL-SOURCE-ATTRIBUTION.md'),
        (b.DATA/'publisher.html','UCI-SOURCE-LICENSE.html')])
    index={'models':models,'source_snapshot_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'input_contract':'Raw publisher units Bx96xC in data.json channel order; output BxHxC in same units.',
        'versions':{'python':'3.12.14','torch':'2.14.1+cpu','numpy':'2.3.5'},
        'payloads':[{'path':alias,'bytes':p.stat().st_size,'sha256':b.sha(p)} for p,alias in members]}
    verifier='''import hashlib,json,numpy as np,torch
from pathlib import Path
torch.set_num_threads(1)
r=Path(__file__).resolve().parent;m=json.loads((r/'INDEX.json').read_text())
for p in m['payloads']:
 f=r/p['path'];assert f.stat().st_size==p['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==p['sha256']
count=0
for row in m['models']:
 model=torch.jit.load(str(r/row['model'])).eval()
 with torch.no_grad():y=model(torch.from_numpy(np.load(r/row['input']))).numpy()
 assert np.array_equal(y,np.load(r/row['expected'])),row['key'];count+=y.size
print('PASS',len(m['models']),'native models;',count,'physical forecast values bit exact; no TAC package or refit')
'''
    readme='''# ME038 trained native forecasters

60 original validation-selected native models: 48 models across six datasets, four forecast horizons and two seeds in the
exposure lane, plus twelve separate compute-budget models. Whole 96-step histories are required for each forecast.
Input/output use raw publisher units, with channel order in data.json. Horizons
are dataset steps, not universally hours. Tetouan models have eight channels; the
last three Tetouan channels are zone power; first five are recorded weather.

    import torch
    model = torch.jit.load("models/exposure-ETTh1-h96-innovation_tac-s38001.jit.pt").eval()
    with torch.no_grad(): forecast = model(raw_history)  # B x 96 x 7

Run `python verify.py` with Torch and NumPy to hash all payloads and reproduce
bundled physical-unit validation probes exactly, without importing TAC or training.
Versions are in INDEX.json; CPU float32 reproducibility is scoped to this environment.
Training-only anchors have been removed; learned weights are unchanged.

See RESULTS.md for all outcomes, including every registered power gate outcome,
separate power-channel errors, prior benchmark exposure and unequal actual compute.
These are research forecasting components, not qualified action controllers,
causal world models, streaming deployments or commercial systems. The complete
archive also contains all 1,008 parent/adaptation checkpoints, optimizer/RNG states,
original sources and training traces. No published-model or compute-parity claim.
'''
    dest=Path('/workspace/me038-native-inference-models.tar.gz')
    with tarfile.open(dest,'w:gz',compresslevel=3) as tar:
        for p,alias in members:tar.add(p,arcname=alias,recursive=False)
        for name,content in [('INDEX.json',json.dumps(index,indent=2)),('verify.py',verifier),('README.md',readme)]:
            payload=content.encode();item=tarfile.TarInfo(name);item.size=len(payload);tar.addfile(item,io.BytesIO(payload))
    receipt={'path':str(dest),'bytes':dest.stat().st_size,'sha256':b.sha(dest),'native_models':len(models),
        'original_weights_no_refit':True,'raw_CSVs_included':False,'public_GitHub_Actions_artifact':False}
    b.dump(b.REPORT/'inference-archive.json',receipt);print(json.dumps(receipt,indent=2))


def full():
    frozen=b.verify();assert json.loads((b.OUT/'audit.json').read_text())['status']=='PASS'
    excluded={'ARTIFACT.md','archive.json','archive-index.json','restoration.json','restoration.log',
        'inference-restoration.json','inference-restoration.log'}
    paths=[p for p in subprocess.check_output(['git','ls-files'],text=True).splitlines()
        if (b.ROOT/p).is_file() and not (p.startswith('research/mathematical-engineering/me038/') and Path(p).name in excluded)]
    for folder in (b.OUT,b.DATA,b.old.DATA):
        paths.extend(str(p.relative_to(b.ROOT)) for p in folder.rglob('*') if p.is_file() and not p.name.endswith('.tmp'))
    # Preserve original AR parents, including all six selection endpoints, without
    # unnecessarily copying the other old ME036 experimental trajectories.
    for parent in json.loads((b.OUT/'parents.json').read_text())['parents']:
        for e in parent['endpoints']:
            p=b.ROOT/e['checkpoint'];assert b.sha(p)==e['sha256']
            paths.extend([str(p.relative_to(b.ROOT)),str((p.parent/f"u{e['step']}-probe.npy").relative_to(b.ROOT))])
    paths=sorted(set(paths));head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    index={'source_snapshot_commit':head,'original_models_no_refit':True,'local_research_delivery':True,
        'payloads':[{'path':p,'bytes':(b.ROOT/p).stat().st_size,'sha256':b.sha(b.ROOT/p)} for p in paths]}
    b.dump(b.REPORT/'archive-index.json',index);dest=Path('/workspace/me038-models-and-evidence.tar.gz')
    with tarfile.open(dest,'w:gz',compresslevel=3) as tar:
        for p in paths:tar.add(b.ROOT/p,arcname=p,recursive=False)
        payload=json.dumps(index,indent=2).encode();item=tarfile.TarInfo('ARCHIVE-INDEX.json');item.size=len(payload);tar.addfile(item,io.BytesIO(payload))
    receipt={'path':str(dest),'bytes':dest.stat().st_size,'sha256':b.sha(dest),'indexed_payloads':len(paths),
        'source_snapshot_commit':head,'original_parent_checkpoints':288,'original_adaptation_checkpoints':720,
        'native_selected_models':60,'public_GitHub_Actions_artifact':False}
    b.dump(b.REPORT/'archive.json',receipt);print(json.dumps(receipt,indent=2))


def restore(archive,dest):
    if dest.exists():raise FileExistsError(dest)
    receipt=json.loads((b.REPORT/'archive.json').read_text());assert b.sha(archive)==receipt['sha256']
    dest.mkdir(parents=True)
    with tarfile.open(archive,'r:gz') as tar:tar.extractall(dest,filter='data')
    index=json.loads((dest/'ARCHIVE-INDEX.json').read_text())
    for row in index['payloads']:
        p=dest/row['path'];assert p.stat().st_size==row['bytes'] and b.sha(p)==row['sha256'],row['path']
    env=dict(os.environ);env.update({'PYTHONPATH':str(dest/'src')+':'+str(dest),
        'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
    with (b.REPORT/'restoration.log').open('w') as log:
        subprocess.run([sys.executable,str(dest/'scripts/me038_audit.py'),'audit'],cwd=dest,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    for row in index['payloads']:assert b.sha(dest/row['path'])==row['sha256'],row['path']
    replay=json.loads((dest/'results/me038/audit.json').read_text())
    result={'status':'OFFLINE_RESTORE_PASS','archive_sha256':b.sha(archive),'indexed_payloads':len(index['payloads']),
        'original_checkpoints_replayed':replay['original_checkpoints_replayed'],
        'original_validation_values_bit_exact':replay['bit_exact_validation_forecast_values'],
        'no_model_refit':True,'test_metrics_recomputed':False,'same_CPU_environment':True,'destination':str(dest)}
    b.dump(b.REPORT/'restoration.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('compact','full','restore'));p.add_argument('--archive',type=Path);p.add_argument('--destination',type=Path)
    args=p.parse_args()
    if args.mode=='restore':restore(args.archive,args.destination)
    else:globals()[args.mode]()
