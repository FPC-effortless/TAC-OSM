"""Compact, package-independent physical-unit native model delivery."""
import io,json,subprocess,tarfile
from pathlib import Path
import scripts.me036_research as b


def run():
    out=b.ROOT/'results/me036b';runtime=json.loads((out/'runtime.json').read_text())
    frozen=json.loads((out/'frozen.json').read_text())
    selections={r['key']:r['selected'] for r in frozen['fits'] if r['arm']=='tac_patch'}
    members=[];models=[]
    for r in runtime['exports']:
        key=r['key'];folder=out/key;alias='models/'+key
        for original,target in [(b.ROOT/r['physical_path'],alias+'.jit.pt'),
            (folder/'physical-serving-input.npy',alias+'-input.npy'),
            (folder/'physical-serving-output.npy',alias+'-expected.npy')]:
            members.append((original,target))
        models.append({'key':key,'dataset':r['dataset'],'horizon':r['horizon'],'seed':r['seed'],
            'model':alias+'.jit.pt','input':alias+'-input.npy','expected':alias+'-expected.npy',
            'model_sha256':r['physical_sha256'],'selected_checkpoint':selections[key],
            'state':'Original selected trained weights; physical normalization wrapper; no refit.'})
    members += [(b.OUT/'data.json','data.json'),(b.REPORT/'PROTOCOL.md','PROTOCOL.md'),
        (b.REPORT/'CONTINUATION.md','CONTINUATION.md'),(b.REPORT/'RESULTS.md','RESULTS.md'),
        (b.DATA/'ETT-LICENSE','ETT-LICENSE'),(b.DATA/'TSL-README.md','SOURCE-ATTRIBUTION.md')]
    index={'models':models,'input_contract':'Raw publisher values, Bx96xC, channel order from data.json. Output BxHxC in the same physical units.',
        'versions':{'python':'3.12','torch':'2.14.1+cpu','numpy':'2.3.5 (verification only)'},
        'source_snapshot_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'payloads':[{'path':alias,'sha256':b.sha(path),'bytes':path.stat().st_size} for path,alias in members]}
    verify='''import hashlib,json,numpy as np,torch
from pathlib import Path
torch.set_num_threads(1)
r=Path(__file__).resolve().parent; m=json.loads((r/'INDEX.json').read_text())
for row in m['payloads']:
 p=r/row['path'];assert p.stat().st_size==row['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
count=0
for row in m['models']:
 model=torch.jit.load(str(r/row['model'])).eval()
 with torch.no_grad(): y=model(torch.from_numpy(np.load(r/row['input']))).numpy()
 expected=np.load(r/row['expected']);assert np.array_equal(y,expected),row['key'];count+=y.size
print('PASS',len(m['models']),'native models;',count,'physical forecast values bit exact; no refit')
'''
    readme='''# ME036 selected native forecasters

40 original selected trained TAC models: five datasets, four full forecast horizons,
two seeds. The files include the native learned fast/slow memory and its joint AR
prediction path. These are complete forecasting systems, not a separately qualified
causal world model. Forecasts require the whole96-step input window.

Load with Torch only, no TAC package needed:

    import torch
    model = torch.jit.load("models/ETTh1-h96-tac_patch-s36001.jit.pt").eval()
    with torch.no_grad(): forecast = model(raw_history)  # B x 96 x 7

Input/output use original publisher units. Preserve the channel order in data.json.
Electricity has321channels, Weather21, ETT7. Horizons are measured in dataset steps,
not universally hours. Choose the matching dataset/horizon checkpoint.

Run `python verify.py` to hash every payload and replay the bundled physical-input
probes exactly, using Torch and NumPy. The weights come directly from the saved
trained checkpoint selected before final test evaluation. INDEX.json records its
original checkpoint path/hash/step and each executable model hash. No retraining
is needed. The full research archive additionally contains all1680checkpoints,
optimizer/RNG states, baselines, source CSVs and logs. This compact delivery excludes
raw CSVs and does not reproduce the entire final test metric matrix.

See RESULTS.md for actual accuracy and the limits of the compact baselines,
unmatched compute, prior ETT exposure, Weather source anomalies and ridge failures.
This is local research delivery, not a commercial or model-promotion release.
'''
    destination=Path('/workspace/me036-native-inference-models.tar.gz')
    with tarfile.open(destination,'w:gz',compresslevel=3) as tar:
        for path,alias in members:tar.add(path,arcname=alias,recursive=False)
        for name,data in [('INDEX.json',json.dumps(index,indent=2).encode()),('verify.py',verify.encode()),('README.md',readme.encode())]:
            item=tarfile.TarInfo(name);item.size=len(data);tar.addfile(item,io.BytesIO(data))
    receipt={'path':str(destination),'bytes':destination.stat().st_size,'sha256':b.sha(destination),
        'native_models':len(models),'original_trained_weights_no_refit':True,'raw_CSVs_included':False}
    b.dump(b.REPORT/'inference-archive.json',receipt);print(json.dumps(receipt,indent=2))


if __name__=='__main__':run()
