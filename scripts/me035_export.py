"""Export selected trained models with their training-only physical normalizer."""
import csv,itertools,json,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
import torch
from tac_osm.me035_forecast_runtime import PhysicalForecast
from scripts.me035_audit import verify
from scripts.me035_architecture_research import source,COLUMNS,build_model,file_sha,write_json


def run(out):
    torch.set_num_threads(1);r=verify(out);_,dev,meta=source(out)
    choice=r['evaluation']['selected_development_candidate']
    with (out/'ETTh1.csv').open() as f:
        rows=np.array([[float(row[c]) for c in COLUMNS] for row in itertools.islice(csv.DictReader(f),11640)],dtype=np.float32)
    raw=torch.from_numpy(rows[dev['indices'][:,None]+np.arange(-95,1)[None,:]].copy())
    mean=torch.tensor(meta['mean']);scale=torch.tensor(meta['scale']);receipts=[]
    for fit in r['fits']:
        if fit['arm']!=choice['arm']:continue
        ck=next(c for c in fit['checkpoints'] if c['step']==choice['step'])
        saved=torch.load(out/(ck['name']+'.pt'),weights_only=True)
        net=build_model(fit['arm'],fit['seed']);net.load_state_dict(saved['state_dict'],strict=True)
        pipeline=torch.jit.script(PhysicalForecast(net,mean,scale)).eval()
        name='physical-'+ck['name']+'.jit.pt';pipeline.save(str(out/name))
        expected=np.load(out/(ck['name']+'-predictions.npy'))*scale.numpy()+mean.numpy()
        with torch.no_grad():actual=torch.cat([pipeline(raw[i:i+256]) for i in range(0,len(raw),256)]).numpy()
        np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-4)
        with tempfile.TemporaryDirectory() as folder:
            tmp=Path(folder);torch.save(raw[:4],tmp/'x.pt')
            code='import sys,torch,numpy as np\nassert not any(n.startswith(("tac_osm","scripts")) for n in sys.modules)\ntorch.set_num_threads(1)\nm=torch.jit.load(sys.argv[1]).eval();x=torch.load(sys.argv[2],weights_only=True)\nwith torch.no_grad():p=m(x)\nnp.save(sys.argv[3],p.numpy())\n'
            subprocess.run([sys.executable,'-c',code,str((out/name).resolve()),str(tmp/'x.pt'),str(tmp/'p.npy')],cwd=tmp,check=True,capture_output=True)
            np.testing.assert_allclose(np.load(tmp/'p.npy'),expected[:4],rtol=0,atol=1e-4)
        receipts.append({'name':name,'sha256':file_sha(out/name),'seed':fit['seed'],'forecast_values':actual.size,
            'max_physical_unit_error':float(np.max(abs(actual-expected))),'fresh_Torch_only_values':84,
            'trainable_weights_original':18924,'new_fits':0})
    result={'status':'PASS','selected_stage':choice,'input':'B x96x7 raw publisher units, source column order',
        'output':'B x3x7 raw publisher units, horizons1/6/24hours','models':receipts}
    write_json(out/'physical-runtime-exports.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('results/me035'));a=p.parse_args();run(a.out)
