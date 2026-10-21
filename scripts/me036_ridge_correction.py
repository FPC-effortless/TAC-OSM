"""Correct the ridge fitting objective; preserve original registered reference."""
import json,time
import numpy as np
import torch
import scripts.me036_research as b
from scripts.me036b_continue import Ridge

OUT=b.ROOT/'results/me036c'


def train():
    records=[];OUT.mkdir(parents=True,exist_ok=True)
    for name in b.DATASETS:
        z,meta=b.load_data(name);train=meta['split_edges'][0]
        for h in b.HORIZONS:
            folder=OUT/f'{name}-h{h}';folder.mkdir(exist_ok=True)
            start=time.process_time();aa=np.zeros((97,97));ab=np.zeros((97,h))
            ix=np.arange(96,train-h+1,24)
            for i in range(0,len(ix),8):
                x,y=b.batch(z,ix[i:i+8],h)
                mean=x.mean(1,keepdim=True);scale=torch.sqrt((x-mean).square().mean(1,keepdim=True)+1e-5)
                # sigma*[normalized input;1] aligns squared residuals with global MSE.
                a=(x-mean).transpose(1,2).reshape(-1,96).double().numpy()
                sig=scale.transpose(1,2).reshape(-1,1).double().numpy()
                a=np.c_[a,sig]
                target=(y-mean).transpose(1,2).reshape(-1,h).double().numpy()
                aa+=a.T@a;ab+=a.T@target
            aa.flat[::98]+=100.;aa[-1,-1]-=100.
            coef=np.linalg.solve(aa,ab);path=folder/'coef.npy';np.save(path,coef)
            record={'dataset':name,'horizon':h,'parameters':coef.size,'ridge_alpha':100.,
                'coefficient':str(path.relative_to(b.ROOT)),'sha256':b.sha(path),
                'charged_training_CPU_seconds':time.process_time()-start,
                'training_anchors':len(ix),'status':'PASS','prospective_qualification':False}
            records.append(record);print('CORRECTED FIT',name,h,round(record['charged_training_CPU_seconds'],2),flush=True)
    b.dump(OUT/'frozen.json',{'protocol_sha256':b.sha(b.REPORT/'RIDGE-CORRECTION.md'),
        'original_freeze_sha256':b.sha(b.ROOT/'results/me036b/frozen.json'),'references':records})


def evaluate():
    frozen=json.loads((OUT/'frozen.json').read_text())
    assert frozen['protocol_sha256']==b.sha(b.REPORT/'RIDGE-CORRECTION.md')
    for r in frozen['references']:assert b.sha(b.ROOT/r['coefficient'])==r['sha256']
    rows=[]
    for name in b.DATASETS:
        z,meta=b.load_data(name);_,val,end=meta['split_edges']
        for r in frozen['references']:
            if r['dataset']!=name:continue
            model=Ridge(np.load(b.ROOT/r['coefficient']))
            row={**r,'test':b.score(model,z,val,end,r['horizon'])};rows.append(row)
            print('CORRECTED TEST',name,r['horizon'],round(row['test']['MSE'],6),flush=True)
    b.dump(OUT/'results.json',{'freeze_sha256':b.sha(OUT/'frozen.json'),'references':rows,'posthoc':True})


if __name__=='__main__':
    torch.set_num_threads(1)
    p=b.argparse.ArgumentParser();p.add_argument('mode',choices=('train','evaluate'))
    args=p.parse_args();globals()[args.mode]()
