"""Preregistered ten-seed full-bank retrieval probe, not policy training."""
import argparse
import concurrent.futures
import itertools
import json
from pathlib import Path
import resource
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from run_learned_stack_dev055 import write,digest,git
PROTOCOL=ROOT/'research/dev060/protocol-foundations.json'


class RetrievalProbe(nn.Module):
    def __init__(self,temporal,seed):
        super().__init__();torch.manual_seed(seed);self.temporal=temporal
        self.content=nn.Sequential(nn.Linear(18,32),nn.GELU(),nn.Linear(32,32),nn.GELU(),nn.Linear(32,1))
        self.raw_decay=nn.Parameter(torch.zeros(()),requires_grad=temporal)

    def forward(self,query,records):
        # Both arms see the SAME timestamp information; no identity matching rule.
        age=1-records[:,:,-1]
        x=torch.cat((query[:,None].expand(-1,records.shape[1],-1),records[:,:,:9],age[:,:,None]),-1)
        score=self.content(x).squeeze(-1)
        if self.temporal:score=score-F.softplus(self.raw_decay)*age
        return score


def data(seed,count,length):
    rng=np.random.default_rng(np.random.SeedSequence([seed,count,length]));queries=[];banks=[];labels=[];matches=[]
    for _ in range(count):
        ids=rng.normal(size=(4,8));ids/=np.linalg.norm(ids,axis=-1,keepdims=True)
        keys=rng.integers(4,size=length);target=int(rng.integers(4));keys[:2]=target
        rng.shuffle(keys);bits=rng.integers(2,size=length)
        relevant=np.flatnonzero(keys==target);bits[relevant[-1]]=1-bits[relevant[-2]]
        bank=np.concatenate((ids[keys],bits[:,None],np.linspace(0,1,length)[:,None]),-1)
        permutation=rng.permutation(length);bank=bank[permutation];mask=keys[permutation]==target
        latest=int(np.flatnonzero(permutation==relevant[-1])[0])
        queries.append(ids[target]);banks.append(bank);labels.append(latest);matches.append(mask)
    return {'query':torch.tensor(np.array(queries),dtype=torch.float32),'records':torch.tensor(np.array(banks),dtype=torch.float32),
            'target':torch.tensor(labels),'matches':torch.tensor(np.array(matches))}


@torch.no_grad()
def score(model,ds):
    logits=model(ds['query'],ds['records']);chosen=logits.argmax(-1);row=torch.arange(len(chosen))
    exact=chosen==ds['target'];matching=ds['matches'][row,chosen]
    status=ds['records'][row,chosen,8]==ds['records'][row,ds['target'],8]
    return {'examples':len(row),'exact':float(exact.float().mean()),'stale_same_identity':float((matching&~exact).float().mean()),
            'wrong_identity':float((~matching).float().mean()),'status_accuracy':float(status.float().mean()),
            'chosen':chosen.tolist(),'targets':ds['target'].tolist()}


def fit(job):
    directory,arm,seed=job;torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    p=json.loads(PROTOCOL.read_text())['memory_probe'];start=time.process_time();model=RetrievalProbe(arm=='temporal',seed)
    train=[data(p['data_seeds']['train'],p['train_examples'],length) for length in p['train_bank_lengths']]
    val=[data(p['data_seeds']['validation'],p['validation_examples'],length) for length in p['train_bank_lengths']]
    opt=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=.003,weight_decay=.0001)
    rng=torch.Generator().manual_seed(seed+1);curves=[]
    for update in range(1,p['updates']+1):
        ds=train[update%len(train)];ids=torch.randint(len(ds['target']),(p['batch'],),generator=rng)
        loss=F.cross_entropy(model(ds['query'][ids],ds['records'][ids]),ds['target'][ids])
        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        if update==1 or update%100==0:
            curves.append({'update':update,'loss':float(loss.detach()),'validation':[score(model,d)['exact'] for d in val]})
    path=Path(directory)/f'{arm}-{seed}';path.mkdir(parents=True,exist_ok=True);torch.save(model.state_dict(),path/'agent.pt')
    report={'arm':arm,'seed':seed,'source':git('rev-parse','HEAD'),'parameters':sum(v.numel() for v in model.parameters()),
            'updates':p['updates'],'curves':curves,'cpu_seconds':time.process_time()-start,'checkpoint_sha256':digest(path/'agent.pt')}
    write(path/'fit.json',report);return report


def evaluate_all(directory):
    torch.set_num_threads(1);root=Path(directory);p=json.loads(PROTOCOL.read_text())['memory_probe'];reports=json.loads((root/'fits.json').read_text())
    assert len(reports)==20 and all(r['updates']==p['updates'] for r in reports)
    result={'scope':'Exact-record full-bank retrieval only. Neither deployed policy reliability nor C5; no advantage possible over perfect latest-record reference.',
            'arms':{},'comparisons':{},'world_completion_claim':False,'protocol_sha256':digest(PROTOCOL)}
    for arm in p['arms']:
        result['arms'][arm]={}
        for seed in p['initialization_seeds']:
            path=root/f'{arm}-{seed}';model=RetrievalProbe(arm=='temporal',seed).eval()
            model.load_state_dict(torch.load(path/'agent.pt',weights_only=True))
            cells={str(length):score(model,data(p['data_seeds']['test'],p['test_examples_per_length'],length)) for length in p['test_bank_lengths']}
            write(path/'test.json',cells);result['arms'][arm][str(seed)]={key:{k:v for k,v in c.items() if k not in ('chosen','targets')} for key,c in cells.items()}
    raw={}
    for length in p['test_bank_lengths']:
        d=np.array([result['arms']['temporal'][str(s)][str(length)]['exact']-result['arms']['content'][str(s)][str(length)]['exact'] for s in p['initialization_seeds']])
        means=np.array([np.mean(d*np.array(v)) for v in itertools.product((-1,1),repeat=10)])
        raw[str(length)]=float(np.mean(np.abs(means)>=abs(d.mean())-1e-12))
        result['comparisons'][str(length)]={'paired_differences':d.tolist(),'mean':float(d.mean()),'paired_seed95_t':[float(d.mean()-2.262157*d.std(ddof=1)/np.sqrt(10)),float(d.mean()+2.262157*d.std(ddof=1)/np.sqrt(10))],'two_sided_signflip_p':raw[str(length)]}
    previous=0
    for rank,length in enumerate(sorted(raw,key=raw.get)):
        previous=max(previous,min(1.,raw[length]*(4-rank)));result['comparisons'][length]['holm_p']=previous
    write(root/'summary.json',result)
    write(root/'manifest.json',{str(f.relative_to(root)):digest(f) for f in sorted(root.rglob('*')) if f.is_file() and f.name!='manifest.json'})
    return result


def run(directory):
    p=json.loads(PROTOCOL.read_text())['memory_probe'];root=Path(directory);root.mkdir(parents=True,exist_ok=True)
    jobs=[(str(root),arm,seed) for arm in p['arms'] for seed in p['initialization_seeds']]
    reports=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        for report in pool.map(fit,jobs):
            reports.append(report);print('fit',report['arm'],report['seed'],round(report['cpu_seconds'],2),flush=True)
            if sum(r['cpu_seconds'] for r in reports)>600:raise RuntimeError('Registered CPU cap exceeded; no complete claim')
    write(root/'fits.json',reports);print(json.dumps(evaluate_all(root)['comparisons']),flush=True)
    write(root/'resources.json',{'children_cpu_seconds':resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime+resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime})
    # Include resource provenance after all outputs exist.
    write(root/'manifest.json',{str(f.relative_to(root)):digest(f) for f in sorted(root.rglob('*')) if f.is_file() and f.name!='manifest.json'})


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--directory',required=True);ap.add_argument('--replay',action='store_true');args=ap.parse_args()
    if args.replay:evaluate_all(args.directory)
    else:run(args.directory)
