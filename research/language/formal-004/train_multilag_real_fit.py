"""MLAG001 authentic Vim FIT/DEV matched-source original 64 vs new 64+96.
Exploratory already examined DEV, no untouched TEST, original checkpoints frozen.
"""
import argparse,hashlib,json,random,time,sys
from pathlib import Path
import numpy as np,torch
import torch.nn.functional as F
SHA='835212d74cc0d2d4d22ff7e75beb12771586cce58f2e63943690e5d859417457'
SEEDS=(1701,1702,1703,1704,1705);MODES=('tanh','silu');V=128

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def chars(raw):return np.frombuffer(bytes(b if b<128 else 63 for b in raw),dtype=np.uint8).astype(np.int64)
def batches(docs):
    arr=[]
    for d in docs:
        for i in np.linspace(0,len(d)-130,8,dtype=np.int64):arr.append((torch.from_numpy(d[i:i+128]),torch.from_numpy(d[i+1:i+129])))
    return [(torch.stack([x for x,y in arr[i:i+8]]),torch.stack([y for x,y in arr[i:i+8]])) for i in range(0,len(arr),8)]
def sample(fit,rng):
    w=np.array([len(a)-129 for a in fit],dtype=np.float64);w/=w.sum();choice=rng.choice(len(fit),size=8,p=w)
    x=np.empty((8,128),np.int64);y=np.empty_like(x)
    for j,di in enumerate(choice):
        d=fit[di];k=int(rng.integers(0,len(d)-129));x[j]=d[k:k+128];y[j]=d[k+1:k+129]
    return torch.from_numpy(x),torch.from_numpy(y)
def evaluate(net,dev):
    net.eval();tot=correct=nchar=0;hist={lag:[0.,0.,0,0.,0] for lag in (64,96)}
    with torch.no_grad():
        for x,y in dev:
            z,trace=net(x,return_trace=True)
            tot+=F.cross_entropy(z.reshape(-1,V),y.reshape(-1),reduction='sum').item()
            correct+=(z.argmax(-1)==y).sum().item();nchar+=y.numel()
            for lag in hist:
                h,t=net.historical_logits(x,trace,lag);wrong,_=net.historical_logits(x,trace,lag+1)
                q=hist[lag];q[0]+=F.cross_entropy(h.reshape(-1,V),t.reshape(-1),reduction='sum').item();q[1]+=(h.argmax(-1)==t).sum().item();q[2]+=t.numel()
                q[3]+=F.cross_entropy(wrong.reshape(-1,V),t[:,1:].reshape(-1),reduction='sum').item();q[4]+=t[:,1:].numel()
    res={'char_nll':tot/nchar,'char_acc':correct/nchar}
    for lag,q in hist.items():res.update({f'hist{lag}_nll':q[0]/q[2],f'hist{lag}_acc':q[1]/q[2],f'hist{lag}_wrong_key_nll':q[3]/q[4]})
    return res

def main(a):
    torch.set_num_threads(1);root=Path(a.archive);src=Path(a.source);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if sha(root/'parallel_tac_language_gate.py')!=SHA:raise ValueError('original source changed')
    sys.path.insert(0,str(root));from parallel_tac_language_gate import ParallelTACLanguage
    protocol=json.loads((root/'PROTOCOL.json').read_text());original=json.loads((root/'RESULTS.json').read_text())
    if original['protocol']!=protocol or (protocol['steps'],protocol['batch'],protocol['context'],protocol['active_learned_params_per_arm'])!=(120,8,128,100680):raise ValueError('original protocol drift')
    hashes=protocol['source']['byte_sha256'];part=protocol['source']['doc_partitions']
    for name,h in hashes.items():
        if Path(name).name!=name or sha(src/name)!=h:raise ValueError('source hash mismatch '+name)
    fit=[chars((src/n).read_bytes()) for n in part['fit']];dev=batches([chars((src/n).read_bytes()) for n in part['dev']])
    pp={'name':'MLAG001','training':'genuine Vim FIT only','evaluation':'exposed DEV not untouched TEST','source_sha256':SHA,'original_protocol_sha256':sha(root/'PROTOCOL.json'),'source_chapter_sha256':hashes,'training_steps':120,'batch':8,'context':128,'optimizer':'AdamW lr=.002 wd=.01 clip=1','learned_parameters':100680,'original_aux':{'64':.3},'new_aux':{'64':.15,'96':.15},'same_original_sample_order':True,'seeds':SEEDS,'modes':MODES,'cpu_fairness_not_required':True}
    (out/'PROTOCOL.json').write_text(json.dumps(pp,indent=2,sort_keys=True))
    results={'protocol':pp,'runs':[]}
    for seed in SEEDS:
        for mode in MODES:
            prior=next(row for row in original['runs'] if row['seed']==seed and row['mode']==mode)
            if sha(root/prior['checkpoint_file'])!=prior['checkpoint_sha256']:raise ValueError('baseline checkpoint mismatch')
            torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
            net=ParallelTACLanguage(decoder=75,position_mode='multiscale',gate_mode=mode)
            if sum(q.numel() for q in net.parameters() if q.requires_grad)!=100680:raise ValueError('param mismatch')
            opt=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.01);rng=np.random.default_rng(seed);order=hashlib.sha256();trace=[];start=time.perf_counter()
            net.train()
            for step in range(120):
                x,y=sample(fit,rng);order.update(x.numpy().tobytes());order.update(y.numpy().tobytes());z,t=net(x,return_trace=True)
                main=F.cross_entropy(z.reshape(-1,V),y.reshape(-1));h64,v64=net.historical_logits(x,t,64);h96,v96=net.historical_logits(x,t,96)
                h64=F.cross_entropy(h64.reshape(-1,V),v64.reshape(-1));h96=F.cross_entropy(h96.reshape(-1,V),v96.reshape(-1))
                loss=main+.15*h64+.15*h96
                if not torch.isfinite(loss):raise ValueError('nonfinite')
                opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
                if (step+1)%40==0:trace.append({'step':step+1,'fit_char_nll':main.item(),'fit_hist64_nll':h64.item(),'fit_hist96_nll':h96.item()})
            if order.hexdigest()!=prior['sample_sha256']:raise ValueError('sample order mismatch with original')
            metrics=evaluate(net,dev);ck={'model':net.state_dict(),'optimizer':opt.state_dict(),'seed':seed,'mode':mode,'protocol':pp,'sample_sha256':order.hexdigest(),'torch_rng':torch.get_rng_state(),'numpy_rng':rng.bit_generator.state,'python_rng':random.getstate()}
            path=out/f'mlag_{mode}_{seed}.pt';torch.save(ck,path)
            saved=torch.load(path,map_location='cpu',weights_only=True);check=ParallelTACLanguage(decoder=75,position_mode='multiscale',gate_mode=mode);check.load_state_dict(saved['model'],strict=True)
            other=evaluate(check,dev)
            if any(abs(metrics[k]-other[k])>1e-6 for k in metrics):raise ValueError('cold reload mismatch')
            baseline={'char_nll':prior['heldout']['nll_nats'],'hist64_nll':prior['historical']['64']['nll'],'hist96_nll':prior['historical']['96']['nll'],'hist64_acc':prior['historical']['64']['accuracy'],'hist96_acc':prior['historical']['96']['accuracy']}
            rr={'mode':mode,'seed':seed,'checkpoint':path.name,'checkpoint_sha256':sha(path),'original_checkpoint_sha256':prior['checkpoint_sha256'],'sample_sha256':order.hexdigest(),'dev':metrics,'original':baseline,'fit_trace':trace,'train_s':time.perf_counter()-start,'cold_replay':'PASS'}
            results['runs'].append(rr);(out/'RESULTS.json').write_text(json.dumps(results,indent=2,sort_keys=True))
            print('FINISHED',mode,seed,'hist64',round(metrics['hist64_nll'],3),'hist96',round(metrics['hist96_nll'],3),'orig96',round(baseline['hist96_nll'],3),flush=True)
    stats={}
    for mode in MODES:
        rr=[x for x in results['runs'] if x['mode']==mode];stats[mode]={}
        for metric in ('char_nll','hist64_nll','hist96_nll','hist64_acc','hist96_acc'):
            stats[mode][metric]={'multi':float(np.mean([x['dev'][metric] for x in rr])),'original':float(np.mean([x['original'][metric] for x in rr]))}
        stats[mode]['hist96_improved_seeds']=sum(x['dev']['hist96_nll']<x['original']['hist96_nll'] for x in rr)
        stats[mode]['wrong_key_nll_penalty96']=float(np.mean([x['dev']['hist96_wrong_key_nll']-x['dev']['hist96_nll'] for x in rr]))
    results['aggregates']=stats;(out/'RESULTS.json').write_text(json.dumps(results,indent=2,sort_keys=True))
    print('RESULTS',json.dumps(stats,sort_keys=True),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--source',required=True);p.add_argument('--out',required=True);main(p.parse_args())
