"""Five-seed exploratory real-document training. Previously exposed DEV; no superiority claims."""
from pathlib import Path
import os
import hashlib,io,json,random,time,zipfile
import numpy as np
import torch
from torch.nn import functional as F
from tac_wave_ced_500k import WaveCED500K

root=Path(__file__).parent
with zipfile.ZipFile(os.environ.get('TAC_LANGUAGE_PREVIOUS_BUNDLE', '/mnt/data/tac_language_current_math_benchmark.zip')) as z:
    proto=json.loads(z.read('tac_language_current_math_benchmark/PROTOCOL_CURRENT.json'))
parts=proto['source']['doc_partitions'];original_sha=proto['source']['byte_sha256']
doc={}
for name in [*parts['fit'],*parts['dev']]:
    raw=(Path(os.environ.get('TAC_VIM_DOC_ROOT','/usr/share/vim/vim91/doc'))/name).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==original_sha[name],name
    a=np.frombuffer(raw,dtype=np.uint8).astype(np.int64).copy();a[a>=128]=63;doc[name]=a
print('36/36 original real source SHA-256 pass',flush=True)
counts=np.ones((128,128),dtype=np.float64)*.5
for n in parts['fit']:
    a=doc[n];np.add.at(counts,(a[:-1],a[1:]),1)
logbg=np.log(counts/counts.sum(1,keepdims=True))
windows=[]
for n in parts['dev']:
    a=doc[n]
    for i in np.linspace(0,len(a)-129,8,dtype=np.int64):windows.append(torch.tensor(a[i:i+129],dtype=torch.long))

def evaluate(m):
    m.eval();nll=0;correct=0;count=0;bg=0
    with torch.no_grad():
        for i in range(0,len(windows),8):
            s=torch.stack(windows[i:i+8]); x=s[:,:-1];y=s[:,1:]
            pred=m(x)
            nll+=F.cross_entropy(pred.reshape(-1,128),y.reshape(-1),reduction='sum').item()
            correct+=(pred.argmax(-1)==y).sum().item();count+=y.numel()
            bg-=logbg[x.numpy(),y.numpy()].sum()
    return {'nll':nll/count,'ppl':float(np.exp(nll/count)),'accuracy':correct/count,'bigram_nll':bg/count,'bigram_ppl':float(np.exp(bg/count)),'predictions':count}

results=[];torch.set_num_threads(2)
for seed in [1701,1702,1703,1704,1705]:
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    rng=np.random.default_rng(seed); model=WaveCED500K();opt=torch.optim.AdamW(model.parameters(),lr=.001)
    assert model.parameter_count()==499290
    sample_sha=hashlib.sha256();t0=time.perf_counter()
    for step in range(120):
        ss=[]
        for j in range(8):
            name=parts['fit'][int(rng.integers(30))]; a=doc[name]
            offset=int(rng.integers(len(a)-129));sample_sha.update(f'{name}:{offset};'.encode())
            ss.append(torch.tensor(a[offset:offset+129],dtype=torch.long))
        s=torch.stack(ss);y=model(s[:,:-1]);loss=F.cross_entropy(y.reshape(-1,128),s[:,1:].reshape(-1))
        opt.zero_grad(set_to_none=True);loss.backward()
        if step==0:assert sum(p.numel() for p in model.parameters() if p.grad is not None)==499290
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
    dev=evaluate(model)
    file=root/f'seed_{seed}.pt'
    torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'torch_rng':torch.get_rng_state(),
                'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),'sampler_rng':rng.bit_generator.state,
                'seed':seed,'samples_sha256':sample_sha.hexdigest(),'dev':dev,'updates':120},file)
    r={'seed':seed,'train_last_nll':loss.item(),'dev':dev,'samples_sha256':sample_sha.hexdigest(),
       'checkpoint':file.name,'checkpoint_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),
       'training_sec':time.perf_counter()-t0}
    results.append(r);print('seed',seed,'ppl',round(dev['ppl'],3),'bigram',round(dev['bigram_ppl'],3),flush=True)

# Independently instantiate and score frozen original state (no optimizer, no future input).
for row in results:
    f=root/row['checkpoint'];assert hashlib.sha256(f.read_bytes()).hexdigest()==row['checkpoint_sha256']
    ck=torch.load(f,map_location='cpu',weights_only=False);m=WaveCED500K();m.load_state_dict(ck['model']);dev=evaluate(m)
    assert abs(dev['nll']-row['dev']['nll'])<1e-8
print('5/5 original checkpoint no-fit local score replay PASS',flush=True)
(root/'RESULTS_REAL_TEXT_FEASIBILITY.json').write_text(json.dumps({
    'study':'TAC-LANG-FLOW-500K-001','evidence':'authentic human text, previously exposed DEV (NOT independent)',
    'trained_parameters':499290,'target':500000,'range':[475000,525000],
    'model_sha256':hashlib.sha256((root/'tac_wave_ced_500k.py').read_bytes()).hexdigest(),
    'source_sha_verified':36,'source_doc_partitions':parts,'source_sha256':original_sha,
    'model_objective':'next-character cross entropy only', 'optimizer_updates_per_seed':120,'batch':8,'context':128,
    'caveats':['no 500K learned GRU/SSM/Transformer controls','no equal compute architecture comparison',
               'no independent TEST','no learned historic retrieval proof','Torch implementation is batch-only; full carried-state parity open',
               'whole graph Lean/AD/IEEE proof open'], 'runs':results},indent=2))
