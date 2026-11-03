"""Verbatim source quotations, temporary keys, causal copy queries and provenance.

    Training uses TRAIN quotations/filler only. DEV quotations/filler and held-out
    key families are separate. The generated task framing is a mechanism probe,
    not authentic-world fact collection or an external intelligence benchmark.
"""
import hashlib,random
import torch
LETTERS=b'abcdefgh'
def pools(data):
    p={v:[] for v in LETTERS}
    for i in range(len(data)-8):
        if data[i] in p and all(32<=x<127 for x in data[i:i+8]):p[data[i]].append(i)
    if any(not z for z in p.values()):raise ValueError('missing source quote class')
    return p
def get_keys(rng,n,partition):
    first='abcdefghijklm' if partition=='train' else 'nopqrstuvwxyz'
    possible=[a+b for a in first for b in 'abcdefghijklmnopqrstuvwxyz']
    return rng.sample(possible,n+2)
def history(data,pool,rng,*,n=4,partition='train',identity=0,gap=256,overwrite=True,class_shift=0):
    keys=get_keys(rng,n,partition);facts=[];latest={};chunks=[]
    def write(k,c):
        offset=rng.choice(pool[LETTERS[(c+class_shift)%8]]);quote=data[offset:offset+8]
        chunks.append(b'@'+k.encode()+b': '+quote+b'\n');latest[k]=quote[:4]
        facts.append(dict(key=k,source_offset=offset,quote=quote.decode('ascii'),answer_hex=quote[:4].hex()))
    for j,k in enumerate(keys[:n]):write(k,identity*n+j+3 if overwrite and j<n//2 else identity*n+j)
    if overwrite:
        for j,k in enumerate(keys[:n//2]):write(k,identity*n+j)
    records=b''.join(chunks);offset=rng.randrange(len(data)-max(gap,1))
    filler=data[offset:offset+gap]
    return dict(prefix=records+filler+b'\n',keys=keys[:n],absent=keys[n:],latest=latest,
        facts=facts,filler_offset=offset,filler_bytes=gap,partition=partition,identity=identity,n=n)
def query(key):return b'Copy '+key.encode()+b': '
def training_batch(data,pool,*,seed,step,length=384):
    rng=random.Random(seed*1000003+step);n=2 if step%2 else 4
    # Fixed sequence budget. Records and query precede four true answer bytes.
    h=history(data,pool,rng,n=n,partition='train',identity=rng.randrange(100000),gap=0,overwrite=step%3==0)
    absent=step%5==0;k=rng.choice(h['absent'] if absent else h['keys'])
    answer=b'????' if absent else h['latest'][k];q=query(k)
    gap=length+1-4-len(q)-len(h['prefix'])
    offset=rng.randrange(len(data)-gap);filler=data[offset:offset+gap]
    prefix=h['prefix']+filler+q;episode=prefix+answer
    assert len(episode)==length+1
    natural_offset=rng.randrange(len(data)-length-1);natural=data[natural_offset:natural_offset+length+1]
    slab=torch.tensor([list(natural),list(episode)],dtype=torch.long)
    mask=torch.zeros(2,length,dtype=torch.bool);mask[1,len(prefix)-1:len(prefix)+3]=True
    metadata=dict(seed=seed,step=step,natural_offset=natural_offset,facts=h['facts'],filler_offset=offset,
        filler_bytes=gap,key=k,absent=absent,answer_hex=answer.hex(),answer_prediction_indices=mask[1].nonzero().flatten().tolist(),
        batch_sha256=hashlib.sha256(slab.numpy().tobytes()).hexdigest())
    return slab[:,:-1],slab[:,1:],mask,metadata
