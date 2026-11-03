"""Same licensed source; generated copy interface is explicitly auxiliary."""
import hashlib,random
import torch
from collect016.data import pools,history,query

def stage(step):return 1 if step<=256 else 2 if step<=512 else 4
def training_batch(data,pool,*,seed,step,length=128):
    rng=random.Random(seed*1000003+step);n=stage(step)
    offset=rng.randrange(len(data)-length-1);rows=[data[offset:offset+length+1]]
    meta=dict(seed=seed,step=step,natural_offset=offset,records=[]);mask=torch.zeros(4,length,dtype=torch.bool);first=mask.clone()
    for row in range(1,4):
        h=history(data,pool,rng,n=n,partition='train',identity=rng.randrange(100000),gap=0,overwrite=n>1 and step%3==0)
        absent=(step*3+row)%5==0;k=rng.choice(h['absent'] if absent else h['keys']);answer=b'????' if absent else h['latest'][k]
        q=query(k);position=rng.randrange(96,124);maxgap=position+1-len(q)-len(h['prefix'])
        gap=rng.randrange(min(8 if n==1 else 16 if n==2 else 24,maxgap)+1);lead=maxgap-gap
        left=rng.randrange(len(data)-max(1,lead));mid=rng.randrange(len(data)-max(1,gap));right=rng.randrange(len(data)-length)
        prefix=data[left:left+lead]+h['prefix']+data[mid:mid+gap]+q
        episode=prefix+answer;tail=length+1-len(episode);episode+=data[right:right+tail]
        assert len(episode)==length+1 and len(prefix)-1==position
        rows.append(episode);mask[row,position:position+4]=True;first[row,position]=True
        meta['records'].append(dict(facts=h['facts'],key=k,absent=absent,answer_hex=answer.hex(),first_position=position,leading_offset=left,leading_bytes=lead,filler_offset=mid,filler_bytes=gap,tail_offset=right,tail_bytes=tail))
    slab=torch.tensor([list(x) for x in rows],dtype=torch.long);meta['batch_sha256']=hashlib.sha256(slab.numpy().tobytes()).hexdigest()
    return slab[:,:-1],slab[:,1:],mask,first,meta
