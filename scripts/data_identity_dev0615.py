"""DEV061.5 public adapters and training labels. Never exported with a policy."""
import numpy as np
import torch
from tac_osm.development.learned_agent_world import World,event,teacher,outcomes


def codebook():
    rng=np.random.default_rng(6150900);rows=[]
    while len(rows)<100:
        x=rng.normal(size=16);x/=np.linalg.norm(x)
        if not rows or np.max(np.array(rows)@x)<.75:rows.append(x)
    return np.array(rows,np.float32)


BOOK=codebook()


class IdentityWorld(World):
    def __init__(self,seed,index,entities,history,family='seen'):
        super().__init__(seed,index,entities,history);self.family=family
        rng=np.random.default_rng(np.random.SeedSequence([seed,index,entities,history,615]))
        order=rng.permutation(entities);self.parents[:]=-1
        for i in range(1,entities):
            if rng.random()<.75:self.parents[order[i]]=order[int(rng.integers(i))]
        records=[(0,int(self.keys[e]),None,int(self.bits[e])) for e in range(entities)]
        records += [(1,int(self.keys[e]),int(self.keys[p]),0) for e,p in enumerate(self.parents) if p>=0]
        records += [(3,int(rng.integers(8)),int(rng.integers(8)),int(rng.integers(2))) for _ in range(history-len(records))]
        rng.shuffle(records);self.events=[event(*r,t) for t,r in enumerate(records)]
        pool=np.arange(80,100) if family=='unseen' else np.arange(80)
        self.labels=rng.choice(pool,8,replace=False);self.vectors=BOOK[self.labels].copy()
        if family in ('fresh','gaussian','collision'):
            while True:
                x=rng.normal(size=(8,16));x/=np.linalg.norm(x,axis=-1,keepdims=True)
                if family!='fresh' or np.max(x@x.T-np.eye(8)*2)<.75:break
            self.vectors=x.astype(np.float32)
            if family=='collision':
                i,j=int(self.keys[0]),int(self.keys[1]);v=rng.normal(size=16);v-=v@x[i]*x[i];v/=np.linalg.norm(v)
                self.vectors[j]=.99*x[i]+np.sqrt(1-.99**2)*v

    def public(self,records=None):
        records=self.events if records is None else records
        result=[]
        for e in records:
            x=np.zeros(38,np.float32);x[:4]=e[:4]
            if e[4:12].any():x[4:20]=self.vectors[int(e[4:12].argmax())]
            if e[12:20].any():x[20:36]=self.vectors[int(e[12:20].argmax())]
            x[36]=e[20];x[37]=e[21]*256;result.append(x)
        return np.array(result,np.float32).reshape(-1,38)

    def public_candidates(self):
        d,v,q,qv,c=self.candidates();desc=np.zeros((13,19),np.float32);desc[:,:3]=d[:,:3]
        for i,(kind,target) in enumerate(c):
            if kind!='stop':desc[i,3:]=self.vectors[self.keys[target]]
        keys=np.zeros((6,16),np.float32);keys[:self.entities]=self.vectors[self.keys]
        return desc,v,keys,qv,c


def snapshot(w):
    d,v,q,qv,commands=w.candidates();one=np.zeros((13,6),np.float32);done=np.zeros(13,np.float32);cost=np.zeros(13,np.float32)
    for i,command in enumerate(commands):
        for prob,bits,c in outcomes(w.bits,w.parents,command):
            one[i,:w.entities]+=prob*bits;done[i]+=prob*float(not bits.any());cost[i]+=prob*c
    current=np.zeros(6,np.float32);current[:w.entities]=w.bits
    return {'events':np.array(w.events,np.float32),'keys':w.keys.copy(),'labels':w.labels.copy(),'descriptors':d,'valid':v,'query_valid':qv,
       'policy':np.pad(teacher(w.bits,w.parents,commands),(0,13-len(commands))), 'current':current,'completion':np.float32(not w.bits.any()),'one':one,'done':done,'cost':cost,
       'depth':len(w.actions),'collection_cost':w.cost,'seed':w.seed,'index':w.index,'n':w.entities,'h':w.history}


def corpus(seed,count):
    records=[]
    for i in range(count):
        w=IdentityWorld(seed,i,(2,4,6)[i%3],(16,32,64)[i//3%3]);rng=np.random.default_rng(np.random.SeedSequence([seed,i,616]))
        for _ in range(int(rng.integers(w.entities+3))):
            _,_,_,_,commands=w.candidates()
            if rng.random()<.5:choice=int(teacher(w.bits,w.parents,commands).argmax())
            else:choice=int(rng.integers(len(commands)))
            if commands[choice][0]=='stop':break
            w.act(commands[choice])
        records.append(snapshot(w))
    return records


def canonical(records,gate):
    rows=[]
    for r in records:
        e=torch.as_tensor(r['events']);admit=(gate(e[:,:4]).squeeze(-1)>=0).detach().numpy();x=dict(r);x['signals']=e[admit][-64:];rows.append(x)
    return rows


def batch(records,ids,model,generator=None,augment=False):
    rs=[records[int(i)] for i in ids];b=len(rs);length=max(len(r['signals']) for r in rs);events=torch.zeros(b,length,38);present=torch.zeros(b,length,dtype=torch.bool)
    labels=torch.tensor(np.stack([r['labels'] for r in rs]));vectors=torch.tensor(BOOK)[labels]
    if augment:
        vectors=torch.randn(b,8,16,generator=generator);vectors=vectors/vectors.norm(dim=-1,keepdim=True)
        bad=(vectors@vectors.transpose(-1,-2)-torch.eye(8)[None]*2).amax((-1,-2))>=.75
        while bad.any():
            v=torch.randn(int(bad.sum()),8,16,generator=generator);v=v/v.norm(dim=-1,keepdim=True);vectors[bad]=v
            bad=(vectors@vectors.transpose(-1,-2)-torch.eye(8)[None]*2).amax((-1,-2))>=.75
    for i,r in enumerate(rs):
        e=r['signals'];n=len(e);present[i,:n]=True;events[i,:n,:4]=e[:,:4];events[i,:n,4:20]=vectors[i,e[:,4:12].argmax(-1)]
        other=e[:,12:20].any(-1);events[i,:n,20:36]=vectors[i,e[:,12:20].argmax(-1)]*other[:,None]
        events[i,:n,36:]=e[:,20:];events[i,:n,37]*=256
    desc=torch.tensor(np.stack([r['descriptors'] for r in rs]));candidates=torch.zeros(b,13,19);candidates[:,:,:3]=desc[:,:,:3]
    action_keys=desc[:,:,3:].argmax(-1);candidates[:,:,3:]=vectors[torch.arange(b)[:,None],action_keys]*desc[:,:,3:].any(-1)[:,:,None]
    keys=torch.zeros(b,6,16)
    for i,r in enumerate(rs):keys[i,:len(r['keys'])]=vectors[i,torch.as_tensor(r['keys'])]
    out={k:torch.tensor(np.stack([r[k] for r in rs])) for k in ('valid','query_valid','policy','current','completion','one','done','cost')}
    out.update(events=events,present=present,keys=keys,candidates=candidates)
    parent_labels=torch.full((b,6),6,dtype=torch.long)
    for i,r in enumerate(rs):
        e=r['signals']
        for j,key in enumerate(r['keys']):
            select=(e[:,1]>0)&(e[:,4+key]>0)
            if select.any():
                other=int(e[select][-1,12:20].argmax());where=np.flatnonzero(r['keys']==other)
                if len(where):parent_labels[i,j]=int(where[0])
    out['parent_labels']=parent_labels
    return out
