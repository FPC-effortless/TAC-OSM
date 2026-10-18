"""DEV062A synthetic physics, exact PUBLIC targets and shared public adapters.

Nothing in this module is shipped with a neural checkpoint.
"""
import numpy as np
import torch
from tac_osm.development.uncertain_world import PublicBelief,STATES,COMMANDS,prior


class PriceBelief(PublicBelief):
    def __init__(self,parents,error=.1,success=.95,inspection=.1):
        super().__init__(parents,error,success);self.inspection=inspection

    def action_costs(self,belief=None,depth=2):
        p=self.p if belief is None else np.asarray(belief)
        terminal=lambda q:.01+float(20*np.dot(q,STATES.sum(-1)))
        result=[terminal(p)]
        for action,(kind,_) in enumerate(COMMANDS[1:],1):
            immediate=self.inspection+.02 if kind=='inspect' else .63
            future=sum(mass*(terminal(q) if depth==1 else float(self.action_costs(q,depth-1).min())) for mass,q in self.branches(p,action))
            result.append(immediate+future)
        return np.array(result)

    def policy(self):
        values=self.action_costs(depth=2)
        if self.p[0]<1-1e-12:values[0]=1e6
        safe=values<=values.min()+1e-10
        return safe.astype(np.float32)/safe.sum()


class BeliefWorld:
    def __init__(self,seed,index,history=32,error=.1,inspection=.1):
        self.seed=seed;self.index=index;self.history=history;self.error=error;self.inspection=inspection;self.step=0;self.stopped=False;self.events=[];self.actions=[]
        rng=np.random.default_rng(np.random.SeedSequence([seed,index,620]))
        self.identities=rng.normal(size=(2,16)).astype(np.float32);self.identities/=np.linalg.norm(self.identities,axis=-1,keepdims=True)
        self.parents=np.array([-1,-1]);order=rng.permutation(2)
        if rng.random()<.85:self.parents[order[1]]=order[0]
        self.bits=STATES[int(rng.choice(4,p=prior()))].copy();self.costs={'observations':history*.01,'inspections':0.,'repairs':0.,'terminal':0.}
        sensor=np.random.default_rng(np.random.SeedSequence([seed,index,621]))
        for entity in range(2):
            if sensor.random()>=.5:self.events.append(self.record(0,entity,int(self.bits[entity])^int(sensor.random()<error)))
        for entity,parent in enumerate(self.parents):
            if parent>=0:self.events.append(self.record(1,entity,0,parent))
        while len(self.events)<history:
            e=self.record(5,None,int(sensor.integers(2)));e[6:22]=sensor.normal(size=16);self.events.append(e)
        sensor.shuffle(self.events)
        for tick,e in enumerate(self.events):e[42]=tick

    def record(self,source,entity,bit=0,other=None,action=None):
        e=np.zeros(46,np.float32);e[source]=1
        if entity is not None:e[6:22]=self.identities[entity]
        if other is not None:e[22:38]=self.identities[other]
        e[38]=bit
        if action is not None:e[39+action]=1
        e[42]=len(self.events);e[43]=source==4;e[44]=self.error;e[45]=self.inspection
        return e

    def reference(self):
        # Derive parents and observations solely from the initial public records.
        parents=np.array([-1,-1]);initial=[]
        for e in self.events[:self.history]:
            if e[1]:parents[int((self.identities@e[6:22]).argmax())]=int((self.identities@e[22:38]).argmax())
            elif e[0]:initial.append((int((self.identities@e[6:22]).argmax()),int(e[38])))
        b=PriceBelief(parents,self.error,inspection=self.inspection)
        for target,bit in initial:b.condition(target,bit)
        return b

    def candidates(self):
        permutation=np.random.default_rng(np.random.SeedSequence([self.seed,self.index,self.step,622])).permutation(5)
        d=np.zeros((5,19),np.float32)
        for row,a in enumerate(permutation):
            kind,target=COMMANDS[a];d[row,('stop','inspect','repair').index(kind)]=1
            if target>=0:d[row,3:]=self.identities[target]
        return d,permutation

    def act(self,action):
        if self.stopped:raise ValueError('Closed world')
        kind,target=COMMANDS[action];before=self.bits.copy();start=len(self.events)
        self.events.append(self.record(2,None if target<0 else target,action=('stop','inspect','repair').index(kind)));self.costs['observations']+=.01
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.index,self.step,action,623]));receipt=[]
        if kind=='stop':self.stopped=True;self.costs['terminal']=20*int(self.bits.sum())
        elif kind=='inspect':
            receipt=[int(self.bits[target])];self.events.append(self.record(4,target,receipt[0]));self.costs['inspections']+=self.inspection;self.costs['observations']+=.01
        else:
            cases=PublicBelief(self.parents).effects(self.bits,COMMANDS[action]);j=int(rng.choice(len(cases),p=[p for p,_ in cases]));self.bits=cases[j][1];receipt=[int(self.bits[e])^int(rng.random()<self.error) for e in range(2)]
            for e,bit in enumerate(receipt):self.events.append(self.record(3,e,bit))
            self.costs['repairs']+=.6;self.costs['observations']+=.02
        self.actions.append({'action':int(action),'kind':kind,'target':int(target),'before':before.tolist(),'after':self.bits.tolist(),'receipt':receipt,'public':[e.tolist() for e in self.events[start:]]});self.step+=1
        return receipt

    def finish(self):
        costs=dict(self.costs)
        if not self.stopped:costs['terminal']=20*int(self.bits.sum())
        return {'seed':self.seed,'index':self.index,'history':self.history,'error':self.error,'inspection_price':self.inspection,'completion':bool(self.stopped and not self.bits.any()),'physical_repair':bool(not self.bits.any()),'faults':int(self.bits.sum()),'stopped':self.stopped,'cost':float(sum(costs.values())),'costs':costs,'actions':self.actions}


def snapshot(w,b):
    d,permutation=w.candidates();p=b.p.copy();after=np.stack([p@b.transitions[a] for a in permutation]);costs=b.action_costs(depth=1)[permutation]
    return {'events':torch.tensor(np.array(w.events)),'keys':torch.tensor(w.identities),'candidates':torch.tensor(d),'prices':torch.tensor([w.error,w.inspection]),'posterior':torch.tensor(p,dtype=torch.float32),'policy':torch.tensor(b.policy()[permutation]),'future':torch.tensor(after,dtype=torch.float32),'cost':torch.tensor(costs,dtype=torch.float32),'truth':int(np.flatnonzero((STATES==w.bits).all(-1))[0]),'depth':w.step,'world_seed':w.seed,'world_index':w.index}


def corpus(seed,count):
    records=[]
    for index in range(count):
        w=BeliefWorld(seed,index,(32,128)[index%2],inspection=(.05,.1,.2)[index%3]);b=w.reference();rng=np.random.default_rng(np.random.SeedSequence([seed,index,624]))
        for _ in range(int(rng.integers(7))):
            a=int(rng.integers(1,5)) if rng.random()<.4 else int(rng.choice(5,p=b.policy()))
            if a==0:break
            b.update(a,w.act(a))
        records.append(snapshot(w,b))
    return records


def batch(records,ids,gate):
    rows=[records[int(i)] for i in ids];signals=[r['events'][(gate(r['events'][:,:6]).squeeze(-1)>=0).detach()][-64:] for r in rows];length=max(1,max(map(len,signals)));events=torch.zeros(len(rows),length,46);present=torch.zeros(len(rows),length,dtype=torch.bool)
    for i,e in enumerate(signals):events[i,:len(e)]=e;present[i,:len(e)]=True
    result={k:torch.stack([r[k] for r in rows]) for k in ('keys','candidates','prices','posterior','policy','future','cost')};result.update(events=events,present=present);return result
