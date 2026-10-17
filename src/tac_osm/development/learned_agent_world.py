"""DEV058 training/evaluator environment. NEVER shipped with the learned agent.

Explicit simulator laws and teacher targets live here, outside deployment.
No function here is called by the exported model or its public JSON runtime.
"""
import numpy as np

KEYS=8
MAX_ENTITIES=6
MAX_ACTIONS=13
EVENT_FEATURES=22
ACTION_FEATURES=11


def event(kind,key,other,bit,tick):
    value=np.zeros(EVENT_FEATURES,dtype=np.float32);value[kind]=1
    if key is not None:value[4+key]=1
    if other is not None:value[12+other]=1
    value[20]=bit;value[21]=tick/256
    return value


def outcomes(bits,parents,command):
    """Ground-truth stochastic transition enumeration, training/evaluator only."""
    kind,target=command
    if kind=='stop':return [(1.,bits.copy(),20.*int(bits.sum()))]
    if kind=='inspect':return [(1.,bits.copy(),.1)]
    parent=parents[target]
    if bits[target] and (parent<0 or not bits[parent]):
        after=bits.copy();after[target]=0
        return [(.95,after,.6),(.05,bits.copy(),.6)]
    children=np.flatnonzero(parents==target)
    if not bits[target] and len(children) and not bits[children[0]]:
        after=bits.copy();after[children[0]]=1
        return [(.8,after,.6),(.2,bits.copy(),.6)]
    return [(1.,bits.copy(),.6)]


def teacher(bits,parents,commands):
    """Training-only safe-progress targets; no call from learned deployment."""
    result=np.zeros(len(commands),dtype=np.float32)
    for i,(kind,target) in enumerate(commands):
        if not bits.any():result[i]=float(kind=='stop')
        elif kind=='repair' and bits[target] and (parents[target]<0 or not bits[parents[target]]):result[i]=1
    if not result.any():raise AssertionError('acyclic fixture has no safe progress action')
    return result/result.sum()


class World:
    def __init__(self,seed,index,entities,history):
        self.seed,self.index,self.entities,self.history=seed,index,entities,history
        self.rng=np.random.default_rng(np.random.SeedSequence([seed,index,entities,history]))
        self.keys=self.rng.choice(KEYS,entities,replace=False)
        self.bits=(self.rng.random(entities)<.55).astype(np.int64)
        if self.rng.random()<.15:self.bits[:]=0
        self.parents=np.full(entities,-1,dtype=np.int64)
        order=self.rng.permutation(entities)
        for p,c in zip(order[::2],order[1::2]):
            if self.rng.random()<.85:self.parents[c]=p
        records=[(0,int(self.keys[e]),None,int(self.bits[e])) for e in range(entities)]
        records.extend((1,int(self.keys[e]),int(self.keys[p]),0) for e,p in enumerate(self.parents) if p>=0)
        records.extend((3,int(self.rng.integers(KEYS)),int(self.rng.integers(KEYS)),int(self.rng.integers(2))) for _ in range(max(0,history-len(records))))
        self.rng.shuffle(records)
        self.events=[event(*r,t) for t,r in enumerate(records)]
        self.tick=len(self.events);self.actions=[];self.stopped=False
        self.cost=.01*len(self.events);self.action_cost=0.;self.failed_actions=0

    def candidates(self):
        commands=[('stop',-1)]+[(kind,e) for e in range(self.entities) for kind in ('inspect','repair')]
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.index,self.tick,581]))
        commands=[commands[i] for i in rng.permutation(len(commands))]
        descriptors=np.zeros((MAX_ACTIONS,ACTION_FEATURES),dtype=np.float32)
        for i,(kind,e) in enumerate(commands):
            descriptors[i,('stop','inspect','repair').index(kind)]=1
            if e>=0:descriptors[i,3+self.keys[e]]=1
        valid=np.arange(MAX_ACTIONS)<len(commands)
        queries=np.zeros((MAX_ENTITIES,KEYS),dtype=np.float32)
        for e,key in enumerate(self.keys):queries[e,key]=1
        return descriptors,valid,queries,np.arange(MAX_ENTITIES)<self.entities,commands

    def act(self,command):
        if self.stopped:raise ValueError('episode already closed')
        kind,target=command
        possible=outcomes(self.bits,self.parents,command)
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.index,self.tick,582]))
        selected=int(rng.choice(len(possible),p=[r[0] for r in possible]))
        before=self.bits.copy();_,self.bits,cost=possible[selected]
        self.cost+=cost;self.action_cost+=cost
        self.tick+=1
        if kind=='stop':self.stopped=True;receipt=[]
        else:
            if kind=='repair' and before[target] and self.bits[target]:self.failed_actions+=1
            observed=[target]+[e for e in range(self.entities) if e!=target and before[e]!=self.bits[e]]
            receipt=[event(2,int(self.keys[e]),None,int(self.bits[e]),self.tick+i) for i,e in enumerate(observed)]
            self.tick+=len(receipt)-1
            self.events.extend(receipt);self.cost+=.01*len(receipt)
        self.actions.append({'kind':kind,'target':int(target),'before':before.tolist(),'after':self.bits.tolist(),
                             'cost':cost,'receipt':[e.tolist() for e in receipt]})
        return receipt

    def finish(self,writes):
        terminal=0. if self.stopped else 20.*int(self.bits.sum())
        return {'seed':self.seed,'index':self.index,'entities':self.entities,'history':self.history,
                'success':bool(self.stopped and not self.bits.any()),'final_faults':int(self.bits.sum()),
                'stopped':self.stopped,'cost':self.cost+.02*writes+terminal,'writes':writes,
                'observations':len(self.events),'action_cost':self.action_cost+terminal,
                'observation_cost':.01*len(self.events),'write_cost':.02*writes,
                'failed_actions':self.failed_actions,'actions':self.actions,
                'initial_events':[e.tolist() for e in self.events[:self.history]],
                'keys':self.keys.tolist(),'parents':self.parents.tolist()}


def targets(world,commands):
    a,n=len(commands),world.entities
    one=np.zeros((MAX_ACTIONS,MAX_ENTITIES),np.float32)
    two=np.zeros((MAX_ACTIONS,MAX_ACTIONS,MAX_ENTITIES),np.float32)
    done=np.zeros(MAX_ACTIONS,np.float32);later_done=np.zeros((MAX_ACTIONS,MAX_ACTIONS),np.float32)
    future=np.zeros((MAX_ACTIONS,MAX_ACTIONS),np.float32);cost=np.zeros(MAX_ACTIONS,np.float32)
    for i,command in enumerate(commands):
        for probability,bits,c in outcomes(world.bits,world.parents,command):
            one[i,:n]+=probability*bits;done[i]+=probability*float(not bits.any());cost[i]+=probability*c
            if command[0]=='stop':
                dist=np.array([float(k=='stop') for k,_ in commands],np.float32)
            else:dist=teacher(bits,world.parents,commands)
            future[i,:a]+=probability*dist
            for j,next_action in enumerate(commands):
                next_cases=[(1.,bits,0.)] if command[0]=='stop' else outcomes(bits,world.parents,next_action)
                for p2,after,_ in next_cases:
                    two[i,j,:n]+=probability*p2*after
                    later_done[i,j]+=probability*p2*float(not after.any())
    graph=np.zeros((MAX_ACTIONS,MAX_ACTIONS),np.float32)
    for i,(kind,e) in enumerate(commands):
        if kind=='stop':graph[i,:a]=1/a
        else:
            parent=int(world.parents[e]);key=parent if parent>=0 else e
            for j,(other,target) in enumerate(commands):graph[i,j]=float(other!='stop' and target==key)
            graph[i]/=graph[i].sum()
    return {'policy':np.pad(teacher(world.bits,world.parents,commands),(0,MAX_ACTIONS-a)),
            'one':one,'two':two,'done':done,'later_done':later_done,'future':future,'cost':cost,
            'completion':np.float32(not world.bits.any()),'graph':graph}


def snapshots(seed,count,histories=(16,32,64,96)):
    records=[]
    for i in range(count):
        n=(2,4,6)[i%3];h=histories[(i//3)%len(histories)]
        world=World(seed,i,n,h)
        rng=np.random.default_rng(np.random.SeedSequence([seed,i,583]))
        for _ in range(int(rng.integers(n+1))):
            _,_,_,_,commands=world.candidates()
            candidates=[c for c in commands if c[0]!='stop']
            world.act(candidates[int(rng.integers(len(candidates)))])
        desc,valid,queries,qvalid,commands=world.candidates()
        records.append({'events':np.stack(world.events),'descriptors':desc,'valid':valid,
                        'queries':queries,'query_valid':qvalid,'targets':targets(world,commands),
                        'keys':world.keys.copy(),'parents':world.parents.copy()})
    return records
