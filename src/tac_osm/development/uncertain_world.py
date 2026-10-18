"""DEV-060 simulator and PUBLIC-information references; never deployment code.

Exact four-state Bayes enumeration, known laws and bounded planning are explicit
reference scaffolding. They must not be relabeled learned components or shipped
inside a neural policy. All noisy repair receipts have state-independent schema.
"""
import itertools
import numpy as np

STATES=np.array(list(itertools.product((0,1),repeat=2)),dtype=np.int64)
COMMANDS=(('stop',-1),('inspect',0),('inspect',1),('repair',0),('repair',1))


def prior():
    p=np.prod(np.where(STATES,.55,.45),axis=1)*.85;p[0]+=.15
    return p


class PublicBelief:
    """Known-law information-matched reference, not a neural-model utility."""
    def __init__(self,parents,sensor_error=.1,repair_success=.95):
        self.parents=np.asarray(parents);self.error=sensor_error;self.success=repair_success
        self.transitions=[]
        for kind,target in COMMANDS:
            matrix=np.zeros((4,4))
            for s,bits in enumerate(STATES):
                for p,after in self.effects(bits,(kind,target)):
                    j=int(np.flatnonzero((STATES==after).all(-1))[0]);matrix[s,j]+=p
            self.transitions.append(matrix)
        self.transitions=np.array(self.transitions)
        self.p=prior()

    def effects(self,bits,command):
        kind,target=command
        if kind!='repair':return [(1.,bits.copy())]
        parent=self.parents[target]
        if bits[target] and (parent<0 or not bits[parent]):
            after=bits.copy();after[target]=0
            return [(self.success,after),(1-self.success,bits.copy())]
        children=np.flatnonzero(self.parents==target)
        if not bits[target] and len(children) and not bits[children[0]]:
            after=bits.copy();after[children[0]]=1
            return [(.8,after),(.2,bits.copy())]
        return [(1.,bits.copy())]

    def condition(self,target,bit,exact=False):
        error=0. if exact else self.error
        likelihood=np.where(STATES[:,target]==bit,1-error,error)
        values=self.p*likelihood;denominator=values.sum()
        if denominator<=0:raise ValueError('impossible public observation')
        self.p=values/denominator
        return self.p.copy()

    def branches(self,belief,action):
        kind,target=COMMANDS[action];after=np.asarray(belief)@self.transitions[action]
        if kind=='stop':return [(1.,after)]
        outputs=(list(itertools.product((0,1),repeat=2)) if kind=='repair' else [(0,),(1,)])
        result=[]
        for observation in outputs:
            if kind=='repair':
                likelihood=np.prod(np.where(STATES==np.array(observation),1-self.error,self.error),axis=1)
            else:likelihood=(STATES[:,target]==observation[0]).astype(float)
            joint=after*likelihood;mass=joint.sum()
            if mass>0:result.append((float(mass),joint/mass))
        return result

    def update(self,action,receipt):
        kind,target=COMMANDS[action]
        self.p=self.p@self.transitions[action]
        if kind=='inspect':self.condition(target,receipt[0],True)
        elif kind=='repair':
            for entity,bit in enumerate(receipt):self.condition(entity,bit,False)
        return self.p.copy()

    def action_costs(self,belief=None,depth=2):
        if depth<1:raise ValueError('positive bounded planning depth required')
        belief=self.p if belief is None else np.asarray(belief)
        terminal=lambda p:float(20*np.dot(p,STATES.sum(-1)))
        costs=[terminal(belief)]
        for action,(kind,_) in enumerate(COMMANDS[1:],1):
            immediate=.11 if kind=='inspect' else .62
            continuation=0.
            for mass,posterior in self.branches(belief,action):
                future=terminal(posterior) if depth==1 else float(self.action_costs(posterior,depth-1).min())
                continuation+=mass*future
            costs.append(immediate+continuation)
        return np.array(costs)

    def choose(self,depth=2):return int(self.action_costs(depth=depth).argmin())


class UncertainWorld:
    def __init__(self,seed,index,history=32):
        self.seed=seed;self.index=index;self.history=history;self.step=0;self.stopped=False
        rng=np.random.default_rng(np.random.SeedSequence([seed,index,600]))
        self.identities=rng.normal(size=(2,8)).astype(np.float32);self.identities/=np.linalg.norm(self.identities,axis=-1,keepdims=True)
        self.parents=np.array([-1,-1]);order=rng.permutation(2)
        if rng.random()<.85:self.parents[order[1]]=order[0]
        self.bits=STATES[int(rng.choice(4,p=prior()))].copy()
        self.events=[]
        for entity in range(2):
            if rng.random()>=.5:
                bit=int(self.bits[entity])^int(rng.random()<.1)
                self.events.append(self.record('initial',entity,bit))
        for entity,parent in enumerate(self.parents):
            if parent>=0:self.events.append(self.record('relation',entity,None,parent))
        while len(self.events)<history:self.events.append({'source':'noise','subject':rng.normal(size=8).tolist(),'value':int(rng.integers(2))})
        rng.shuffle(self.events)
        self.cost={'observations':history*.01,'inspections':0.,'repairs':0.,'terminal':0.}
        self.actions=[]

    def record(self,source,entity,bit,other=None):
        r={'source':source,'subject':self.identities[entity].tolist(),'value':bit}
        if other is not None:r['other']=self.identities[other].tolist()
        return r

    def payload(self):
        # Returned API contains neither true bits nor latent transition outcomes.
        descriptors=np.zeros((5,11),np.float32)
        permutation=np.random.default_rng(np.random.SeedSequence([self.seed,self.index,self.step,601])).permutation(5)
        for row,action in enumerate(permutation):
            kind,target=COMMANDS[action];descriptors[row,('stop','inspect','repair').index(kind)]=1
            if target>=0:descriptors[row,3:]=self.identities[target]
        events=np.zeros((len(self.events),24),np.float32)
        for t,r in enumerate(self.events):
            source=r['source'];events[t,('initial','relation','repair','noise','inspect').index(source) if source!='inspect' else 2]=1
            events[t,4:12]=r['subject']
            if source=='relation':events[t,12:20]=r['other']
            if r['value'] is not None:events[t,20]=r['value']
            events[t,21]=t/256
            events[t,22]=source=='inspect';events[t,23]=source=='repair'
        return {'events':events,'candidates':descriptors,'queries':self.identities.astype(np.float32).copy()},permutation

    def reference(self):
        # Derive reference entity order/relations from PUBLIC payload identities.
        lookup={tuple(identity):i for i,identity in enumerate(self.identities.tolist())}
        parents=np.full(2,-1)
        for r in self.events[:self.history]:
            if r['source']=='relation':parents[lookup[tuple(r['subject'])]]=lookup[tuple(r['other'])]
        belief=PublicBelief(parents)
        for r in self.events[:self.history]:
            if r['source']=='initial':belief.condition(lookup[tuple(r['subject'])],r['value'])
        return belief

    def act(self,action):
        if self.stopped:raise ValueError('closed world')
        kind,target=COMMANDS[action];before=self.bits.copy();receipt=[]
        rng=np.random.default_rng(np.random.SeedSequence([self.seed,self.index,self.step,action,602]))
        if kind=='stop':
            self.stopped=True;self.cost['terminal']=20*int(self.bits.sum())
        elif kind=='inspect':
            receipt=[int(self.bits[target])];self.events.append(self.record('inspect',target,receipt[0]));self.cost['inspections']+=.1;self.cost['observations']+=.01
        else:
            law=PublicBelief(self.parents);cases=law.effects(self.bits,(kind,target))
            chosen=int(rng.choice(len(cases),p=[p for p,_ in cases]));self.bits=cases[chosen][1]
            receipt=[int(self.bits[i])^int(rng.random()<.1) for i in range(2)]
            # Both identities always returned, regardless of what secretly changed.
            for i,bit in enumerate(receipt):self.events.append(self.record('repair',i,bit))
            self.cost['repairs']+=.6;self.cost['observations']+=.02
        self.actions.append({'action':int(action),'kind':kind,'before':before.tolist(),'after':self.bits.tolist(),'receipt':receipt})
        self.step+=1;return receipt

    def finish(self):
        costs=dict(self.cost)
        if not self.stopped:costs['terminal']=20*int(self.bits.sum())
        return {'seed':self.seed,'index':self.index,'completion':bool(self.stopped and not self.bits.any()),'physical_repair':bool(not self.bits.any()),
                'faults':int(self.bits.sum()),'stopped':self.stopped,'costs':costs,'cost':float(sum(costs.values())),
                'actions':self.actions,'initial_public':[dict(e) for e in self.events[:self.history]],'parents':self.parents.tolist(),'identities':self.identities.tolist()}
