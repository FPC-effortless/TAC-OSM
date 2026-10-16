"""DEV-032 engineering Test6: actual closed loop, random weights, no capability claim.

Public single-entity maintenance differs from the all-entity DEV-031 task.
CDL routes relevance; OSM forecasts latent consequences; verifier/repair are
separate. This uses preserved neural CASM, not the original Boolean executor.
"""
from dataclasses import dataclass
import copy
import torch
from torch import nn
from .delayed_maintenance import Action,PublicObservation
from .schema_maintenance import future_bits
from .full_plm import DevelopmentPLM
from .full_plm_reference import FullPLMConfig

ARMS=('full','memory_reset','cdl_uniform','casm_mlp','osm_off','commit_all','repair_off')
KINDS=('inspect','change','scheduled','wait')


@dataclass(frozen=True)
class PublicRequest:
    entity:int
    horizon:int


def public_tokens(observation,request):
    """Only immutable public observation and explicitly supplied public query."""
    if not isinstance(observation,PublicObservation) or not isinstance(request,PublicRequest):
        raise TypeError('public boundary requires observation and request')
    tokens=[1+observation.entity,10+KINDS.index(observation.kind),20+observation.value+1,70+request.entity]
    for value in (observation.tick,max(0,observation.due),request.horizon):
        if not 0<=value<65536:raise ValueError('registered timestamp encoding exceeded')
        tokens.extend(32+((value>>shift)&15) for shift in (0,4,8,12))
    return torch.tensor([tokens],dtype=torch.long)


class MaintenanceSession:
    """Evaluator-owned world: no world object or future target enters policy."""
    def __init__(self,episode,request):
        self.request=request;self.history=episode.observations
        self._world=copy.deepcopy(episode.world);self._untouched=copy.deepcopy(episode.world)
        self.services=[];self.inspections=0;self.receipts=[]

    def act(self,action):
        if len(self.receipts)>=2:raise ValueError('initial action plus one repair only')
        entity=self.request.entity
        if action==0:command=Action('wait',entity)
        elif action==1:
            command=Action('service',entity,0,self.request.horizon);self.services.append(entity)
        elif action==2:
            command=Action('inspect',entity);self.inspections+=1
        else:raise ValueError('unknown public action')
        receipt=self._world.act(command);self.receipts.append(receipt)
        return receipt,True  # acceptance, not hidden prediction correctness

    def score(self):
        # Hidden worlds are evaluated only AFTER every action is fixed.
        entity=self.request.entity
        outcome=future_bits(self._world,self.request.horizon)[entity]
        untreated=future_bits(self._untouched,self.request.horizon)[entity]
        return dict(target_failures=outcome,services=len(self.services),
            unnecessary_services=sum(untreated==0 for _ in self.services),inspections=self.inspections,
            total_cost=4*outcome+len(self.services)+.1*self.inspections,
            failure_scope='queried public entity only',all_entity_benchmark=False)


class Test6Controller(nn.Module):
    __test__=False
    def __init__(self,seed=32032):
        super().__init__();torch.manual_seed(seed)
        self.model=DevelopmentPLM(FullPLMConfig(vocab_size=128,d_model=24,num_heads=4,action_count=3))
        self.mlp=nn.Sequential(nn.Linear(48,48),nn.Tanh(),nn.Linear(48,24))
        self.eval()

    def observe(self,observation,request,state):
        return self.model.observe(public_tokens(observation,request),torch.zeros(1,1,16,16),torch.zeros(1,1,64),state)

    def decide(self,read,experts,arm):
        if arm=='cdl_uniform':
            selected=experts;weights=read.new_full(experts.shape[:2],1/experts.shape[1])
        else:selected,weights,_=self.model.cdl(read,experts)
        if arm=='casm_mlp':computation=self.mlp(torch.cat((read,(selected*weights.unsqueeze(-1)).sum(1)),-1))
        else:computation,*_=self.model.casm(read,selected,weights)
        base=self.model.action_head(torch.cat((computation,read),-1))
        if arm=='osm_off':return base
        actions=torch.arange(3);states=read.expand(3,-1)
        predicted=self.model.osm.predict(states,actions,read.new_zeros(3))
        goal=self.model.goal(read).expand(3,-1)
        planning=self.model.plan_head(torch.cat((computation.expand(3,-1),predicted,goal),-1)).view(1,3)
        return base+planning

    @torch.no_grad()
    def rollout(self,session,arm='full',force_gate=None):
        if arm not in ARMS or force_gate not in (None,0.,1.):raise ValueError('unregistered arm or fixture')
        state=None;read=experts=None;trace=[]
        for observation in session.history:
            if arm=='memory_reset':state=None
            state,read,experts=self.observe(observation,session.request,state)
        logits=self.decide(read,experts,arm);probabilities=logits.softmax(-1)
        action=int(logits.argmax(-1));repair_executed=False;rejections=0;acceptances=0
        for attempt in range(2):
            receipt,accepted=session.act(action)
            state,post_read,experts=self.observe(receipt,session.request,state)
            outcome=read.new_tensor([float(accepted)])
            verification=float(self.model.verifier(read,post_read,probabilities,outcome).sigmoid())
            gate=force_gate if force_gate is not None else verification
            if arm=='commit_all':gate=1.
            before=state.memory.clone()
            state=self.model.commit(state,post_read,torch.tensor([action]),outcome,read.new_tensor([gate]))
            commit_accepted=gate>=self.model.config.verifier_threshold
            if commit_accepted:acceptances+=1
            else:
                rejections+=1
                if not torch.equal(before,state.memory):raise AssertionError('rejected commit changed memory')
            trace.append(dict(attempt=attempt,action=action,public_receipt=receipt.__dict__,
                receipt_acceptance=accepted,learned_verifier_probability=verification,
                commit_accepted=commit_accepted,forced_gate_fixture=force_gate is not None))
            if attempt or commit_accepted or arm=='repair_off':break
            repair_logits=self.model.repair(post_read,probabilities,outcome)
            action=int(repair_logits.argmax(-1));probabilities=repair_logits.softmax(-1)
            repair_executed=True;read=post_read
        if not torch.isfinite(state.memory).all():raise AssertionError('nonfinite integrated state')
        return dict(arm=arm,initial_action_logits=logits.tolist(),trace=trace,
            repair_executed=repair_executed,accepted_commits=acceptances,rejected_commits=rejections,
            forced_gate_fixture=force_gate is not None,cost=session.score(),
            memory_state_scalars=state.memory.numel(),capability_validated=False,
            original_cdl_teacher_used=False,original_boolean_casm_used=False,
            implementation='DevelopmentPLM learned routing/neural CASM/OSM; actual public receipt loop',
            future_targets_supplied_to_policy=False)
