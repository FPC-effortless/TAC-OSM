"""Shared edge-physical research interface with read-only command queries.

A source-authorized observed history (x_t, EXECUTED u_{t-1}) is applied to
the inherited source-scheduled parallel TAC memory. At the query time, any
candidate u_t is projected through a read-only action-response head; the
candidate is NEVER written as an observed command or state transition.

No factual action effect or safe actuation follows from the forward pass.
This is an UNTRAINED model proposal. For source timing and real-fit obligations,
see research/directions/SHARED-EDGE-PHYSICAL-PRIORITY.md.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from .evidence_retrieval_world import EvidenceRetrievalWorld,EvidenceState


@dataclass(frozen=True)
class EdgeObservedState:
    """608 floating bytes at width=88; provenance integers are extra."""
    evidence: EvidenceState
    owner: torch.Tensor                 # int64[B], source entity identity
    last_hour: torch.Tensor             # int64[B], most recent REAL source observation
    model_token: int
    parameter_versions: tuple[int, ...]


@dataclass(frozen=True)
class ConditionalStateForecast:
    """[B,Q,D] point proposal; NO trained policy, causal claim or uncertainty."""
    mean: torch.Tensor
    trained_action_policy: bool = False
    causal_action_effect_established: bool = False
    autonomous_actuation_authorized: bool = False


class ActionQueryEdgeWorld(nn.Module):
    """TAC observed-state scan + vectorized counterfactual-SHAPED advisory read.

    Supervision, if separately authorized, must use actual x[t+1] paired to
    known-at-time ORIGINAL action[t], NOT the historical action[t-1] used to
    summarize past observations. Conditioning on an action is observational
    without randomized/sequential identification. Predictions do not mutate
    carried world-state evidence.
    """
    def __init__(self,state_dim: int=14,action_dim: int=14):
        super().__init__()
        if type(state_dim) is not int or type(action_dim) is not int or state_dim<1 or action_dim<1:
            raise ValueError("positive source dimensions required")
        self.state_dim=state_dim
        self.action_dim=action_dim
        self.core=EvidenceRetrievalWorld(state_dim=state_dim,action_dim=action_dim)
        # Reuse core encoder/address for proposed-command read-only retrieval.
        features=2*state_dim+action_dim+48+16
        self.command_head=nn.Sequential(
            nn.Linear(features,32),nn.SiLU(),nn.Linear(32,state_dim)
        )
        learned=sum(p.numel() for p in self.parameters())
        if not 95_000<=learned<=105_000:
            raise ValueError(f"trained/deployed parameter budget not met: {learned}")
        self.active_parameter_count=learned

    def _versions(self):
        return tuple(p._version for p in self.parameters())

    def initial_state(self,owner:torch.Tensor):
        if owner.ndim!=1 or owner.dtype!=torch.int64 or torch.any(owner<0):
            raise ValueError("owner must be nonnegative int64[B]")
        p=next(self.parameters())
        if owner.device!=p.device:raise ValueError("owner device mismatch")
        batch=owner.numel()
        return EdgeObservedState(self.core.initial_state(batch),owner.clone(),
            torch.full((batch,),-2**60,dtype=torch.int64,device=owner.device),
            id(self),self._versions())

    def forward(self,observed_states:torch.Tensor,executed_previous:torch.Tensor,
                candidate_commands:torch.Tensor,observed_hours:torch.Tensor,
                owner:torch.Tensor,*,state:EdgeObservedState|None=None,
                valid:torch.Tensor|None=None,return_trace:bool=False):
        if observed_states.ndim!=3 or observed_states.shape[-1]!=self.state_dim:
            raise ValueError("observations must be [B,T,state_dim]")
        B,T,D=observed_states.shape
        if T<1 or candidate_commands.ndim!=3 or candidate_commands.shape[:1]!=(B,) or (
            candidate_commands.shape[-1]!=self.action_dim or candidate_commands.shape[1]<1):
            raise ValueError("candidate commands must be [B,Q,action_dim]")
        if executed_previous.shape!=(B,T,self.action_dim):
            raise ValueError("history contains *previous executed* commands [B,T,action_dim]")
        if observed_hours.shape!=(B,T) or observed_hours.dtype!=torch.int64 or (
            owner.shape!=(B,) or owner.dtype!=torch.int64 or torch.any(owner<0)):
            raise ValueError("chronological int64 hours and source owners required")
        p=next(self.parameters())
        for z in (observed_states,executed_previous,candidate_commands):
            if z.dtype!=p.dtype or z.device!=p.device or not torch.isfinite(z).all():
                raise ValueError("matching finite observed/candidate floating input required")
        if observed_hours.device!=p.device or owner.device!=p.device:
            raise ValueError("source provenance device mismatch")
        if T>1 and torch.any(observed_hours[:,1:]<=observed_hours[:,:-1]):
            raise ValueError("source observation clock not strictly increasing")
        if valid is None:valid=torch.ones((B,T),device=p.device,dtype=torch.bool)
        if valid.shape!=(B,T) or valid.dtype!=torch.bool or valid.device!=p.device or not valid[:,-1].all():
            raise ValueError("query must end on a REAL valid observation, not padding")
        state=self.initial_state(owner) if state is None else state
        if not isinstance(state,EdgeObservedState) or state.model_token!=id(self) or (
            state.parameter_versions!=self._versions()):
            raise ValueError("foreign/stale model evidence")
        if state.owner.shape!=owner.shape or not torch.equal(state.owner,owner) or (
            state.last_hour.shape!=(B,) or torch.any(observed_hours[:,0]<=state.last_hour)):
            raise ValueError("foreign owner, replayed or nonmonotone observation")
        # The causal scan is conditioned only on REAL historical observations and
        # lagged EXECUTED commands, NOT proposed current command candidates.
        base,world=self.core(observed_states,executed_previous,
                             valid=valid,state=state.evidence,return_state=True)
        query_state=observed_states[:,-1]
        Q=candidate_commands.shape[1]
        expanded=query_state[:,None].expand(B,Q,D)
        query_encoded=self.core.encoder(torch.cat((expanded,candidate_commands),-1)).tanh()
        key=F.normalize(self.core.address(query_encoded),dim=-1,eps=1e-6)
        retrieved=(world.memory[:,None]@key[...,None]).squeeze(-1)
        # Include inherited complete-model output to keep its original trained
        # pathway live; do not use hidden simulated future truth.
        inherited=base[:,-1,None,:].expand(B,Q,D)
        features=torch.cat((expanded,candidate_commands,inherited,query_encoded,key,retrieved),-1)
        delta=self.command_head(features).tanh()
        prediction=inherited+self.core.delta_scale*delta
        final=EdgeObservedState(world,owner.clone(),observed_hours[:,-1].clone(),
                                id(self),self._versions())
        forecast=ConditionalStateForecast(prediction)
        if return_trace:
            return forecast,final,dict(query_key=key,read=retrieved,
                world_memory=world.memory,observation_count=world.count,
                statement="candidate action is read-only; no causal identification")
        return forecast,final


def instantiate_edge_research_profile(name:str,*,state_dim:int=14,action_dim:int=14):
    """Six registry family labels unchanged; no inference-time skill promotion."""
    if name=="temporal_sensor":
        from .belief_sensor_world import BeliefSensorWorld
        if (state_dim,action_dim)!=(14,14):
            raise ValueError("sensor profile has independent fixed observation schema")
        return BeliefSensorWorld()   # ME101 source belief hypothesis, not control
    if name in ("robot_physical","vehicle_physical"):
        return ActionQueryEdgeWorld(state_dim=state_dim,action_dim=action_dim)
    raise ValueError("choose qualified source schema, not an imaginary universal head")
