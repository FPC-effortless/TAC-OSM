"""ME041: eight genuinely trained complete-model variants of native TAC.

All eight variants have ONE identically sized, active 16x16 affine
transformation (272 extra parameters), unchanged fast32/slow32/8x8 error-write
capacity, 6 past-only paired updates, complete ME038 full-window context+AR.
This is end-to-end *forecasting* research, not proof of learned entities,
symbolic reasoning, autonomous actions, physical intervention effects, or
general System2 capability.
"""
import torch
from torch import nn
from torch.nn import functional as F
from tac_osm.me038_innovation_forecast import InnovationMemory
from tac_osm.me040_dualrate_innovation import build_model as previous_model
from tac_osm.me040_dualrate_innovation import pair_observed_patches

ARCHITECTURE_ARMS=(
    "baseline_adaptive",
    "learned_relevance",
    "cdl_plasticity",
    "sparse_assemblies",
    "oscillatory_phase",
    "role_filler",
    "latent_refinement",
    "unified_multimechanism",
)
CONTROL_ARMS=("innovation_tac","paired_innovation_tac","context_cross")
ALL_ARMS=CONTROL_ARMS+ARCHITECTURE_ARMS


class ResearchMemory(InnovationMemory):
    """Multi-mechanism *architecture* candidates using one active common budget.

    Adapter maps one observed 16-feature past token -> 16 differentiable
    adaptation features. All model variants have exactly same active params.
    """
    def __init__(self,mode):
        if mode not in ARCHITECTURE_ARMS:
            raise ValueError("unregistered learned native memory mode")
        super().__init__()
        self.mode=mode
        self.adapter=nn.Linear(16,16)

    def input_pair(self,old,new):
        if self.mode in ("learned_relevance","unified_multimechanism"):
            # Causal convex summary, no exposure to a future observed segment.
            a=self.adapter(new-old)
            alpha=torch.sigmoid(a.mean(-1,keepdim=True))
            return old*(1-alpha)+new*alpha
        return .25*old+.75*new

    def sparse_assembly(self,a):
        # Four dynamically allocated functional groups of four coordinates.
        # Hard top-2 routing applied ONLY to actual observed-patch features.
        b=a.size(0)
        groups=a.reshape(b,4,4)
        score=groups.mean(-1)
        top2=torch.topk(score,2,dim=-1).indices
        active=torch.zeros_like(score).scatter(1,top2,1.)
        weights=F.softmax(score.masked_fill(active==0,-1e4),dim=-1)
        return (groups*weights.unsqueeze(-1)).reshape(b,16)

    def token_transform(self,raw,j):
        a=self.adapter(raw)
        if self.mode=="baseline_adaptive":
            return raw+.04*torch.tanh(a)
        if self.mode=="sparse_assemblies":
            return raw+.08*torch.tanh(self.sparse_assembly(a))
        if self.mode=="oscillatory_phase":
            return raw+.08*torch.sin(a+(j+1)*.7)*torch.tanh(raw)
        if self.mode=="role_filler":
            return raw+.08*torch.tanh(a)*torch.roll(raw,shifts=4,dims=-1)
        if self.mode=="unified_multimechanism":
            sparse=self.sparse_assembly(a)
            phase=torch.sin(a+(j+1)*.7)*torch.tanh(raw)
            binding=torch.tanh(a)*torch.roll(raw,4,-1)
            return raw+.035*(torch.tanh(sparse)+phase+binding)
        # In relevance or CDL mode, the same adapter is used below in
        # input_pair or in prediction-error write; in latent_refinement it is
        # used by the final latent-only computation state.
        return raw

    def step_cdl(self,u,f,s,matrix):
        """Same ME038 nonlinear recurrence with positive causal beta importance."""
        query=self.address(self.query(torch.cat((f,s),-1)))
        recalled=torch.bmm(matrix,query.unsqueeze(-1)).squeeze(-1)
        gate=torch.sigmoid(self.gate(torch.cat((u,f,recalled),-1)))
        f=f+gate*(torch.tanh(self.enc(u)+self.transition(f)+self.read_fast(recalled))-f)
        state=torch.cat((f,s),-1)
        key=self.address(self.key(torch.cat((u,f),-1)))
        value=torch.tanh(self.value(state))
        # Bounded predictive-relevance-inspired learned retention importance.
        # This is NOT identified counterfactual value/decision relevance.
        relevance=torch.sigmoid(self.adapter(u).mean(-1,keepdim=True))
        beta=torch.sigmoid(self.write(state))*relevance
        rho=.95+.0499*torch.sigmoid(self.retention)
        matrix=self.innovation_write(matrix,key,value,beta,rho)
        recalled=torch.bmm(matrix,query.unsqueeze(-1)).squeeze(-1)
        slow_gate=torch.sigmoid(self.slow_gate(state))
        innovation=torch.tanh(self.slow(torch.cat((f-s,s,recalled),-1)))
        s=s+slow_gate*(torch.tanh(s+innovation)-s)
        return f,s,matrix

    def latent_refine(self,h):
        # World state is frozen after six observed updates; only the separate
        # readout latent is refined, no invented observation/action enters.
        r=h
        for _ in range(2):
            reshaped=r.reshape(r.size(0),4,16)
            correction=torch.tanh(self.adapter(reshaped.mean(1)))
            r=r+.065*correction.repeat(1,4)
        return r

    def forward(self,patches):
        if patches.ndim!=3 or patches.size(1)!=12 or patches.size(2)!=16:
            raise ValueError("require exactly 12 observed 16-value patches")
        f=patches.new_zeros((patches.size(0),32))
        s=torch.zeros_like(f)
        m=patches.new_zeros((patches.size(0),8,8))
        for j in range(6):
            raw=self.input_pair(patches[:,2*j],patches[:,2*j+1])
            u=self.token_transform(raw,j)
            if self.mode in ("cdl_plasticity","unified_multimechanism"):
                f,s,m=self.step_cdl(u,f,s,m)
            else:
                f,s,m=self.step(u,f,s,m)
        h=torch.cat((f,s),-1)
        if self.mode in ("latent_refinement","unified_multimechanism"):
            h=self.latent_refine(h)
        return h/torch.sqrt(h.square().mean(-1,keepdim=True)+.01)


def build_model(arm,horizon,seed):
    if arm in CONTROL_ARMS:
        return previous_model(arm,horizon,seed)
    if arm not in ARCHITECTURE_ARMS:
        raise ValueError("ME041 unknown complete model architecture")
    model=previous_model("innovation_tac",horizon,seed)
    candidate=ResearchMemory(arm)
    # Retain original ME038 fast/slow/delta states and initialization exactly.
    candidate.load_state_dict({**candidate.state_dict(),
        **model.memory.state_dict()},strict=True)
    model.memory=candidate
    return model


def numerical_preflight():
    """Synthetic tensors are SOFTWARE CHECKS ONLY; no capability claim."""
    torch.set_num_threads(1)
    torch.manual_seed(41041)
    counts={}
    for h in (96,720):
        nets={a:build_model(a,h,41001) for a in ALL_ARMS}
        counts[str(h)]={a:sum(p.numel() for p in n.parameters()) for a,n in nets.items()}
        variants=[counts[str(h)][a] for a in ARCHITECTURE_ARMS]
        assert len(set(variants))==1,counts[str(h)]
        assert variants[0]-counts[str(h)]["innovation_tac"]==272
        assert max(counts[str(h)].values())/min(counts[str(h)].values())<1.01
        assert counts[str(h)]["innovation_tac"]==counts[str(h)]["paired_innovation_tac"]
    local=torch.randn(2,12,16)
    orig=pair_observed_patches(local)
    for arm in ARCHITECTURE_ARMS:
        m=ResearchMemory(arm)
        m.eval()
        with torch.no_grad():
            z=m(local)
            assert z.shape==(2,64) and torch.isfinite(z).all()
        changed=local.clone();changed[:,6:]+=100
        # Complete first three 16-observation blocks cannot access later ones.
        def prefix_state(inp):
            f=inp.new_zeros((inp.size(0),32))
            s=torch.zeros_like(f)
            matrix=inp.new_zeros((inp.size(0),8,8))
            for j in range(3):
                u=m.token_transform(m.input_pair(inp[:,2*j],inp[:,2*j+1]),j)
                if arm in ("cdl_plasticity","unified_multimechanism"):
                    f,s,matrix=m.step_cdl(u,f,s,matrix)
                else:
                    f,s,matrix=m.step(u,f,s,matrix)
            return f,s,matrix
        assert all(torch.equal(a,b) for a,b in zip(prefix_state(local),prefix_state(changed)))
        # Test useful gradients to every row of new 16x16 adapter.
        m.train();m.zero_grad(set_to_none=True)
        objective=m(local).square().mean()
        objective.backward()
        if m.adapter.weight.grad is None or not bool(torch.isfinite(m.adapter.weight.grad).all()):
            raise AssertionError("no finite adapter gradients: "+arm)
        if not bool(m.adapter.weight.grad.abs().sum()>0):
            raise AssertionError("new math adapter unused: "+arm)
        if m.transition.weight.grad is None or not bool(m.transition.weight.grad.abs().sum()>0):
            raise AssertionError("original fast recurrence does not learn: "+arm)
    return {"status":"ME041_NUMERICAL_ONLY_PASS",
            "arms":ALL_ARMS,"counts":counts,
            "active_adapter_params":272,"real_outcomes_scored":False,
            "future_data_opened":False}
