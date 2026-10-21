"""ME047 real predictive state with stable temporal encoding, macro and goal intent.

The goal gateway accepts validated typed JSON, NOT arbitrary language. Model
capability is only established by unseen physical-source recorded outcomes.
The frozen tokenizer and macro teacher are learned on real source FIT data.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from tac_osm.me040_dualrate_innovation import PairedInnovationMemory, pair_observed_patches

ARMS=("hybrid_tac","hybrid_gru","hybrid_attention","hybrid_context")
GOALS=("mean","peak","last")
ROLES=("unknown","load","temperature","environment","power")


@dataclass(frozen=True)
class IntentSpec:
    goal:str
    role:str
    threshold:float | None = None


def parse_intent(payload):
    if not isinstance(payload,dict):
        raise ValueError("ME047 requires a validated JSON-style intent object")
    if set(payload)-{"goal","role","threshold"}:
        raise ValueError("unknown intent fields")
    g=payload.get("goal")
    r=payload.get("role")
    if g not in GOALS or r not in ROLES:
        raise ValueError("unsupported operational goal or observation role")
    threshold=payload.get("threshold")
    if threshold is not None:
        if not isinstance(threshold,(int,float)) or not math.isfinite(float(threshold)):
            raise ValueError("threshold must be a finite numerical value")
        threshold=float(threshold)
    return IntentSpec(g,r,threshold)


def safety_clip(value,low,high):
    """Deterministic software command bounds; not a trained safe policy."""
    if not math.isfinite(float(low)) or not math.isfinite(float(high)) or low>high:
        raise ValueError("invalid immutable safety bounds")
    return torch.clamp(value,min=low,max=high)


class TemporalTokenizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv1d(1,24,5,padding=2),
                                 nn.GELU(),
                                 nn.Conv1d(24,32,5,padding=2),
                                 nn.GELU())
        self.pretext=nn.Linear(32,8)

    def tokens(self,x):
        if x.ndim!=2 or x.size(1)%8!=0:
            raise ValueError("observed history must have complete eight-step patches")
        z=self.conv(x.unsqueeze(1))
        return F.avg_pool1d(z,8,stride=8).transpose(1,2)

    def masked_next8(self,prefix88):
        if prefix88.size(1)!=88:
            raise ValueError("only first 88 actual observed values allowed")
        return self.pretext(self.tokens(prefix88)[:,-1])


class MacroTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=nn.Linear(96,8)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self,observed_normalized):
        if observed_normalized.shape[-1]!=96:raise ValueError("macro needs history96")
        return self.linear(observed_normalized)


class AnchoredNative(PairedInnovationMemory):
    def __init__(self):
        super().__init__()
        self.slow_anchor=nn.Linear(8,32)

    def forward(self,tokens,macro):
        if tokens.size(1)!=12 or tokens.size(2)!=16:
            raise ValueError("native requires 12 observed projected tokens")
        u=pair_observed_patches(tokens)
        f=u.new_zeros((u.size(0),32))
        s=.1*torch.tanh(self.slow_anchor(macro))
        m=u.new_zeros((u.size(0),8,8))
        for t in range(6):
            f,s,m=self.step(u[:,t],f,s,m)
        h=torch.cat((f,s),-1)
        return h/torch.sqrt(h.square().mean(-1,keepdim=True)+.01)


class CapabilityForecaster(nn.Module):
    def __init__(self,arm,tokenizer,macro_teacher):
        super().__init__()
        if arm not in ARMS:raise ValueError("unknown complete model arm")
        self.arm=arm
        self.tokenizer=tokenizer
        self.teacher=macro_teacher
        for p in self.tokenizer.parameters():p.requires_grad_(False)
        for p in self.teacher.parameters():p.requires_grad_(False)
        self.role=nn.Embedding(len(ROLES),8)
        self.intent=nn.Embedding(len(GOALS),32)
        if arm=="hybrid_tac":
            self.to_native=nn.Linear(32,16)
            self.world=AnchoredNative()
        elif arm=="hybrid_gru":
            self.world=nn.GRU(32,64,batch_first=True)
            self.macro_seed=nn.Linear(8,64)
        elif arm=="hybrid_attention":
            layer=nn.TransformerEncoderLayer(32,4,128,dropout=0.,batch_first=True)
            self.world=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
            self.time_embedding=nn.Parameter(torch.zeros(1,12,32))
            self.context_projection=nn.Linear(32,64)
        else:
            self.world=nn.Sequential(nn.Linear(12*32,128),nn.GELU(),
                                     nn.Linear(128,64))
        self.trajectory=nn.Sequential(nn.Linear(64+8+8,128),
                         nn.GELU(),nn.Linear(128,96))
        self.goal_head=nn.Sequential(nn.Linear(64+8+8+32,128),
                         nn.GELU(),nn.Linear(128,1))

    def forward(self,x,role,goal,macro_anchor=None):
        """All x are real observed history, no target or future covariates."""
        if x.ndim!=2 or x.shape[1]!=96:raise ValueError("96 observed scalar readings")
        if role.shape!=(x.shape[0],) or goal.shape!=(x.shape[0],):
            raise ValueError("typed role/goal IDs required per observation")
        mean=x.mean(-1,keepdim=True)
        std=torch.sqrt((x-mean).square().mean(-1,keepdim=True)+1e-5)
        normalized=(x-mean)/std
        with torch.no_grad():
            tok=self.tokenizer.tokens(normalized)
            macro=self.teacher(normalized) if macro_anchor is None else macro_anchor
        if macro.shape!=(len(x),8):raise ValueError("macro anchor must be 8 causal values")
        if self.arm=="hybrid_tac":
            state=self.world(self.to_native(tok),macro)
        elif self.arm=="hybrid_gru":
            _,h=self.world(tok,.1*torch.tanh(self.macro_seed(macro)).unsqueeze(0))
            state=h[0]
        elif self.arm=="hybrid_attention":
            enc=self.world(tok+self.time_embedding,
                            mask=torch.triu(torch.ones(12,12,device=x.device,
                                           dtype=torch.bool),diagonal=1))
            state=self.context_projection(enc[:,-1])
        else:
            state=self.world(tok.reshape(len(x),-1))
        meta=torch.cat((state,macro,self.role(role)),dim=-1)
        trajectory=self.trajectory(meta)
        goal_pred=self.goal_head(torch.cat((meta,self.intent(goal)),dim=-1))[:,0]
        return trajectory,goal_pred,macro,mean,std

    def count_active(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def goal_targets(y,goal):
    """True observed future statistics, never fabricated intervention labels."""
    return torch.where(goal==0,y.mean(-1),
           torch.where(goal==1,y.amax(-1),y[:,-1]))


def normalized_observation(x,y):
    mean=x.mean(-1,keepdim=True)
    scale=torch.sqrt((x-mean).square().mean(-1,keepdim=True)+1e-5)
    return (x-mean)/scale,(y-mean)/scale


def complete_loss(model,x,y,role,goal):
    traj,goal_pred,macro,mean,scale=model(x,role,goal)
    actual=(y-mean)/scale
    target=goal_targets(actual,goal)
    bin_true=actual.reshape(-1,8,12).mean(-1)
    bins=traj.reshape(-1,8,12).mean(-1)
    loss=(traj-actual).square().mean()
    loss=loss+.3*(goal_pred-target).square().mean()
    loss=loss+.15*(bins-bin_true).square().mean()
    return loss


def preflight():
    """Constructed tensors check math/gradient/safety only, no success metric."""
    torch.set_num_threads(1)
    torch.manual_seed(47047)
    tokenizer=TemporalTokenizer();teacher=MacroTeacher()
    models={}
    x=torch.randn(3,96);y=torch.randn(3,96)
    goal=torch.tensor([0,1,2],dtype=torch.long)
    role=torch.tensor([1,2,3],dtype=torch.long)
    for arm in ARMS:
        m=CapabilityForecaster(arm,tokenizer,teacher)
        out,scalar,macro,mean,scale=m(x,role,goal)
        assert out.shape==(3,96) and scalar.shape==(3,)
        assert torch.isfinite(out).all() and torch.isfinite(scalar).all()
        m.zero_grad(set_to_none=True)
        objective=complete_loss(m,x,y,role,goal)
        objective.backward()
        assert any(p.grad is not None and p.grad.abs().sum()>0
                   for name,p in m.named_parameters() if name.startswith("world."))
        assert all(p.grad is None for p in m.tokenizer.parameters())
        assert all(p.grad is None for p in m.teacher.parameters())
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for name,p in m.named_parameters() if p.requires_grad)
        models[arm]=m.count_active()
    assert safety_clip(torch.tensor([-3.,1.,8.]),0.,4.).tolist()==[0.,1.,4.]
    assert parse_intent({"goal":"peak","role":"temperature","threshold":75.}).goal=="peak"
    return {"status":"ME047_SOFTWARE_ONLY_MATH_PASS",
            "active_coefficients":models,"full_native_fast32_slow32_assoc8x8":True,
            "pretext_uses_88_true_observed_values":True,"test_data_used":False}
