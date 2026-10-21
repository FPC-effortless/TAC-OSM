"""ME046 jointly trained multiscale forecasting: full context, GRU, native and
conservatively loop-refined native/GRU complete end-to-end models.

Uses unchanged native 32/32 fast/slow + 8x8 innovation memory, fixed 16-mode
spline/harmonic basis and original 96 observed real measurement windows.
DeepLoop-inspired tied residual refinement is a tested hypothesis, NOT a
claimed implementation or result of DeepLoop on Transformers.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from tac_osm.me044_multiscale_decoder import MultiscaleForecaster
from tac_osm.me040_dualrate_innovation import pair_observed_patches

ARMS=("joint_context","joint_gru16","joint_gru40",
      "joint_native","loop_gru16","loop_native")
LOOP_ARMS=("loop_gru16","loop_native")
NATIVE_ARMS=("joint_native","loop_native")


class StableTiedRefiner(nn.Module):
    """Repeated internal latent refinement with bounded per-step residual.

    r_{k+1}=r_k + (sigmoid(eta)/K)*tanh(W LayerNorm(r_k)).
    Original external observations and persistent TAC recurrence are unchanged.
    """
    def __init__(self,width=64,steps=2):
        super().__init__()
        self.steps=steps
        self.norm=nn.LayerNorm(width)
        self.map=nn.Linear(width,width)
        self.eta=nn.Parameter(torch.tensor(-3.0))
        nn.init.orthogonal_(self.map.weight,gain=.2)
        nn.init.zeros_(self.map.bias)

    def forward(self,x):
        z=x
        alpha=torch.sigmoid(self.eta)/self.steps
        for _ in range(self.steps):
            z=z+alpha*torch.tanh(self.map(self.norm(z)))
        return z


class JointMultiscaleForecaster(MultiscaleForecaster):
    def __init__(self,arm,horizon,period):
        if arm not in ARMS:raise ValueError("ME046 arm unregistered")
        base=("pyramid_native" if arm in NATIVE_ARMS else
              "pyramid_gru" if arm in ("joint_gru16","joint_gru40","loop_gru16")
              else "pyramid_context")
        super().__init__(base,horizon,period)
        self.arm046=arm
        if arm in ("joint_gru16","loop_gru16"):
            self.branch=nn.GRU(16,16,batch_first=True)
            self.branch_projection=nn.Linear(16,64)
        if arm in LOOP_ARMS:
            self.refiner=StableTiedRefiner(64,2)

    def expert_from(self,feat):
        _,tokens,ctx,latent,window,_,scale=feat
        # True complete model: full-history context/AR is jointly trainable.
        # Detached branch inputs keep expert's gradient independent from
        # cheap-path feature update, preventing a moving optimizer shortcut.
        ctx=ctx.detach()
        latent=latent.detach()
        if self.arm046 in NATIVE_ARMS:
            state=self.branch(tokens,latent)
        elif self.arm046=="joint_context":
            state=self.branch(window)
        else:
            paired=pair_observed_patches(tokens)
            width=self.branch.hidden_size
            _,h=self.branch(paired,.1*torch.tanh(latent[:,:width]).unsqueeze(0))
            state=self.branch_projection(h[0])
        if self.arm046 in LOOP_ARMS:
            state=self.refiner(state)
        v=torch.cat((ctx,latent,state),-1)
        coeff=self.coefficients(v)
        trust=torch.sigmoid(self.trust(v))
        norm_resid=.5*trust*torch.tanh((coeff@self.forecast_modes)/.5)
        b,_,c=scale.shape
        physical=norm_resid.reshape(b,c,self.horizon).transpose(1,2)*scale
        return physical,coeff,trust

    def full_parts(self,x):
        feat=self.features(x)
        cheap,cheap_correction=self.cheap_from(feat)
        residual,coeff,trust=self.expert_from(feat)
        return cheap,cheap+residual,cheap_correction,residual,coeff,trust,feat[-1]

    def forward(self,x):
        return self.full_parts(x)[1]


def build_model(arm,horizon,seed,period,initial_context=None):
    torch.manual_seed(seed)
    model=JointMultiscaleForecaster(arm,horizon,period)
    if initial_context is not None:
        source=initial_context.state_dict()
        target=model.state_dict()
        for name,weights in source.items():
            if name not in target:
                raise KeyError("incompatible common context: "+name)
            target[name]=weights.detach().clone()
        model.load_state_dict(target,strict=True)
    return model


def objective(model,x,y):
    """Joint real-data objective; unlike frozen ME045, train cheap and expert.

    DeepLoop-like iterative inner refinement is always applied only after
    complete native/GRU state update; no invented steps or real actions.
    """
    cheap,pred,delta,residual,coeff,_,scale=model.full_parts(x)
    loss_full=(pred-y).square().mean()
    loss_cheap=(cheap-y).square().mean()
    target=((y-cheap.detach())/scale).transpose(1,2).reshape(-1,model.horizon)
    target_coeff=(target@model.forecast_modes.T/model.horizon).detach()
    loss=loss_full+.35*loss_cheap
    loss=loss+.04*(coeff-target_coeff).square().mean()
    loss=loss+.01*residual.square().mean()+.01*delta.square().mean()
    loss=loss+10*model.anchor_loss()
    return loss,loss_full,loss_cheap


def clip_trainable(model):
    cheap=[p for n,p in model.named_parameters()
           if not n.startswith(("branch.","branch_projection.","coefficients.",
                                "trust.","refiner."))]
    extra=[p for n,p in model.named_parameters()
           if n.startswith(("branch.","branch_projection.","coefficients.",
                            "trust.","refiner."))]
    c=torch.nn.utils.clip_grad_norm_(cheap,1.)
    e=torch.nn.utils.clip_grad_norm_(extra,1.)
    return torch.maximum(c,e)


def preflight():
    """Constructed tensors check software invariants, NOT model performance."""
    torch.set_num_threads(1)
    torch.manual_seed(46046)
    counts={}
    for horizon in (96,720):
        counts[str(horizon)]={}
        models={arm:build_model(arm,horizon,46001,24) for arm in ARMS}
        for arm,model in models.items():
            counts[str(horizon)][arm]=sum(p.numel() for p in model.parameters())
            x=torch.randn(2,96,3);y=torch.randn(2,horizon,3)
            assert torch.isfinite(model(x)).all()
            # At initialization same cheap context and all-zero final residual.
            cheap,pred,*_=model.full_parts(x)
            assert torch.equal(cheap,pred)
            loss,lf,lc=objective(model,x,y)
            assert torch.isfinite(loss)
            model.zero_grad(set_to_none=True);loss.backward()
            for name,p in model.named_parameters():
                if p.grad is None or not bool(torch.isfinite(p.grad).all()):
                    raise AssertionError(f"inactive {arm}.{name}")
            assert model.branch is not None
    a=build_model("loop_native",96,46001,24)
    b=build_model("joint_native",96,46001,24)
    for name,p in b.state_dict().items():
        assert torch.equal(p,a.state_dict()[name])
    # True tied 2-step loop with bounded total coordinate shift.
    z=torch.randn(2,64)
    refine=a.refiner
    with torch.no_grad():
        diff=(refine(z)-z).abs().max().item()
        assert diff<=float(torch.sigmoid(refine.eta))+1e-6
    return {"status":"ME046_NUMERICAL_MATH_ONLY_PASS",
            "arms":ARMS,"count":counts,
            "weights_tied_refinement_steps":2,
            "all_native_legacy_coefficients_preserved":True,
            "real_model_scores_computed":False}
