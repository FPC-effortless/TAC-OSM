"""ME028 nonlinear observable readout only; exact native fast/slow dynamics unchanged.

Outcome Brier on real events required; mathematical preflight isn't evidence.
"""
from __future__ import annotations
import torch
from torch import nn
from tac_osm.state_value import StateValueModel


class NativeNonlinearObservableReadout(StateValueModel):
    """ME027 native recurrent body + 12-d tanh observable correction (310 params)."""
    def __init__(self, mean: float, scale: float, popularity: torch.Tensor):
        super().__init__("tac", mean, scale, popularity)
        self.head = nn.Linear(64, 1, bias=False)
        self.observable_hidden = nn.Linear(26, 12)
        self.observable_output = nn.Linear(12, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.observable_output.weight)
        nn.init.zeros_(self.observable_output.bias)

    @torch.jit.export
    def predict_features(self, latent: torch.Tensor, common: torch.Tensor,
                         prior: torch.Tensor) -> torch.Tensor:
        return (self.head(latent) +
                self.observable_output(torch.tanh(self.observable_hidden(common))) +
                torch.logit(prior))


def original_parameter_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def native_nonlinear_math_preflight():
    """Software-only numerical derivatives, causal prefix, active gradients."""
    torch.set_num_threads(1)
    torch.manual_seed(28028)
    pop=torch.full((16000,), 0.4)
    a=StateValueModel("tac",0.6,0.2,pop)
    b=NativeNonlinearObservableReadout(0.6,0.2,pop)
    c=StateValueModel("mlp",0.6,0.2,pop)
    counts=[original_parameter_count(m) for m in (a,b,c)]
    if counts != [135067,135377,135057]:
        raise AssertionError(f"ME028 active params changed: {counts}")
    if (b.head.weight.numel(), b.observable_hidden.weight.numel(),
            b.observable_hidden.bias.numel(),
            b.observable_output.weight.numel(),
            b.observable_output.bias.numel()) != (64,312,12,12,1):
        raise AssertionError("head coefficient decomposition changed")
    # Constructed samples are NOT evidence of prediction performance.
    torch.manual_seed(28029)
    items=torch.randint(16000,(2,128))
    meta=torch.rand(2,128,3)
    meta[:,:,1]=(meta[:,:,1]>.5).float()
    current=torch.randint(16000,(2,))
    duration=torch.rand(2,1)
    for net in (a,b,c):
        net.eval()
        lg=net(items,meta,current,duration)
        prior=torch.logit((meta[:,-32:,1].sum(-1,keepdim=True)+1)/34)
        if not torch.allclose(lg,prior,atol=1e-6,rtol=1e-6):
            raise AssertionError("non-identical zero-head prior initialization")
    # Set original native core weights identical to isolate head path for parity.
    ka, kb=a.state_dict(),b.state_dict()
    b.load_state_dict({**kb,**{
        n:v for n,v in ka.items() if n in kb and not n.startswith("head.")
    }},strict=True)
    with torch.no_grad():
        sa=a.latent_features(items,meta)
        sb=b.latent_features(items,meta)
        if not torch.allclose(sa,sb,atol=1e-7,rtol=1e-7):
            raise AssertionError("original learned state altered by nonlinear readout")
    scripted=torch.jit.script(b)
    if not torch.allclose(b(items,meta,current,duration),
                          scripted(items,meta,current,duration),atol=1e-6):
        raise AssertionError("TorchScript changes native readout")
    # Nonzero head test prevents zero-head causality tests being vacuous.
    with torch.no_grad():
        b.head.weight.copy_(torch.randn_like(b.head.weight)*0.1)
        b.observable_output.weight.copy_(torch.randn_like(b.observable_output.weight)*0.1)
        b.observable_output.bias.fill_(.17)
    b.eval()
    base=b.prefix_logits(items,meta,current,duration)
    if base.shape!=(2,4): raise AssertionError("prefix output shape changed")
    for depth,col in ((32,0),(64,1),(96,2)):
        altered_items=items.clone();altered_meta=meta.clone()
        altered_items[:,depth+1:]=torch.randint(16000,altered_items[:,depth+1:].shape)
        altered_meta[:,depth+1:]=torch.rand_like(altered_meta[:,depth+1:])
        altered_meta[:,depth+1:,1]=(altered_meta[:,depth+1:,1]>.5).float()
        # The next item identity and duration at index depth are known query
        # inputs and MUST be held fixed. Only its unknown watch response is
        # perturbed; later fully unobserved events may change arbitrarily.
        altered_meta[:,depth,1]=1.-altered_meta[:,depth,1]
        changed=b.prefix_logits(altered_items,altered_meta,current,duration)
        if not torch.allclose(base[:,col],changed[:,col],atol=3e-6,rtol=3e-6):
            raise AssertionError(f"ME028 future event leaked through {depth} prefix")
    # x is a standardized 26-vector; coord 8 is standardized duration.
    x=torch.randn(1,26,dtype=torch.float64,requires_grad=True)
    state=torch.randn(1,64,dtype=torch.float64)
    net=b.double().eval()
    prior=torch.full((1,1),.4,dtype=torch.float64)
    derivative=torch.autograd.grad(net.predict_features(state,x,prior).sum(),x,create_graph=True)[0][0,8]
    second=torch.autograd.grad(derivative,x)[0][0,8]
    z=net.observable_hidden(x)
    v=net.observable_output.weight[0,:]
    A=net.observable_hidden.weight[:,8]
    manual_first=torch.sum(v*(1-torch.tanh(z[0]).square())*A)
    manual_second=torch.sum(-2*v*torch.tanh(z[0])*(1-torch.tanh(z[0]).square())*A.square())
    if not torch.allclose(derivative,manual_first,atol=1e-10):
        raise AssertionError("observable nonlinear head first derivative mismatch")
    if not torch.allclose(second,manual_second,atol=1e-10):
        raise AssertionError("observable nonlinear head second derivative mismatch")
    net.float().train()
    pred=net(items,meta,current,duration)
    target=torch.tensor([[0.],[1.]])
    loss=nn.functional.binary_cross_entropy_with_logits(pred,target)
    loss.backward()
    if (net.transition.weight.grad is None or
            net.head.weight.grad is None or
            net.observable_hidden.weight.grad is None or
            net.observable_output.weight.grad is None):
        raise AssertionError("native recurrent or nonlinear readout inactive")
    if (net.transition.weight.grad.norm()<=0 or
            net.observable_hidden.weight.grad.norm()<=0):
        raise AssertionError("no original-state/nonlinear gradients")
    return {"status":"ME028_MATH_AND_PREFIX_FALSIFIER_PASS",
            "active_model_parameters":{"tac_linear":counts[0],
                                       "tac_nonlinear":counts[1],
                                       "mlp252":counts[2]},
            "readout_extra":310,"recurrence_unchanged":True,
            "new_real_predictions_scored":False,"reserved_labels_read":False}
