"""ME040 six-step native TAC with paired causal innovation memory.

Retains every ME038 coefficient and exact nonlinear fast/slow/delta memory step.
Changing the number and content of recurrent input tokens changes predictions;
this is a NEW TRAINED ARCHITECTURE, not an equivalent execution optimization.
"""
import torch
from tac_osm.me038_innovation_forecast import InnovationMemory, build_model as original_model


@torch.jit.script
def pair_observed_patches(patches: torch.Tensor) -> torch.Tensor:
    """Aggregate each adjacent pair of already-observed eight-step tokens.

    Convex weights: oldest 1/4, newest 3/4. Input (B,12,16);
    output (B,6,16). No unavailable future observations are used.
    Predictions are only defined after all 96 history observations arrive.
    """
    if patches.dim() != 3 or patches.size(1) != 12 or patches.size(2) != 16:
        raise ValueError("ME040 requires 12 genuine 16-feature observed patches")
    a = patches[:, 0::2, :]
    b = patches[:, 1::2, :]
    return (a + 3.0 * b) * 0.25


class PairedInnovationMemory(InnovationMemory):
    """Exact original nonlinear ME038 step applied to six paired tokens."""
    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        tokens = pair_observed_patches(patches)
        f = tokens.new_zeros((tokens.size(0), 32))
        s = torch.zeros_like(f)
        matrix = tokens.new_zeros((tokens.size(0), 8, 8))
        for t in range(6):
            f, s, matrix = self.step(tokens[:, t], f, s, matrix)
        state = torch.cat((f, s), dim=-1)
        return state / torch.sqrt(state.square().mean(-1, keepdim=True) + .01)


def build_model(arm: str, horizon: int, seed: int):
    """Original unchanged comparator or identical-initialization paired native.

    Keep the AR/context/head exactly as original ME038, and copy every
    initially generated original memory weight before changing forward().
    No inactive parameters or dummy matching coefficients.
    """
    if arm == "paired_innovation_tac":
        model = original_model("innovation_tac", horizon, seed)
        paired = PairedInnovationMemory()
        paired.load_state_dict(model.memory.state_dict(), strict=True)
        model.memory = paired
        return model
    if arm not in ("innovation_tac", "context_cross"):
        raise ValueError("unregistered ME040 model")
    return original_model(arm, horizon, seed)


def preflight():
    """Constructed numerical tests ONLY; never real-world accuracy evidence."""
    torch.set_num_threads(1)
    torch.manual_seed(40040)
    sizes = {}
    for h in (96,720):
        models = {arm:build_model(arm,h,40001) for arm in (
            "innovation_tac","paired_innovation_tac","context_cross")}
        sizes[str(h)] = {arm:sum(p.numel() for p in m.parameters())
                         for arm,m in models.items()}
        if sizes[str(h)]["paired_innovation_tac"] != sizes[str(h)]["innovation_tac"]:
            raise AssertionError("new memory secretly changed active parameter capacity")
        a = models["innovation_tac"]
        b = models["paired_innovation_tac"]
        # All model coefficient tensors are identical before training.
        for key,value in a.state_dict().items():
            if key not in b.state_dict() or not torch.equal(value,b.state_dict()[key]):
                raise AssertionError("initial native weight mismatch at "+key)
        if max(sizes[str(h)].values())/min(sizes[str(h)].values()) > 1.01:
            raise AssertionError("active model parameters differ more than 1%")
    p = torch.randn(2,12,16)
    q = pair_observed_patches(p)
    assert q.shape == (2,6,16)
    assert torch.equal(q, .25*p[:,0::2]+.75*p[:,1::2])
    for prefix in (2,4,6,8,10):
        changed=p.clone();changed[:,prefix:]+=123.
        if not torch.equal(pair_observed_patches(p)[:,:prefix//2],
                           pair_observed_patches(changed)[:,:prefix//2]):
            raise AssertionError("causal prefix depends on a future patch")
    const = torch.randn(2,1,16).expand(2,12,16)
    assert torch.equal(pair_observed_patches(const),const[:,:6])
    orig=build_model("innovation_tac",96,40001)
    pair=build_model("paired_innovation_tac",96,40001)
    for net in (orig,pair):
        net.eval()
        with torch.no_grad():torch.nn.init.normal_(net.out.weight,std=.01)
        x=torch.randn(2,96,3)
        script=torch.jit.script(net)
        with torch.no_grad():
            assert torch.allclose(net(x),script(x),atol=1e-6,rtol=1e-6)
        assert net.memory(torch.randn(2,12,16)).shape==(2,64)
        opt=torch.optim.AdamW(net.parameters(),lr=.001)
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            objective=net(x).square().mean()
            objective.backward()
            opt.step()
        for name,param in net.named_parameters():
            if param.grad is None or not torch.isfinite(param.grad).all():
                raise AssertionError("inactive or nonfinite "+name)
        if not any(n.startswith("memory.") and p.grad.abs().sum()>0
                   for n,p in net.named_parameters()):
            raise AssertionError("actual native memory has zero gradient")
    return {"status":"ME040_SOFTWARE_ONLY_MATH_PASS",
            "param_counts":sizes,"recurrent_steps_original":12,
            "recurrent_steps_paired":6,"original_weights_untouched":True,
            "new_real_fits_executed":False,
            "future_outcomes_read":False}
