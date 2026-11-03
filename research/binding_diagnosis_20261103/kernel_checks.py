"""Independent boundary-derivative falsifiers; no optimization or model fitting."""
import torch
from write_repair import write_step as repaired


def require(condition, message):
    if not condition:
        raise ValueError(message)


def run(legacy):
    rows = []
    dtype = torch.float64
    memory = torch.zeros(1, 1, 2, dtype=dtype)
    precision = torch.eye(2, dtype=dtype)[None]
    key = torch.tensor([[1., 0.]], dtype=dtype)
    value = torch.tensor([[2.]], dtype=dtype)
    for target, expected in (("memory", 2.), ("precision", -1.)):
        found = {}
        for name, operation in (("legacy", legacy), ("repaired", repaired)):
            gain = torch.zeros(1, dtype=dtype, requires_grad=True)
            a, p = operation(memory, precision, key, value, gain)
            loss = a[0, 0, 0] if target == "memory" else p[0, 0, 0]
            found[name] = float(torch.autograd.grad(loss, gain)[0])
        epsilon = 1e-6
        a0, p0 = legacy(memory, precision, key, value, torch.zeros(1, dtype=dtype))
        a1, p1 = legacy(memory, precision, key, value, torch.full((1,), epsilon, dtype=dtype))
        difference = float(((a1-a0) if target == "memory" else (p1-p0)).flatten()[0]/epsilon)
        require(abs(found["legacy"]) < 1e-14, "legacy zero-gain defect no longer reproduced")
        require(abs(found["repaired"]-expected) < 1e-12, "repaired boundary derivative incorrect")
        require(abs(difference-expected) < 3e-6, "independent right derivative differs")
        rows.append({"check": target+"_zero_gain_boundary", "legacy_status": "FAIL_REPRODUCED", "legacy_autograd": found["legacy"], "candidate_status": "PASS", "candidate_autograd": found["repaired"], "analytic_right_derivative": expected, "finite_right_difference": difference})

    # Independent softmax Jacobian: d p_1 / d logits = [-p_0 p_1, p_0 p_1].
    expected = None
    router = {}
    for name, operation in (("legacy", legacy), ("repaired", repaired)):
        logits = torch.tensor([.4, -.2], dtype=dtype, requires_grad=True)
        probability = logits.softmax(-1)
        hard = torch.nn.functional.one_hot(probability.argmax(), 2).to(dtype)
        gain = hard+(probability-probability.detach())
        a, _ = operation(torch.zeros(2, 1, 2, dtype=dtype), torch.eye(2, dtype=dtype).repeat(2, 1, 1), key.repeat(2, 1), torch.tensor([[2.], [3.]], dtype=dtype), gain)
        router[name] = torch.autograd.grad(a[1, 0, 0], logits)[0].tolist()
        scale = float(3*probability[0]*probability[1])
        expected = [-scale, scale]
    require(max(abs(x) for x in router["legacy"]) == 0, "legacy unselected write gradient unexpectedly present")
    require(max(abs(a-b) for a, b in zip(expected, router["repaired"])) < 1e-12, "repaired straight-through chain incorrect")
    rows.append({"check": "unselected_bank_router_credit", "legacy_status": "FAIL_REPRODUCED", "candidate_status": "PASS", "legacy_logit_gradient": router["legacy"], "candidate_logit_gradient": router["repaired"], "independent_softmax_chain": expected})

    passed = []
    for dtype in (torch.float32, torch.float64):
        for kind in ("delta", "preconditioned"):
            torch.manual_seed(13001)
            m = torch.randn(2, 3, 4, dtype=dtype)
            m[0, 0, 0] = -0.
            p = torch.eye(4, dtype=dtype).repeat(2, 1, 1) if kind == "preconditioned" else None
            k = torch.randn(2, 4, dtype=dtype)
            v = torch.randn(2, 3, dtype=dtype)
            a, b = repaired(m, p, k, v, torch.zeros(2, dtype=dtype), kind=kind)
            require(torch.equal(a.view(torch.uint8), m.view(torch.uint8)), "zero-gain memory freeze changed bits")
            if p is not None:
                require(torch.equal(b.view(torch.uint8), p.view(torch.uint8)), "zero-gain precision freeze changed bits")
            passed.append(f"{dtype}_{kind}_bitwise_zero_freeze")
            # Away from zero, all derivatives must remain source-identical.
            inputs = [m, k, v, torch.tensor([.1, .7], dtype=dtype)]
            gradients = []
            outputs = []
            for operation in (legacy, repaired):
                mm, kk, vv, gg = [x.detach().clone().requires_grad_() for x in inputs]
                aa, pp = operation(mm, p, kk, vv, gg, kind=kind)
                loss = aa.square().sum() + (0 if pp is None else pp.square().sum())
                gradients.append(torch.autograd.grad(loss, (mm, kk, vv, gg)))
                outputs.append(aa)
            require(torch.equal(*outputs), "positive-gain forward changed")
            require(all(torch.equal(a, b) for a, b in zip(*gradients)), "positive-gain derivative changed")
            passed.append(f"{dtype}_{kind}_positive_gain_forward_and_gradient_exact")

    # A denied observation must not learn a write, even with the repaired adjoint.
    x = torch.tensor([.2], dtype=torch.float64, requires_grad=True)
    gain = x.sigmoid()*torch.zeros_like(x)
    a, _ = repaired(memory, precision, key, value, gain)
    require(float(torch.autograd.grad(a.sum(), x)[0]) == 0, "observation freeze leaked learning")
    passed.append("observation_mask_blocks_gate_learning")

    # Independent finite differences on the valid SPD-state manifold. Arbitrary
    # nonsymmetric precision perturbations are outside the source's state domain.
    torch.manual_seed(13002)
    m = torch.randn(2, 2, 3, dtype=torch.float64, requires_grad=True)
    chol = torch.eye(3, dtype=torch.float64).repeat(2, 1, 1).requires_grad_()
    k = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    v = torch.randn(2, 2, dtype=torch.float64, requires_grad=True)
    gain = torch.tensor([0., .2], dtype=torch.float64, requires_grad=True)
    def update(mm, ll, kk, vv, gg):
        return repaired(mm, ll@ll.transpose(-1, -2), kk, vv, gg)
    require(torch.autograd.gradcheck(update, (m, chol, k, v, gain), eps=1e-6, atol=2e-5, rtol=2e-4), "candidate finite-difference check failed")
    passed.append("FP64_gradcheck_including_zero_gain_valid_SPD_parameterization")
    return {"boundary_falsifiers": rows, "additional_checks": [{"check": x, "status": "PASS"} for x in passed], "candidate_checks_passed": len(rows)+len(passed), "legacy_expected_failures_retained": len(rows), "scope": "Declared smooth-candidate/straight-through adjoint, not differentiation of hard argmax. No fit, no new capability evidence."}
