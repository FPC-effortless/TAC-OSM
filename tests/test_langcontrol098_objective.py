"""Independent, no-fit falsifiers for the fixed-coefficient objective.

These constructed examples test the loss and its information boundary. They
are not measurements of trained acquisition, transfer, or intelligence.
"""
import importlib.util
import math
from pathlib import Path
import sys

import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "research/mathematical-engineering/langcontrol098"


def load(name):
    spec = importlib.util.spec_from_file_location(
        "langcontrol098_independent_objective_test_" + name, HERE / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


objective = load("objective")
models = load("models")


class FixedLogits(nn.Module):
    """Expose logits as independent leaves, so each coefficient is observable."""

    def __init__(self, logits):
        super().__init__()
        self.values = nn.Parameter(logits.clone())
        self.calls = []

    def forward(self, inputs, **kwargs):
        self.calls.append((inputs.detach().clone(), set(kwargs)))
        assert self.values.shape[:2] == inputs.shape
        return self.values, None


def fixture():
    # Unequal lengths both within and between domains expose token-weighting
    # and global-denominator regressions that equal-length examples hide.
    inputs = torch.arange(28, dtype=torch.long).reshape(4, 7)
    targets = (inputs * 3 + 11) % 256
    valid = torch.zeros_like(inputs, dtype=torch.bool)
    query = torch.zeros_like(valid)
    for row, (length, future_count) in enumerate(((7, 2), (5, 1), (6, 3), (4, 1))):
        valid[row, :length] = True
        query[row, length - future_count:length] = True
    logits = torch.sin(torch.arange(4 * 7 * 256, dtype=torch.float64)).reshape(4, 7, 256)
    domains = ("lexical", "lexical", "arc", "arc")
    return inputs, targets, query, valid, logits, domains


def scalar_reference(logits, targets, query, valid, domains, prefix_lambda):
    """Python scalar reference, independent of torch's CE/reduction code."""
    raw = logits.detach().tolist()
    answer = 0.0
    for domain in ("lexical", "arc"):
        indices = [i for i, owner in enumerate(domains) if owner == domain]
        episode_values = []
        for i in indices:
            prefix, future = [], []
            for t in range(len(raw[i])):
                if not valid[i, t]:
                    continue
                values = raw[i][t]
                largest = max(values)
                ce = largest + math.log(sum(math.exp(x - largest) for x in values))
                ce -= values[int(targets[i, t])]
                (future if query[i, t] else prefix).append(ce)
            episode_values.append(sum(future) / len(future)
                                  + prefix_lambda * sum(prefix) / len(prefix))
        answer += 0.5 * sum(episode_values) / len(episode_values)
    return answer


def expected_gradient(logits, targets, query, valid, domains, prefix_lambda):
    expected = torch.zeros_like(logits)
    for i, domain in enumerate(domains):
        domain_count = domains.count(domain)
        for t in range(logits.shape[1]):
            if not valid[i, t]:
                continue
            selected = query[i] if query[i, t] else (valid[i] & ~query[i])
            multiplier = 1.0 if query[i, t] else prefix_lambda
            coefficient = 0.5 * multiplier / (domain_count * int(selected.sum()))
            values = logits[i, t].detach()
            # The softmax gradient formula is independent of loss reductions.
            probabilities = (values - values.max()).exp()
            probabilities /= probabilities.sum()
            expected[i, t] = coefficient * probabilities
            expected[i, t, int(targets[i, t])] -= coefficient
    return expected


@pytest.mark.parametrize("arm,prefix_lambda", [("query", 0), ("full", 1)])
def test_scalar_loss_and_every_logit_gradient_have_registered_coefficients(arm, prefix_lambda):
    inputs, targets, query, valid, logits, domains = fixture()
    model = FixedLogits(logits)
    result = objective.sequence_loss(model, inputs, targets, query, valid_mask=valid,
                                     domains=domains, objective=arm)
    assert float(result.loss.detach()) == pytest.approx(
        scalar_reference(logits, targets, query, valid, domains, prefix_lambda),
        abs=2e-12, rel=0,
    )
    result.loss.backward()
    torch.testing.assert_close(model.values.grad,
                               expected_gradient(logits, targets, query, valid, domains, prefix_lambda),
                               atol=2e-15, rtol=2e-13)
    assert torch.isfinite(model.values.grad).all()
    assert torch.equal(model.values.grad[~valid], torch.zeros_like(model.values.grad[~valid]))


def test_answer_coefficients_remain_identical_when_prefix_supervision_is_added():
    inputs, targets, query, valid, logits, domains = fixture()
    gradients = []
    for arm in ("query", "full"):
        model = FixedLogits(logits)
        result = objective.sequence_loss(model, inputs, targets, query, valid_mask=valid,
                                         domains=domains, objective=arm)
        result.loss.backward()
        gradients.append(model.values.grad)
    torch.testing.assert_close(gradients[0][query], gradients[1][query], atol=0, rtol=0)
    assert gradients[0][valid & ~query].count_nonzero() == 0
    assert gradients[1][valid & ~query].count_nonzero() > 0


@pytest.mark.parametrize("arm,prefix_lambda", [("query", 0), ("full", 1)])
def test_domain_weight_is_not_replaced_by_its_batch_fraction(arm, prefix_lambda):
    inputs, targets, query, valid, logits, domains = fixture()
    # Two lexical episodes and one ARC episode must still have equal total
    # domain coefficients; silently averaging all three episodes is wrong.
    inputs, targets, query, valid, logits = (
        tensor[:3] for tensor in (inputs, targets, query, valid, logits)
    )
    domains = domains[:3]
    model = FixedLogits(logits)
    result = objective.sequence_loss(model, inputs, targets, query, valid_mask=valid,
                                     domains=domains, objective=arm)
    assert float(result.loss.detach()) == pytest.approx(
        scalar_reference(logits, targets, query, valid, domains, prefix_lambda),
        abs=2e-12, rel=0,
    )
    result.loss.backward()
    torch.testing.assert_close(model.values.grad,
                               expected_gradient(logits, targets, query, valid, domains, prefix_lambda),
                               atol=2e-15, rtol=2e-13)


@pytest.mark.parametrize("arm", ["query", "full"])
def test_logit_derivatives_also_match_finite_differences(arm):
    inputs, targets, query, valid, logits, domains = fixture()
    model = FixedLogits(logits)
    result = objective.sequence_loss(model, inputs, targets, query, valid_mask=valid,
                                     domains=domains, objective=arm)
    result.loss.backward()
    analytic = model.values.grad.detach().clone()
    # Answer, prefix, and padding in each domain, including target/non-target
    # classes, exercise the coefficient identity without reimplementing it.
    positions = ((0, 6), (0, 1), (1, 6), (2, 5), (2, 0), (3, 6))
    epsilon = 1e-5
    for row, time in positions:
        for token in (int(targets[row, time]), (int(targets[row, time]) + 1) % 256):
            base = float(model.values[row, time, token].detach())
            evaluations = []
            for direction in (-1, 1):
                with torch.no_grad():
                    model.values[row, time, token] = base + direction * epsilon
                measured = objective.sequence_loss(model, inputs, targets, query, valid_mask=valid,
                                                    domains=domains, objective=arm)
                evaluations.append(float(measured.loss.detach()))
            with torch.no_grad():
                model.values[row, time, token] = base
            numerical = (evaluations[1] - evaluations[0]) / (2 * epsilon)
            assert numerical == pytest.approx(float(analytic[row, time, token]), abs=2e-9, rel=0)


def test_prefix_target_poison_changes_only_the_prefix_supervised_arm():
    inputs, targets, query, valid, logits, domains = fixture()
    poison = targets.clone()
    poison[valid & ~query] = (poison[valid & ~query] + 117) % 256
    for arm in ("query", "full"):
        outputs, gradients = [], []
        for labels in (targets, poison):
            model = FixedLogits(logits)
            result = objective.sequence_loss(model, inputs, labels, query, valid_mask=valid,
                                             domains=domains, objective=arm)
            result.loss.backward()
            outputs.append(result.loss.detach())
            gradients.append(model.values.grad.detach())
        if arm == "query":
            torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
            torch.testing.assert_close(gradients[0], gradients[1], atol=0, rtol=0)
        else:
            assert not torch.equal(outputs[0], outputs[1])
            assert not torch.equal(gradients[0], gradients[1])


def test_padding_target_poison_never_changes_loss_or_gradient():
    inputs, targets, query, valid, logits, domains = fixture()
    poison = targets.clone()
    poison[~valid] = (poison[~valid] + 73) % 256
    for arm in ("query", "full"):
        outputs, gradients = [], []
        for labels in (targets, poison):
            model = FixedLogits(logits)
            result = objective.sequence_loss(model, inputs, labels, query, valid_mask=valid,
                                             domains=domains, objective=arm)
            result.loss.backward()
            outputs.append(result.loss.detach())
            gradients.append(model.values.grad.detach())
        torch.testing.assert_close(outputs[0], outputs[1], atol=0, rtol=0)
        torch.testing.assert_close(gradients[0], gradients[1], atol=0, rtol=0)


def test_loss_metadata_and_targets_never_become_model_inputs():
    inputs, targets, query, valid, logits, domains = fixture()
    original_inputs = inputs.clone()
    changed_query = query.clone()
    changed_query[0, 4] = True
    changed_targets = (targets + 97) % 256
    model = FixedLogits(logits)
    first = objective.sequence_loss(model, inputs, targets, query, valid_mask=valid,
                                    domains=domains, objective="query")
    second = objective.sequence_loss(model, inputs, changed_targets, changed_query, valid_mask=valid,
                                     domains=tuple(reversed(domains)), objective="full")
    torch.testing.assert_close(first.logits, second.logits, atol=0, rtol=0)
    assert len(model.calls) == 2
    for actual_inputs, keyword_names in model.calls:
        assert torch.equal(actual_inputs, original_inputs)
        assert keyword_names <= {"owners", "state"}
    assert torch.equal(inputs, original_inputs)


@pytest.mark.parametrize("domains", [None, (), ("lexical",) * 4,
                                     ("lexical", "arc", "arc"),
                                     ("lexical", "lexical", "arc", "unknown"),
                                     "lexical"])
def test_missing_unbalanced_or_unknown_domain_contract_fails_closed(domains):
    inputs, targets, query, valid, logits, _ = fixture()
    kwargs = {} if domains is None else {"domains": domains}
    with pytest.raises(ValueError):
        objective.sequence_loss(FixedLogits(logits), inputs, targets, query,
                                valid_mask=valid, objective="query", **kwargs)


@pytest.mark.parametrize("mutation", ["empty_query", "empty_prefix", "outside_valid",
                                      "noncontiguous_valid", "noncontiguous_query", "float_mask"])
def test_invalid_or_empty_supervised_segments_fail_closed(mutation):
    inputs, targets, query, valid, logits, domains = fixture()
    if mutation == "empty_query":
        query[0] = False
    elif mutation == "empty_prefix":
        query[0] = valid[0]
    elif mutation == "outside_valid":
        query[1, -1] = True
    elif mutation == "noncontiguous_valid":
        valid[0, 1] = False
    elif mutation == "noncontiguous_query":
        query[0, 3] = True
    elif mutation == "float_mask":
        query = query.float()
    with pytest.raises(ValueError):
        objective.sequence_loss(FixedLogits(logits), inputs, targets, query,
                                valid_mask=valid, domains=domains)


def test_single_domain_diagnostic_keeps_registered_half_weight():
    inputs, targets, query, valid, logits, _ = fixture()
    model = FixedLogits(logits[:1])
    result = objective.sequence_loss(model, inputs[:1], targets[:1], query[:1],
                                     valid_mask=valid[:1], domains=("arc",),
                                     allow_single_domain=True, objective="query")
    result.loss.backward()
    expected = expected_gradient(logits[:1], targets[:1], query[:1], valid[:1], ("arc",), 0)
    torch.testing.assert_close(model.values.grad, expected, atol=2e-15, rtol=2e-13)


@pytest.mark.parametrize("family", ["tac", "gru", "ssm", "transformer"])
def test_fixed_answer_loss_reaches_actual_acquisition_state(family):
    torch.set_num_threads(1)
    model = models.make_model(family, 1701)
    acquisition = bytes(range(8))
    inputs, targets, query = objective.episode_tensors(
        acquisition, bytes(range(40, 46)), bytes(range(80, 84))
    )
    inputs, targets, query = (value.repeat(2, 1) for value in (inputs, targets, query))
    result = objective.sequence_loss(model, inputs, targets, query,
                                     domains=("lexical", "arc"), objective="query",
                                     acquisition_end=len(acquisition), retain_acquisition_grad=True)
    result.loss.backward()
    receipt = objective.acquisition_gradient_receipt(result)
    assert receipt["status"] == "PASS"
    # These bytes occur only before the split; future supervision must reach
    # their learned encoding through the actual model's state writes.
    assert float(model.core.embedding.weight.grad[:8].abs().sum()) > 0
