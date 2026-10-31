"""No-fit agnosticism falsifiers: native, nonnative recurrent and stateless.

All data are constructed fixtures. These tests do not substantiate model
learning, safety, correct science, or generalization of trained checkpoints.
"""
from dataclasses import replace
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
import torch
from torch import nn

from tac_osm.predictive_state_agent import PredictiveStateAgent
from tac_osm.research_probe import (
    ResearchProbeAdapter, inspect_variant, probe_memory,
    probe_dependencies, probe_entity_isolation, probe_runtime,
)
from tac_osm.research_probe_adapters import native_predictive_adapter
from tac_osm.research_anticheat import (
    AuditLedger, AuditGateError, probe_prediction_invariance, Finding,
    trace_torch_noninterference,
)


class GatedEpisode(nn.Module):
    """Independent tiny recurrent cell: ONE arbitrary state and two heads."""
    def __init__(self):
        super().__init__()
        self.cell = nn.GRUCell(4, 6)
        self.scores = nn.Linear(6 + 3, 1)
        self.reconstruct = nn.Linear(6, 4)


def episode():
    torch.manual_seed(205)
    return {"signal": torch.randn(3, 4, 4),
            "options": torch.randn(3, 2, 3)}


def gated_adapter():
    def validate(model, sample):
        if not isinstance(model, GatedEpisode) or set(sample) != {"signal", "options"}:
            raise ValueError("contract")
        h, a = sample["signal"], sample["options"]
        if h.ndim != 3 or a.ndim != 3 or h.shape[0] != a.shape[0] or (
                h.shape[-1] != 4 or a.shape[-1] != 3 or a.shape[1] < 1):
            raise ValueError("shapes")
        if not all(torch.isfinite(t).all() for t in (h, a)):
            raise ValueError("nonfinite")
    def forecast(model, state, sample):
        options = sample["options"]
        b, n, _ = options.shape
        latent = state["cell_state"]
        x = torch.cat([latent[:, None].expand(b, n, -1), options], dim=-1)
        return {"scores": model.scores(x),
                "reconstruction": model.reconstruct(latent)}
    def grads(model, state, sample):
        s = state["cell_state"].detach().clone().requires_grad_(True)
        opts = sample["options"].detach().clone().requires_grad_(True)
        output = forecast(model, {"cell_state": s}, {**sample, "options": opts})
        return output, {"latent": s, "choices": opts}
    def permute(sample):
        reverse = list(reversed(range(sample["options"].shape[1])))
        other = dict(sample)
        other["options"] = sample["options"][:, reverse]
        def restore(outputs):
            return {"scores": outputs["scores"][:, reverse],
                    "reconstruction": outputs["reconstruction"]}
        return other, restore
    return ResearchProbeAdapter(
        name="diagnostic-gated-episode-v7", validate=validate,
        steps=lambda sample: sample["signal"].shape[1],
        start=lambda model, sample: {
            "cell_state": sample["signal"].new_zeros(sample["signal"].shape[0], 6)},
        advance=lambda model, state, sample, t: {
            "cell_state": model.cell(sample["signal"][:, t], state["cell_state"])},
        forecast=forecast, state_views=lambda s: s,
        patch_state=lambda state, name, value: {**state, name: value},
        batch_size=lambda sample: sample["signal"].shape[0],
        batch_item=lambda sample, i: {k: v[i:i + 1] for k, v in sample.items()},
        gradient_graph=grads, permute_candidates=permute,
    )


class StatelessNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Linear(4, 9), nn.GELU(), nn.Linear(9, 2))
    def forward(self, x):
        return self.features(x)


def stateless_adapter():
    def validate(m, sample):
        if not isinstance(m, StatelessNet) or set(sample) != {"features"}:
            raise ValueError("wrong architecture or features")
        x = sample["features"]
        if x.ndim != 2 or x.shape[-1] != 4 or not bool(torch.isfinite(x).all()):
            raise ValueError("invalid source tensor")
    def grads(m, state, sample):
        inputs = sample["features"].detach().clone().requires_grad_(True)
        unused = inputs.detach().clone().requires_grad_(True)
        return {"risk": m(inputs)}, {"features": inputs, "unused_probe": unused}
    return ResearchProbeAdapter(
        name="stateless-feedforward-v21", validate=validate,
        steps=lambda sample: 0, start=lambda model, sample: {},
        advance=lambda model, state, sample, t: state,
        forecast=lambda model, state, sample: {"risk": model(sample["features"])},
        state_views=lambda state: state,
        patch_state=lambda state, name, value: {},
        batch_size=lambda sample: sample["features"].shape[0],
        batch_item=lambda sample, i: {"features": sample["features"][i:i + 1]},
        gradient_graph=grads,
    )


def test_nonnative_gated_recurrent_architecture_is_fully_measured():
    model, sample = GatedEpisode(), episode()
    report = inspect_variant(model, sample, gated_adapter(), isolation=True)
    assert report["adapter"] == "diagnostic-gated-episode-v7"
    assert report["memory"]["state_components"] == ["cell_state"]
    assert report["memory"]["events"] == 4
    assert report["memory"]["query_does_not_write_state"]
    assert report["memory"]["candidate_permutation"]["status"] == "PASS"
    assert report["dependencies"]["heads"]["scores"]["choices"]["connected"]
    assert report["dependencies"]["heads"]["reconstruction"]["choices"]["connected"] is False
    assert report["entity_isolation"]["status"] == "PASS"
    assert report["integrity_status"] == "INCONCLUSIVE", "no local runtime measurement must remain UNKNOWN"


def test_native_model_runs_through_exact_same_generic_protocol():
    torch.manual_seed(25)
    model = PredictiveStateAgent(5, 3, 1)
    sample = {"history": torch.randn(2, 3, 5),
              "candidates": torch.randn(2, 4, 3)}
    result = inspect_variant(model, sample, native_predictive_adapter, isolation=True)
    assert set(result["memory"]["state_components"]) == {"fast", "slow", "memory"}
    assert result["memory"]["candidate_permutation"]["status"] == "PASS"
    assert result["entity_isolation"]["status"] == "PASS"
    assert not result["dependencies"]["heads"]["next_observation"]["associative_memory"]["connected"]


def test_stateless_architecture_has_explicit_empty_state_without_false_recall_claim():
    model, adapter = StatelessNet(), stateless_adapter()
    sample = {"features": torch.randn(2, 4)}
    result = inspect_variant(model, sample, adapter, isolation=True, profile=True)
    assert result["memory"]["events"] == 0
    assert result["memory"]["state_components"] == []
    assert result["memory"]["coverage"] == "stateless_model"
    assert result["dependencies"]["heads"]["risk"]["features"]["connected"]
    assert not result["dependencies"]["heads"]["risk"]["unused_probe"]["connected"]
    assert result["memory"]["candidate_permutation"]["status"] == "INCONCLUSIVE_NOT_DECLARED"
    assert result["runtime"]["state_bytes_batch"] == 0


def test_adapter_without_gradients_or_batch_split_is_inconclusive_not_pass():
    base = stateless_adapter()
    missing = replace(base, gradient_graph=None, batch_item=None)
    report = inspect_variant(StatelessNet(), {"features": torch.randn(2, 4)}, missing)
    assert report["dependencies"]["status"] == "INCONCLUSIVE_UNINSTRUMENTED"
    assert report["entity_isolation"]["status"] == "INCONCLUSIVE_UNINSTRUMENTED"
    assert report["integrity_status"] == "INCONCLUSIVE"


def test_incorrect_adapter_rejects_model_swap_not_silently_rebrand():
    with pytest.raises(ValueError, match="wrong architecture"):
        probe_memory(GatedEpisode(), {"features": torch.randn(2, 4)}, stateless_adapter())


def test_no_training_or_mutation_in_model_inspection():
    model = GatedEpisode()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    sample = episode()
    original = {k: v.clone() for k, v in sample.items()}
    inspect_variant(model, sample, gated_adapter(), isolation=True, profile=True)
    for key, tensor in model.state_dict().items():
        assert torch.equal(tensor, before[key])
    for key in sample:
        assert torch.equal(sample[key], original[key])


def test_batch_dependent_nonnative_architecture_is_flagged():
    class CrossEntity(GatedEpisode):
        pass
    model = CrossEntity()
    adapter = gated_adapter()
    bad = replace(adapter, advance=lambda m, s, x, t: {
        "cell_state": m.cell(x["signal"][:, t], s["cell_state"])
                      + x["signal"][:, t].mean()})
    result = probe_entity_isolation(model, episode(), bad, tolerance=1e-7)
    assert result["status"] == "FAIL"
    assert result["max_absolute_difference"] > 1e-7


def test_unauthorized_buffer_update_in_forecast_fails_purity():
    class Stateful(StatelessNet):
        def __init__(self):
            super().__init__()
            self.register_buffer("hidden_count", torch.zeros(1))
    model = Stateful()
    base = stateless_adapter()
    bad = replace(base,
        validate=lambda m, s: None,
        forecast=lambda m, s, x: change_cache(m, x),
    )
    def _unused(): pass
    report = probe_memory(model, {"features": torch.randn(2, 4)}, bad)
    assert report["query_does_not_write_state"] is False

def change_cache(model, sample):
    model.hidden_count.add_(1)
    return {"risk": model(sample["features"])}


def test_missing_candidate_semantics_not_automatically_marked_pass():
    model = GatedEpisode()
    report = probe_memory(model, episode(), replace(gated_adapter(),
                                                        permute_candidates=None))
    assert report["candidate_permutation"]["status"] == "INCONCLUSIVE_NOT_DECLARED"


def test_adversarial_model_leakage_probe_is_version_independent():
    class SideChannel(nn.Module):
        def forward(self, inputs):
            return inputs["sensor"] * 2 + inputs["oracle"]
    model = SideChannel()
    clean = {"sensor": torch.ones(1, 2), "oracle": torch.zeros(1, 2)}
    poison = {**clean, "oracle": torch.ones(1, 2)}
    finding = trace_torch_noninterference(
        model, lambda network, record: network(record), clean, poison)
    assert finding.status == "FAIL"
    assert finding.evidence["first_changed_module"] == "<untraced_functional_or_external_path>"


def test_required_adapter_diagnostics_cannot_silently_be_missed():
    ledger = AuditLedger("real-run-required", ["source_boundary", "model_noninterference"])
    ledger.add(Finding("source_boundary", "PASS", "allowlist", {}))
    assert ledger.report()["status"] == "INCONCLUSIVE"
    with pytest.raises(AuditGateError):
        ledger.assert_clean()


def test_batch_isolation_no_support_returns_inconclusive_not_verdict():
    model, sample = GatedEpisode(), episode()
    result = probe_entity_isolation(model, sample, replace(gated_adapter(), batch_item=None))
    assert result["status"] == "INCONCLUSIVE_UNINSTRUMENTED"


def test_generic_local_runtime_accepts_unusual_one_cell_memory_state():
    result = probe_runtime(GatedEpisode(), episode(), gated_adapter(), warmups=0, repeats=2)
    assert result["state_bytes_batch"] == 3 * 6 * 4
    assert result["end_to_end_ms_median"] >= 0


def test_probes_reject_future_fields_even_if_other_model_valid():
    model = GatedEpisode()
    bad = {**episode(), "target": torch.zeros(3)}
    with pytest.raises(ValueError, match="contract"):
        inspect_variant(model, bad, gated_adapter())


def test_dependency_missing_graph_is_explicit_not_misattributed():
    result = probe_dependencies(GatedEpisode(), episode(),
                                replace(gated_adapter(), gradient_graph=None))
    assert result["status"] == "INCONCLUSIVE_UNINSTRUMENTED"



def test_historical_observatory_api_accepts_non_native_adapter_without_retyping_model():
    from tac_osm.research_observatory import (
        memory_tomography, input_dependency_map, entity_isolation_probe,
        profile_native_runtime,
    )
    m, sample, adapter = GatedEpisode(), episode(), gated_adapter()
    assert memory_tomography(m, sample, adapter=adapter)["state_components"] == ["cell_state"]
    assert input_dependency_map(m, sample, adapter=adapter)["status"] == "MEASURED"
    assert entity_isolation_probe(m, sample, adapter=adapter)["status"] == "PASS"
    assert profile_native_runtime(m, sample, adapter=adapter,
                                  warmups=0, repeats=2)["state_bytes_batch"] == 72



def test_interpretation_understands_arbitrary_memory_and_head_names():
    from tac_osm.research_interpretation import interpret_snapshot
    report = {"memory": {
        "query_does_not_write_state": True,
        "candidate_permutation": {"status": "PASS"},
        "interventions": {
            "entity_slot_k": {"head_mean_absolute_change": {
                "policy_scores": 0., "physical_residual": 0.}},
            "<reset_all>": {"head_mean_absolute_change": {
                "policy_scores": 0., "physical_residual": 0.}},
        }}}
    result = interpret_snapshot(report)
    assert {x["area"] for x in result["alerts"]} == {"memory"}
    assert "named-state" in result["alerts"][0]["hypothesis"]


def test_interpretation_does_not_invent_memory_for_stateless_model():
    report = inspect_variant(
        StatelessNet(), {"features": torch.randn(2, 4)},
        stateless_adapter(), isolation=True)
    assert "interpretation" in report
    assert not any(x["area"] == "memory" for x in report["interpretation"]["alerts"])


class DiscretePrefixNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(256, 4)
        self.output = nn.Linear(4, 2)


def discrete_prefix_adapter():
    def validate(m, x):
        if not isinstance(m, DiscretePrefixNet) or set(x) != {"history", "query"}:
            raise ValueError("discrete prefix contract")
    def start(m, x):
        return {"prefix": x["history"][:, :0].clone(),
                "seen": torch.zeros(x["history"].shape[0], 1, dtype=torch.bool)}
    def advance(m, s, x, t):
        return {"prefix": torch.cat((s["prefix"], x["history"][:, t:t+1]), 1),
                "seen": torch.ones_like(s["seen"])}
    def forecast(m, s, x):
        context = (m.embedding(s["prefix"]).mean(1) if s["prefix"].shape[1]
                   else m.embedding.weight.new_zeros(x["history"].shape[0], 4))
        return {"logits": m.output(context + m.embedding(x["query"][:, 0]))}
    return ResearchProbeAdapter(name="integer-prefix-with-boolean-flag", validate=validate,
        steps=lambda x: x["history"].shape[1], start=start, advance=advance,
        forecast=forecast, state_views=lambda s:s,
        patch_state=lambda s,name,value:{**s,name:value},
        batch_size=lambda x:x["history"].shape[0],
        batch_item=lambda x,i:{k:v[i:i+1] for k,v in x.items()})


def test_discrete_prefix_growth_and_boolean_state_are_counted_without_float_memory():
    model = DiscretePrefixNet()
    sample = {"history":torch.tensor([[1,2,3],[4,5,6]]),"query":torch.tensor([[7],[8]])}
    adapter = discrete_prefix_adapter()
    report = probe_memory(model, sample, adapter)
    assert report["coverage"] == "named_adapter_state"
    assert report["state_dtypes"] == {"prefix":"torch.int64","seen":"torch.bool"}
    assert report["state_storage_bytes"] == 2*3*8+2
    assert report["query_does_not_write_state"]
    assert all(row["components"]["prefix"]["shape_changed"] for row in report["timeline"])
    assert all(row["components"]["prefix"]["change_l2"] is None for row in report["timeline"])
    runtime = probe_runtime(model,sample,adapter,warmups=0,repeats=2)
    assert runtime["state_bytes_batch"] == 50


def test_large_integer_identity_mismatch_is_exact_even_with_large_float_tolerance():
    model = StatelessNet()
    sample = {"features":torch.randn(2,4)}
    adapter = replace(stateless_adapter(), start=lambda m,x:{
        "source_id":torch.full((x["features"].shape[0],1), 2**62 +
                         (1 if x["features"].shape[0]==1 else 0),dtype=torch.int64)})
    result = probe_entity_isolation(model,sample,adapter,tolerance=1e8)
    assert result["status"] == "FAIL"
    assert result["discrete_state_exactly_equal"] is False


def test_boolean_state_identity_mismatch_does_not_subtract_boolean_tensors():
    model = StatelessNet(); sample = {"features":torch.randn(2,4)}
    adapter = replace(stateless_adapter(),start=lambda m,x:{
        "flag":torch.full((x["features"].shape[0],1),x["features"].shape[0]==1,dtype=torch.bool)})
    result = probe_entity_isolation(model,sample,adapter)
    assert result["status"] == "FAIL"
    assert result["discrete_state_exactly_equal"] is False


def test_discrete_state_support_does_not_accept_integer_prediction_heads():
    bad = replace(stateless_adapter(),forecast=lambda m,s,x:{"risk":m(x["features"]).long()})
    with pytest.raises(ValueError,match="nonfloating"):
        probe_memory(StatelessNet(),{"features":torch.randn(2,4)},bad)
