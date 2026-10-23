"""Six no-fit research adapter contracts; ALL fixtures are software tests, not model skill."""
from datetime import datetime, timedelta, timezone
import pytest

from tac_osm.variants import VARIANTS, Variant, describe_variant
from tac_osm.variants.contracts import ObservationStamp, ForecastEvidence
from tac_osm.variants.system1 import ReactiveAdvisory
from tac_osm.variants.physical import PhysicalAdvisory
from tac_osm.variants.enterprise import EnterpriseAdvisory, EnterpriseEvent, HORIZONS
from tac_osm.variants.temporal import TemporalReconstructor, TemporalRevision
from tac_osm.variants.edge_lite import EdgeExportCandidate
from tac_osm.variants.counterfactual import ScenarioAdvisor, ScenarioEstimate

T = datetime(2026, 10, 20, 12, tzinfo=timezone.utc)
HASH = "f" * 64


def stamp(entity="case-1", valid=T, recorded=T):
    return ObservationStamp(entity, valid, recorded, "authentic-source-reference")


def evidence():
    return ForecastEvidence(HASH, "registered original source", "DEV_CAPABILITY_OBSERVED")


def test_six_lanes_are_explicit_and_unqualified_not_fake_live_models():
    assert set(VARIANTS) == set(Variant)
    assert len(VARIANTS) == 6
    for lane in VARIANTS:
        spec = describe_variant(lane.value)
        assert spec.deployment_state == "not_qualified"
        assert spec.archived_studies and spec.target and spec.current_evidence
    with pytest.raises(ValueError):
        describe_variant("agi")


def test_reactive_is_risk_only_never_permission_or_sla():
    request = stamp(recorded=T + timedelta(minutes=1))
    adapter = ReactiveAdvisory(lambda values: values[0] / 10, feature_width=2)
    with pytest.raises(ValueError):
        adapter.forecast(request, [2.0, 1.0], known_as_of=T)
    r = adapter.forecast(request, [2.0, 1.0], known_as_of=T + timedelta(minutes=2))
    assert r.proposed_risk == 0.2
    assert r.forecast.scope == "advisory_only"
    assert r.forecast.payload["verified_policy_permission"] is False
    assert r.measured_call_ns >= 0
    with pytest.raises(ValueError):
        adapter.forecast(request, [float("nan"), 1], known_as_of=T + timedelta(minutes=2))


def test_physical_defaults_persistence_and_cannot_fake_neural_calibration():
    a = PhysicalAdvisory(dimension=2, residual_step=lambda x, u: (100, 100))
    assert a.forecast((1, 2), [(3, 4), (3, 4)]).states == ((1., 2.), (1., 2.))
    assert a.forecast((1, 2), [(3, 4)]).method == "persistence"
    with pytest.raises(ValueError):
        PhysicalAdvisory(dimension=2, residual_real_transfer_qualified=True)
    b = PhysicalAdvisory(dimension=2, baseline="servo",
                          servo_step=lambda state, action: tuple(x + y for x, y in zip(state, action)))
    assert b.forecast((1, 2), [(3, 4), (3, 4)]).states == ((4., 6.), (7., 10.))
    with pytest.raises(ValueError):
        b.forecast((1, 2), [(3, 4, 5)])


def test_enterprise_enforces_observed_asof_and_labels_eor_as_proxy():
    seen = []
    def fake_frozen_predictor(rows):
        seen.extend(rows)
        return {h: {"illustrative_only": True} for h in HORIZONS}
    model = EnterpriseAdvisory(fake_frozen_predictor, evidence())
    observed = [EnterpriseEvent(stamp(), "START"), EnterpriseEvent(
        stamp(valid=T + timedelta(seconds=1), recorded=T + timedelta(seconds=5)), "CHECK")]
    with pytest.raises(ValueError):
        model.forecast(observed, known_as_of=T + timedelta(seconds=3), valid_at=T + timedelta(seconds=3))
    prediction = model.forecast(observed, known_as_of=T + timedelta(seconds=5),
                                valid_at=T + timedelta(seconds=5))
    assert len(seen) == 2
    assert prediction.payload["end_of_record_is_verified_business_completion"] is False
    assert prediction.scope == "advisory_only"
    with pytest.raises(ValueError):
        model.forecast(observed + [EnterpriseEvent(stamp("other"), "TEST")],
                       known_as_of=T + timedelta(seconds=5), valid_at=T + timedelta(seconds=5))


def test_backdated_revision_cannot_change_earlier_record_time_truth():
    history = TemporalReconstructor()
    first = TemporalRevision(stamp(), "contract-status", "r1", "pending")
    backdate = TemporalRevision(stamp(valid=T - timedelta(days=3),
                                     recorded=T + timedelta(days=1)), "contract-status",
                                "r2", "corrected")
    history.append(first)
    history.append(backdate)
    yesterday_view = history.as_of("case-1", valid_at=T, known_at=T + timedelta(hours=2))
    assert [r.payload for r in yesterday_view.revisions] == ["pending"]
    today_view = history.as_of("case-1", valid_at=T, known_at=T + timedelta(days=2))
    assert [r.payload for r in today_view.revisions] == ["corrected"]
    assert history.replay("case-1", valid_at=T, known_at=T + timedelta(days=2),
                          initial=(), reduce=lambda state, row: state + (row.payload,)) == ("corrected",)
    with pytest.raises(ValueError):
        history.append(first)


def test_temporal_entity_isolation_future_ingestion_and_timezone_required():
    store = TemporalReconstructor()
    store.append(TemporalRevision(stamp("a"), "risk", "1", "a-only"))
    store.append(TemporalRevision(stamp("b"), "risk", "1", "b-only"))
    assert len(store.as_of("a", valid_at=T, known_at=T).revisions) == 1
    assert store.as_of("a", valid_at=T - timedelta(days=1), known_at=T).revisions == ()
    with pytest.raises(ValueError):
        store.as_of("a", valid_at=T.replace(tzinfo=None), known_at=T)


def test_edge_report_does_not_claim_unmeasured_device_latency():
    candidate = EdgeExportCandidate(evidence(), 99781, 99781, "INT8", 100000)
    assert candidate.qualified_under_50ms is False
    with pytest.raises(ValueError):
        EdgeExportCandidate(evidence(), 15000, 15000, "INT8", 15000)
    with pytest.raises(ValueError):
        EdgeExportCandidate(evidence(), 99781, 99781, "INT8", 100000,
                            measured_device_latency_ms=2.0)


def test_scenario_branch_is_never_causal_or_executable():
    engine = ScenarioAdvisor(lambda state, action: {"risk_proxy": len(action) / 10})
    results = engine.assess({"current_state": "unknown"}, ["route-a", "route-b"])
    assert len(results) == 2
    assert all(not r.causal_effect_established and not r.may_execute for r in results)
    assert all(r.evidence_kind == "observational_conditioning" for r in results)
    with pytest.raises(ValueError):
        ScenarioEstimate("route-a", {}, "observational_conditioning", may_execute=True)
    with pytest.raises(ValueError):
        ScenarioEstimate("route-a", {}, "observational_conditioning", causal_effect_established=True)
    with pytest.raises(ValueError):
        engine.assess({}, ["duplicate", "duplicate"])
