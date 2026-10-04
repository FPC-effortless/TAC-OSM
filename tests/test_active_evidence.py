from tac_osm.active_evidence import (
    AlwaysSufficient,
    EvidencePacket,
    NoProbePolicy,
    NeverSufficient,
    ProbeAction,
)


def test_probe_action_validates_budget():
    action = ProbeAction("scalar_output", (0,), expected_schema="bit", budget_units=1)
    assert action.kind == "scalar_output"
    assert action.budget_units == 1


def test_probe_action_rejects_invalid():
    for kwargs in (
        {"kind": "", "budget_units": 1},
        {"kind": "x", "budget_units": -1},
    ):
        try:
            ProbeAction(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid probe action must fail closed")


def test_evidence_packet_validates_cost_and_confidence():
    action = ProbeAction("scalar_output", (0,))
    packet = EvidencePacket(action, "bit", 1, 1.0, confidence=0.75)
    assert packet.payload == 1
    assert packet.confidence == 0.75


def test_evidence_packet_rejects_invalid_fields():
    action = ProbeAction("scalar_output", (0,))
    for kwargs in (
        {"cost_units": 0, "confidence": 1},
        {"cost_units": 1, "confidence": 2},
    ):
        try:
            EvidencePacket(action=action, schema="bit", payload=0, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid evidence must fail closed")


def test_controls_define_backward_compatible_semantics():
    policy = NoProbePolicy()
    assert policy.choose(None, None, (), ()) is None
    assert AlwaysSufficient().sufficient(None, None, ()) is True
    assert NeverSufficient().sufficient(None, None, ()) is False
