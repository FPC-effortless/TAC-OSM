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



def test_active_evidence_runs_before_routing_and_is_recorded():
    from tac_osm import (
        Candidate, ExecutionResult, Outcome, Query, RoutingDecision,
        StateRead, StateWrite, Structure, VerificationResult,
    )
    from tac_osm.model import ModelConfig, TacOsmModel
    from tac_osm.active_evidence import ActiveEvidenceConfig, ProbePolicy
    from tac_osm.environment import Task

    events = []

    class Env:
        def next_task(self, state):
            return Task(
                query=Query("0 0\t", context=(1, 0), provenance="test"),
                target_action=0,
                candidates=(
                    Candidate("c0", (0, 0), action=0),
                    Candidate("c1", (1, 1), action=1),
                ),
            )

        def probe(self, state, action):
            events.append(("probe", action.kind))
            return EvidencePacket(
                action=action,
                schema="bit",
                payload=1,
                cost_units=1.0,
                provenance={"source": "test"},
            )

        def transition(self, state, action, query):
            events.append(("transition", action))
            return Outcome(success=True, value=1.0)

        def success(self, query, outcome):
            return bool(outcome.success)

    class Policy:
        def choose(self, query, state, candidates, evidence):
            events.append(("choose", len(evidence)))
            return ProbeAction("scalar_output", (0,), expected_schema="bit", budget_units=1.0)

    class Compiler:
        def compile(self, query, state, evidence):
            events.append(("compile", len(evidence)))
            return Query(
                query.text,
                context=query.context + (int(evidence[-1].payload),),
                step=query.step,
                provenance=query.provenance,
            )

    class Router:
        def route(self, query, state, candidates):
            events.append(("route", tuple(query.context)))
            return RoutingDecision(selected=0, scores=(1.0, 0.0), provenance="test")

    class State:
        def read(self, query):
            return StateRead(keys=(), values=(), slot_used=())

        def write(self, update):
            return StateWrite(committed=True, key=update.key, step=update.step)

    class Exec:
        def execute(self, structure, inputs):
            events.append(("execute", structure.key))
            return ExecutionResult(output=1.0, node_values=(1.0,), provenance="test")

    class Verify:
        def verify(self, computation, outcome):
            events.append(("verify", True))
            return VerificationResult(passed=True)

    model = TacOsmModel(
        state=State(),
        router=Router(),
        executor=Exec(),
        verifier=Verify(),
        repair=None,
        environment=Env(),
        config=ModelConfig(n_steps=1, learn=False),
        active_evidence=ActiveEvidenceConfig(enabled=True, max_probes=1),
        probe_policy=Policy(),
        probe_environment=Env(),
        evidence_sufficiency=NeverSufficient(),
        evidence_compiler=Compiler(),
    )
    step = model.run().steps[0]
    assert step.evidence
    assert events[:4] == [
        ("choose", 0),
        ("probe", "scalar_output"),
        ("compile", 1),
        ("route", (1, 0, 1)),
    ]
    assert events.index(("route", (1, 0, 1))) < events.index(("execute", "c0"))
    assert step.provenance["probe_count"] == 1
    assert step.provenance["probe_cost_units"] == 1.0
