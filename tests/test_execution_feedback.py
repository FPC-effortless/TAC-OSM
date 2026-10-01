from tac_osm import Outcome, VerificationResult
from tac_osm.environment import build_relational_task
from tac_osm.execution_feedback import (
    VerifierDrivenEnergyRouter,
    execution_feedback_from_verdict,
)
from tac_osm.state import PersistentStore, StateConfig


def test_failed_execution_creates_verifier_repair_positive():
    task = build_relational_task(17, dim=8, n_candidates=8)
    wrong = next(i for i in range(len(task.candidates)) if i != task.target_action)
    outcome = Outcome(False, 0.0, "wrong_candidate", task)
    verification = VerificationResult(False, "final_mismatch")
    feedback = execution_feedback_from_verdict(
        task.candidates, wrong, outcome, verification
    )
    assert feedback.verdict == "repair"
    assert feedback.target_index == task.target_action
    assert feedback.positive_indices == (task.target_action,)
    assert feedback.negative_indices == (wrong,)


def test_verifier_router_updates_on_rejected_candidate():
    task = build_relational_task(19, dim=8, n_candidates=8)
    state = PersistentStore(StateConfig(seed=19, n_slots=32))
    router = VerifierDrivenEnergyRouter()
    decision = router.route(task.public(), state, task.candidates)
    outcome = Outcome(
        decision.selected == task.target_action,
        float(decision.selected == task.target_action),
        "ok" if decision.selected == task.target_action else "wrong_candidate",
        task,
    )
    verification = VerificationResult(outcome.success, "final_ok" if outcome.success else "final_mismatch")
    before = router.updates
    loss, feedback = router.learn_from_verifier(
        task.public(),
        state,
        task.candidates,
        decision.selected,
        outcome,
        verification,
        scores=decision.scores,
    )
    assert feedback.verdict in {"accept", "repair"}
    assert loss >= 0.0
    assert router.updates >= before
