"""Correct-by-construction REAL-ROBOT one-step, observed-action target contract.

This module transforms original physical trajectory records, not synthetic
tasks. Only records from a source-authorized FIT episode may be passed to
the training adapter by the research runner; the cohort gate enforces that.
Not a model trainer or causal-action estimator. Inputs end at real frame t;
targets are actually observed at t+1.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RealLoggedWindow:
    observed_history_state: tuple
    previously_executed_actions: tuple
    current_measured_state: tuple
    current_logged_action: tuple
    currently_human_controlled: bool
    next_measured_state: tuple
    new_intervention_onset: bool | None
    current_frame_index: int
    episode_index: int
    # Never add future actions, next-time control identity or future state to
    # model_inputs(). They are factual outcomes, not knowable action inputs.

    def model_inputs(self):
        return {"past_and_current_measured_states":self.observed_history_state,
                "strictly_prior_executed_actions":self.previously_executed_actions,
                "current_measured_state":self.current_measured_state,
                "current_logged_action":self.current_logged_action,
                "currently_human_controlled":self.currently_human_controlled}

    def factual_targets(self):
        return {"next_measured_state":self.next_measured_state,
                "next_intervention_onset_when_policy_controlled":
                    self.new_intervention_onset}


def real_windows_from_original_episode(table, episode_index, history=256):
    """Extract exact original logged one-step triples, no cross-episode joins.

    Actions at t are the source's executed commands; only later measured
    qpos[t+1] can supervise their *observed factual* consequences. Human
    authority at t is an input; t+1 takeover is known only as a target.
    Operator takeovers do not independently establish physical failure.
    """
    if type(history) is not int or history<2:
        raise ValueError("history must be >=2 authentic prior-or-current frames")
    data={key:table[key].to_pylist() for key in (
        "frame_index","episode_index","timestamp","intervention","control_source",
        "action.qpos","observations.state.qpos")}
    n=table.num_rows
    if n<=history:raise ValueError("insufficient real future observations")
    if data["frame_index"] != list(range(n)) or set(data["episode_index"])!={episode_index}:
        raise ValueError("episode/time continuity or identity invalid")
    if any(not math.isfinite(t) for t in data["timestamp"]) or any(
            b<=a for a,b in zip(data["timestamp"],data["timestamp"][1:])):
        raise ValueError("noncausal or invalid publisher time sequence")
    for i in range(n):
        state,action=data["observations.state.qpos"][i],data["action.qpos"][i]
        if not (len(state)==len(action)==14 and
                all(math.isfinite(v) for v in (*state,*action))):
            raise ValueError("nonfinite physical state/action")
        flag=data["intervention"][i]
        if type(flag) is not bool or data["control_source"][i]!=(
                "intervention" if flag else "policy"):
            raise ValueError("unknown current human/policy authority")
    for t in range(history-1,n-1):
        state_window=tuple(tuple(float(v) for v in q)
                         for q in data["observations.state.qpos"][t-history+1:t+1])
        # The final logged action a[t] is queried separately; the recurrence
        # sees ONLY genuinely earlier executed commands a[t-history+1:t].
        action_window=tuple(tuple(float(v) for v in a)
                            for a in data["action.qpos"][t-history+1:t])
        current_human=data["intervention"][t]
        next_human=data["intervention"][t+1]
        yield RealLoggedWindow(
            observed_history_state=state_window,
            previously_executed_actions=action_window,
            current_measured_state=state_window[-1],
            current_logged_action=tuple(float(v) for v in data["action.qpos"][t]),
            currently_human_controlled=current_human,
            next_measured_state=tuple(float(v) for v in data["observations.state.qpos"][t+1]),
            new_intervention_onset=(None if current_human else bool(next_human)),
            current_frame_index=t,
            episode_index=episode_index)


def assert_no_future_inputs(sample):
    inputs=sample.model_inputs()
    assert len(inputs["past_and_current_measured_states"]) == (
        len(inputs["strictly_prior_executed_actions"])+1)
    assert len(inputs["current_logged_action"])==14
    assert "next_measured_state" not in inputs
    assert "new_intervention_onset_when_policy_controlled" not in inputs
    targets=sample.factual_targets()
    assert len(targets["next_measured_state"])==14
    assert targets["next_intervention_onset_when_policy_controlled"] is None or (
        not sample.currently_human_controlled)
    return True
