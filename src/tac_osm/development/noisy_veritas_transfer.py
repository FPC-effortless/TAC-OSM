"""DEV-052A: public sensor noise / uncertain second repair via native Veritas.

Hidden outcomes are modified ONLY in generated oracle effects. No policy receives
hidden state, repair effectiveness or true sensor bit except through a noisy and
charged observable inspection receipt. This is an OOD frozen-transfer test.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import math
import random
from copy import deepcopy

from .veritas_public_challenge import make_episode
from .learned_veritas_policy import PublicBelief,choose_action,ACTIONS

ACCURACY=.85
REPAIR_SUCCESS=(.75,.80,1.0)
SENSOR_LIMIT=4


def make_shift_episode(seed,index):
    original=make_episode(seed,index)
    copied=original.model_copy(deep=True)
    rng=random.Random(f"dev052-independent-second:{seed}:{index}")
    modified=[]
    for effect in copied.oracle.action_effects:
        if effect.action_name=="service" and effect.required_parameters.get("attempt")==2:
            e=int(effect.required_parameters["entity"])
            old=int(copied.oracle.initial_state[f"e{e}.fault"])
            success=rng.random()<REPAIR_SUCCESS[1]
            effect=effect.model_copy(update={"set_state":{
                **effect.set_state,f"e{e}.fault":0 if success else old}})
        modified.append(effect)
    from investigation_world.operational.models import HiddenActionEffect,StateAssertion
    for entity in range(2):
        modified.append(HiddenActionEffect(
            action_name="service",
            required_parameters={"entity":entity,"attempt":3},
            required_state=[
                StateAssertion(object_id="plant",field_name="slot",expected_value=entity),
                StateAssertion(object_id=f"e{entity}",field_name="attempt",expected_value=2)],
            set_state={f"e{entity}.fault":0,f"e{entity}.attempt":3,"plant.slot":-1},
            observable_result={"accepted":True}))
    copied.oracle=copied.oracle.model_copy(update={"action_effects":modified,"max_cost":240,"max_tool_calls":30})
    task=copied.task.model_copy(deep=True)
    task.constraints=list(task.constraints)+[
        "Inspections are noisy (85% sensitivity/specificity); second repair can fail.",
        "Third service attempt is available, and succeeds in this synthetic study only."]
    task.metadata={**task.metadata,"inspection_accuracy":ACCURACY,
                   "repair_success_probabilities":list(REPAIR_SUCCESS),
                   "max_service_attempts":3}
    copied.task=task
    copied.metadata={**copied.metadata,"sensor_accuracy":ACCURACY}
    return copied


def observed_inspect(runtime,seed,index,entity,read_number):
    """Harness-only sensor corruption; result is public, state remains private."""
    result=runtime.act("inspect",entity=entity)
    true=int(runtime.state[f"e{entity}.fault"])
    rng=random.Random(f"dev052-independent-sensor:{seed}:{index}:{entity}:{read_number}")
    bit=true if rng.random()<ACCURACY else 1-true
    return {**result,"fault":bit}


def posterior(prior,reading):
    if not 0<=prior<=1 or reading not in (0,1):raise ValueError("invalid Bayes inputs")
    a=prior*(ACCURACY if reading else 1-ACCURACY)
    b=(1-prior)*((1-ACCURACY) if reading else ACCURACY)
    return a/(a+b) if a+b>0 else prior


@lru_cache(maxsize=None)
def _value(probability,attempts,senses):
    """Bellman recursion for known synthetic Bernoulli dynamics and paid tools.

    Public prior is 0.5 because DEV-050 records are independently correct .5.
    No future labels, world IDs or hidden outcome values enter this recursion.
    """
    p=float(probability)
    if attempts==3:return 4*p,"done"
    options=[(4*p,"done")]
    nextp=p*(1-REPAIR_SUCCESS[attempts])
    vs,_=_value(round(nextp,10),attempts+1,senses)
    options.append((1.2+vs,"service"))
    if senses<SENSOR_LIMIT:
        py=p*ACCURACY+(1-p)*(1-ACCURACY)
        positive=posterior(p,1)
        negative=posterior(p,0)
        plus,_=_value(round(positive,10),attempts,senses+1)
        minus,_=_value(round(negative,10),attempts,senses+1)
        options.append((.1+py*plus+(1-py)*minus,"inspect"))
    best=min(options,key=lambda z:(z[0],("inspect","service","done").index(z[1])))
    return best


def public_bayes_action(prior,attempts,senses):
    return _value(round(prior,10),attempts,senses)[1]


def run_ood(episode,seed,index,name,model=None,gate=None):
    """One public episode under a common native runtime for every arm."""
    from investigation_world.operational.models import EpisodeSubmission
    from investigation_world.operational.runtime import OperationalRuntime
    runtime=OperationalRuntime(episode)
    public=runtime.public_payload()
    assert "oracle" not in public and "initial_state" not in str(public)
    actions=[];fallbacks=decisions=0;escalations=0
    for rec in public["records"]:
        e=int(rec["fields"]["entity"])
        state=PublicBelief(reported_fault=int(rec["fields"]["reported_fault"]))
        prior=.5
        senses=0;attempts=0;isolated=False
        for step in range(16):
            if name=="wait":
                command="done"
            elif name=="all_three":
                command="service" if isolated else "isolate"
                if attempts>=3:command="done"
            elif name=="bayesian_dp":
                command="service" if isolated else public_bayes_action(prior,attempts,senses)
                if command=="service" and not isolated:command="isolate"
            elif name=="legacy_verify_retry":
                # Transfer the DEV-050 explicit controller to the noisy receipts;
                # intentionally stops at two attempts on this OOD task.
                if state.attempts>=2:
                    command="done"
                elif state.known_fault is None:command="inspect"
                elif state.known_fault==0:command="done"
                elif state.isolated:command="service"
                else:command="isolate"
            else:
                # Former learner is FROZEN. External safety capacity guard is
                # not a learned third-attempt capability; count handoffs.
                if state.attempts>=2:
                    command="done"
                    if state.known_fault!=0:escalations+=1
                else:
                    choice,fallback,confidence=choose_action(state,model,gate)
                    command=ACTIONS[choice]
                    fallbacks+=int(fallback and model is not None)
                    decisions+=1
            if command=="done":
                actions.append(dict(entity=e,command="done"))
                if isolated:
                    # This indicates a policy left the shared resource occupied;
                    # do NOT silently clear it or grant success.
                    escalations+=1
                break
            if command=="inspect":
                receipt=observed_inspect(runtime,seed,index,e,senses)
                senses+=1
                prior=posterior(prior,int(receipt["fault"]))
                state.observe("inspect",receipt)
            elif command=="isolate":
                receipt=runtime.act("isolate",entity=e)
                isolated=bool(receipt.get("accepted"))
                state.observe("isolate",receipt)
            elif command=="service":
                if not isolated:
                    # Feasibility is enforced, without letting oracle outcomes
                    # influence the model's chosen action.
                    receipt={"accepted":False,"reason":"isolation_required"}
                else:
                    receipt=runtime.act("service",entity=e,attempt=attempts+1)
                    if receipt.get("accepted"):
                        prior*=1-REPAIR_SUCCESS[attempts]
                        attempts+=1;isolated=False
                state.observe("service",receipt)
            else:raise AssertionError("unknown action "+command)
            actions.append(dict(entity=e,command=command,receipt=dict(receipt)))
        else:
            escalations+=1
            actions.append(dict(entity=e,command="step_limit"))
    snapshot=runtime.state_snapshot()
    verification=runtime.submit(EpisodeSubmission(conclusion="DEV-052 noisy repair transfer"))
    fail=sum(snapshot[f"e{e}.fault"] for e in range(2))
    return dict(world=public["episode_id"],policy=name,failures=fail,
                complete=int(verification.state==1),
                cost=runtime.budget.spent/10+4*fail,
                tool_calls=len(runtime.events),
                inspections=sum(t.action_name=="inspect" for t in runtime.events),
                blocked=sum(t.blocked for t in runtime.events),
                escalations=escalations,fallbacks=fallbacks,decisions=decisions,
                actions=actions,verification=verification.model_dump(mode="json"))


def study_cohort(seeds,worlds):
    for seed in seeds:
        for index in range(worlds):
            yield seed,index,make_shift_episode(seed,index)


def summarize(rows):
    n=len(rows)
    if not n:raise ValueError("empty cohort")
    return dict(worlds=n,cost_per_entity=sum(r["cost"] for r in rows)/(2*n),
                residual_faults=sum(r["failures"] for r in rows),
                completed_worlds=sum(r["complete"] for r in rows),
                blocked_actions=sum(r["blocked"] for r in rows),
                escalations=sum(r["escalations"] for r in rows),
                mean_calls=sum(r["tool_calls"] for r in rows)/n,
                mean_inspections=sum(r["inspections"] for r in rows)/n,
                fallback_rate=sum(r["fallbacks"] for r in rows)/max(1,sum(r["decisions"] for r in rows)))
