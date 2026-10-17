"""DEV-052 bounded OOD Veritas: pay for inspections and uncertain repairs."""
import pytest

from tac_osm.development.noisy_veritas_transfer import (
    ACCURACY,REPAIR_SUCCESS,SENSOR_LIMIT,posterior,public_bayes_action,
    make_shift_episode,observed_inspect,run_ood,study_cohort)


def test_registered_sensor_and_repair_probabilities():
    assert (ACCURACY,REPAIR_SUCCESS,SENSOR_LIMIT)==(.85,(.75,.8,1.),4)
    assert posterior(.5,1)==pytest.approx(.85)
    assert posterior(.5,0)==pytest.approx(.15)


def test_posterior_handles_repeated_independent_public_sensor_reads():
    after=posterior(posterior(.5,1),0)
    assert after==pytest.approx(.5)
    assert posterior(posterior(.5,1),1)>.85


def test_native_public_episode_discloses_parameters_not_oracle():
    pytest.importorskip("investigation_world.operational.runtime")
    ep=make_shift_episode(520201,0)
    public=ep.public_payload()
    assert "oracle" not in public
    assert "initial_state" not in str(public)
    assert public["task"]["metadata"]["inspection_accuracy"]==.85
    assert any(e.required_parameters=={"entity":0,"attempt":3}
               for e in ep.oracle.action_effects)


def test_independent_paid_noisy_sensor_is_deterministic_per_entity_and_read():
    pytest.importorskip("investigation_world.operational.runtime")
    from investigation_world.operational.runtime import OperationalRuntime
    a=OperationalRuntime(make_shift_episode(520201,4))
    b=OperationalRuntime(make_shift_episode(520201,4))
    vals_a=[observed_inspect(a,520201,4,1,i)["fault"] for i in range(5)]
    vals_b=[observed_inspect(b,520201,4,1,i)["fault"] for i in range(5)]
    assert vals_a==vals_b
    assert a.budget.spent==b.budget.spent==5


def test_third_native_repair_succeeds_and_is_charged():
    pytest.importorskip("investigation_world.operational.runtime")
    from investigation_world.operational.runtime import OperationalRuntime
    candidate=next(make_shift_episode(520201,i) for i in range(2000)
                   if make_shift_episode(520201,i).oracle.initial_state["e0.fault"]==1
                   and any(e.action_name=="service" and e.required_parameters=={"entity":0,"attempt":2}
                   and e.set_state["e0.fault"]==1 for e in make_shift_episode(520201,i).oracle.action_effects)
                   and any(e.action_name=="service" and e.required_parameters=={"entity":0,"attempt":1}
                   and e.set_state["e0.fault"]==1 for e in make_shift_episode(520201,i).oracle.action_effects))
    rt=OperationalRuntime(candidate)
    for attempt in (1,2,3):
        assert rt.act("isolate",entity=0)["accepted"]
        assert rt.act("service",entity=0,attempt=attempt)["accepted"]
    assert rt.state_snapshot()["e0.fault"]==0
    assert rt.state_snapshot()["e0.attempt"]==3
    assert rt.budget.spent==36


def test_explicit_bayes_policy_and_native_cost_have_no_hidden_state_arguments():
    assert public_bayes_action(.5,0,0) in ("inspect","service","done")
    assert public_bayes_action(0.,0,0)=="done"
    assert public_bayes_action(1.,3,3)=="done"


def test_all_three_services_eliminate_faults_under_registered_synthetic_guarantee():
    pytest.importorskip("investigation_world.operational.runtime")
    seed,index,ep=next(study_cohort([520201],1))
    result=run_ood(ep,seed,index,"all_three")
    assert result["failures"]==0 and result["blocked"]==0
    assert result["complete"]==1
    assert result["cost"]==pytest.approx(7.2)
    assert result["tool_calls"]==12


def test_public_bayes_and_legacy_have_repeatable_native_outcomes():
    pytest.importorskip("investigation_world.operational.runtime")
    for name in ("bayesian_dp","legacy_verify_retry","wait"):
        x=run_ood(make_shift_episode(520201,2),520201,2,name)
        y=run_ood(make_shift_episode(520201,2),520201,2,name)
        assert x==y
        assert x["blocked"]==0


def test_calibration_and_evaluation_world_namespace_disjoint():
    pytest.importorskip("investigation_world.operational.runtime")
    ca={ep.episode_id for _,_,ep in study_cohort([520101],3)}
    ev={ep.episode_id for _,_,ep in study_cohort([520201],3)}
    assert ca.isdisjoint(ev)
