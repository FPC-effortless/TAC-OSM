"""DEV-037 public-only acquisition and action-target contracts."""
from scripts.run_simulator_action_dev037 import choose,policy,teacher_action
from tac_osm.development.noisy_memory import parameter_hash


def test_teacher_chooses_lower_physical_cost():
    for wait in (0,1):
        for service in (0,1):
            action=teacher_action(wait,service)
            assert action==int(1+4*service<4*wait)


def test_selection_is_stratified_and_uses_only_public_probabilities():
    rows=[]
    for d in (32,128,512):
        for role in ('retention','acquisition','pending_cancel','reissued_cancel'):
            for deadline in ('before','at','after'):
                for rep in range(8):
                    for partner in (0,1):
                        rows.append(dict(delay=d,role=role,deadline=deadline,rep=rep,partner=partner,
                            probabilities=[[.1+.01*rep,.2,.4]]*8))
    random=choose(rows,'random');active=choose(rows,'active')
    assert len(random)==len(active)==288
    for selected in (random,active):
        strata={}
        for i in selected:
            row=rows[i];key=(row['delay'],row['role'],row['deadline'])
            strata.setdefault(key,set()).add(row['rep'])
        assert len(strata)==36 and all(len(reps)==4 for reps in strata.values())
        for i in selected:
            row=rows[i]
            assert any(rows[j]['rep']==row['rep'] and rows[j]['partner']!=row['partner'] and
                rows[j]['delay']==row['delay'] and rows[j]['role']==row['role'] and rows[j]['deadline']==row['deadline'] for j in selected)


def test_policy_initial_weights_reproduce_for_equal_budget_arms():
    assert parameter_hash(policy())==parameter_hash(policy())
    assert sum(p.numel() for p in policy().parameters())==81
