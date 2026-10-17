"""DEV-042 paired history causal, last-event indistinguishability contracts."""
from tac_osm.development.order_causal_cases import make_order_pair,last_observations
from tac_osm.development.schema_maintenance import exact_public_targets,execute


def test_mature_wait_is_opposite_for_both_partners():
    for deadline in ("at","after"):
        a,b=make_order_pair(1,128,32,deadline)
        assert (a.wait_targets[a.target_entity],b.wait_targets[b.target_entity])==(1,0)
        assert a.service_targets==b.service_targets


def test_before_due_future_is_not_different():
    a,b=make_order_pair(2,128,64,"before")
    assert a.wait_targets==b.wait_targets


def test_last_observations_and_query_are_identical():
    for delay in (32,128,512):
        a,b=make_order_pair(3,512,delay,"at")
        assert last_observations(a.observations)==last_observations(b.observations)
        assert (a.tick,a.horizon,a.target_entity)==(b.tick,b.horizon,b.target_entity)
        assert a.fingerprint!=b.fingerprint


def test_public_reference_is_exact_in_each_partner():
    for deadline in ("before","at","after"):
        for c in make_order_pair(4,512,128,deadline):
            assert exact_public_targets(c.observations,c.tick,c.horizon)==c.wait_targets
            assert len(c.observations)==512
            assert all(o.tick==i for i,o in enumerate(c.observations,1))


def test_generation_reset_flips_only_the_future_target():
    a,b=make_order_pair(5,128,32,"at")
    target=a.target_entity
    schedule_a=next(o for o in a.observations if o.kind=="scheduled" and o.entity==target)
    schedule_b=next(o for o in b.observations if o.kind=="scheduled" and o.entity==target)
    assert schedule_a.due==schedule_b.due==a.tick+32
    assert schedule_a.tick==9 and schedule_b.tick==10
    assert a.current_targets==b.current_targets
    assert sum(x!=y for x,y in zip(a.wait_targets,b.wait_targets))==1


def test_world_physical_intervention_and_last_event_information_barrier():
    a,b=make_order_pair(6,128,32,"at")
    target=a.target_entity
    assert last_observations(a.observations)==last_observations(b.observations)
    initial=execute(a.world,a.horizon,(),0)
    assert initial["failures"]>=1
    res=execute(a.world,a.horizon,(target,),0)
    assert res["failures"]==initial["failures"]-1 and res["unnecessary_services"]==0
    bres=execute(b.world,b.horizon,(target,),0)
    assert bres["unnecessary_services"]==1
