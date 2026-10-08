"""Contract tests for the fused TAC-OSM research surface."""
from __future__ import annotations
from tac_osm import Candidate, Query, StateRead, Outcome
from tac_osm.environment import build_relational_task
from tac_osm.fusion.ablation import full_config, lesion_matrix
from tac_osm.fusion.builder import build_fused_model
from tac_osm.fusion.coordinator import CoordinatorConfig, SparseCoordinator
from tac_osm.fusion.gates import all_gates, router_observation_gate
from tac_osm.fusion.interfaces import MemoryContext, RegimeContext
from tac_osm.fusion.memory import MemoryConfig, MultiScaleMemory
from tac_osm.fusion.verifier import FusionVerifier, FusionVerifierConfig

def test_fused_model_constructs_and_exposes_all_modules():
    built=build_fused_model(full_config(0),n_steps=3)
    inspected=built.model.inspect()
    assert inspected["model"]=="tacosm_fused_bioinspired_2026-10-08"
    assert len(inspected["operators"]["modules"])==8
    assert inspected["coordinator"]["n_modules"]==8
    assert "decays" in inspected["memory"]

def test_fused_model_runs_without_target_in_router_query():
    built=build_fused_model(full_config(0),n_steps=3)
    steps=built.model.run()
    assert len(steps)==3
    for step in steps:
        assert "target_action" not in step.query.__dict__
        assert step.provenance["router_sees_target"] is False
        assert step.provenance["outcome_created_after_route"] is True

def test_sparse_coordination_reduces_module_candidates():
    common=dict(n_modules=8,top_k=2,dim=8,n_buckets=4,seed=0)
    sparse=SparseCoordinator(CoordinatorConfig(mode="sparse",**common))
    dense=SparseCoordinator(CoordinatorConfig(mode="dense",**common))
    query=Query(text="0 1 0 1 0 1 0 1",context=(1,0,1,0,1,0,1,0))
    empty=StateRead(keys=(),values=(),slot_used=())
    memory=MemoryContext((),(),(),0.0,0.0)
    regime=RegimeContext((0.0,0.0,0.0,0.0),0.5,0.0,0.0,0.5)
    s=sparse.route(query=query,state_read=empty,memory=memory,regime=regime)
    d=dense.route(query=query,state_read=empty,memory=memory,regime=regime)
    assert len(s.candidate_module_ids)<len(d.candidate_module_ids)
    assert len(s.selected_modules)==2
    assert len(d.selected_modules)==2

def test_multi_timescale_memory_has_distinct_decay_scales():
    memory=MultiScaleMemory(MemoryConfig(n_timescales=4,decays=(0.2,0.6,0.9,0.985),dim=4,n_modules=2))
    memory.observe(module_ids=(0,),signal=(1,1,1,1),reward=1.0,surprise=1.0,success=True)
    values=[row[0] for row in memory.context((0,)).global_states]
    assert values[0]!=values[-1]
    assert values[0]>values[-1]

def test_memory_reset_really_removes_dynamic_state():
    memory=MultiScaleMemory(MemoryConfig(dim=4,n_modules=2))
    memory.observe(module_ids=(0,),signal=(1,0,1,0),reward=1.0,surprise=1.0,success=True)
    assert memory.inspect()["priming_norm"]>0
    memory.reset()
    report=memory.inspect()
    assert report["priming_norm"]==0.0
    assert all(norm==0.0 for norm in report["global_norms"])

def test_lesion_excludes_requested_module():
    cfg=lesion_matrix(0)[3]
    steps=build_fused_model(cfg,n_steps=2).model.run()
    assert all(3 not in step.coordination.selected_modules for step in steps)

def test_failed_outcome_cannot_commit_under_verified_write_gate():
    verifier=FusionVerifier(FusionVerifierConfig(mode="path",require_positive_outcome_for_commit=True))
    assert verifier.commit_allowed(passed=False,outcome=Outcome(False)) is False

def test_integrity_gates_pass_before_experiments():
    results=all_gates()
    assert results
    assert all(result.passed for result in results)
    assert router_observation_gate().passed

def test_public_task_does_not_expose_hidden_target():
    task=build_relational_task(123,dim=8,n_candidates=8)
    public=task.public()
    assert "target_action" not in public.__dict__
    assert task.target_action>=0
    assert len(task.candidates)==8
