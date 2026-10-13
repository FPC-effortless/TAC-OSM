"""Development-only tests for the actual TAC-OSM hardened-loop CED adapter."""
from __future__ import annotations

import dataclasses
import torch
import pytest

from tac_osm import Candidate, Query, StateRead, StateUpdate
from tac_osm.temporal import TemporalPersistentState
from tac_osm.addressing import ContentAddressIndex
from tac_osm.hardened import HardenedLoop, TemporalBenchmark
from tac_osm.development.ced_hardened_router import QueryOnlyCED, HardenedCEDRouter


def _candidates(n=7):
    return tuple(Candidate(key=f"c{i}", descriptor=tuple((i >> (j % 3)) & 1
                                    for j in range(8)), action=i) for i in range(n))


def test_query_only_shapes_and_true_attention_work():
    ref = torch.tensor([1., 0., 1., 0., 0., 0., 0., 0.])
    mask = torch.tensor([1., 1., 0., 0., 0., 0., 0., 0.])
    desc = torch.tensor([c.descriptor for c in _candidates()], dtype=torch.float32)
    for mode in ("dense", "shared", "sparse"):
        model = QueryOnlyCED(mode=mode, topk=3)
        out, ledger = model(ref, mask, desc)
        assert out.shape == (7,)
        assert ledger.encoder_score_pairs == 2 * 2 * 7 * 7
        assert ledger.first_global_score_pairs == 2 * 7
        if mode == "dense":
            assert ledger.global_kv_scalars == 2 * 2 * 7 * 32
        else:
            assert ledger.global_kv_scalars == 2 * 7 * 32
        if mode == "sparse":
            assert ledger.subsequent_global_score_pairs == 2 * 3
            assert ledger.final_score_pairs == 2 * 3
            assert (out == -1e6).sum() == 4
        else:
            assert ledger.subsequent_global_score_pairs == 2 * 7
            assert ledger.final_score_pairs == 2 * 7


def test_shared_sparse_matched_initial_state():
    torch.manual_seed(43)
    shared = QueryOnlyCED(mode="shared", topk=3)
    torch.manual_seed(43)
    sparse = QueryOnlyCED(mode="sparse", topk=3)
    for key, value in shared.state_dict().items():
        assert torch.equal(value, sparse.state_dict()[key]), key


def test_candidate_encoder_is_causal_prefix_invariant():
    torch.manual_seed(4)
    model = QueryOnlyCED(mode="shared")
    values = torch.randn(5, 16)
    more = torch.randn(4, 16) * 100
    model.eval()
    with torch.no_grad():
        original = model.candidate_embed(values)
        extended = model.candidate_embed(torch.cat((values, more)))
        for layer in model.encoder:
            original = layer(original)
            extended = layer(extended)
        torch.testing.assert_close(original, extended[:5], atol=1e-5, rtol=1e-5)


def test_task_gradient_flows_through_router_head():
    torch.manual_seed(9)
    model = QueryOnlyCED(mode="dense")
    ref = torch.tensor([1.,0,1,0,1,0,0,0])
    marks = torch.tensor([1.,1,1,1,0,0,0,0])
    descriptors = torch.rand(5, 8).round()
    logits, _ = model(ref, marks, descriptors)
    loss = torch.nn.functional.cross_entropy(logits[None], torch.tensor([3]))
    loss.backward()
    assert model.candidate_embed.weight.grad is not None
    assert model.candidate_embed.weight.grad.abs().sum() > 0
    assert model.kv_producers[0].weight.grad is not None
    assert model.kv_producers[0].weight.grad.abs().sum() > 0


def test_temporal_state_not_readable_early_and_used_when_available():
    state = TemporalPersistentState()
    router = HardenedCEDRouter(QueryOnlyCED())
    candidates = _candidates()
    query = Query(text="\tpersistent:probe", context=(1,1,0,0,0,0,0,0), step=2)
    state.stage_world_write(StateUpdate(key="persistent:probe",value=(1,0,1,1,0,0,1,0),step=0),delay=2)
    with pytest.raises(ValueError, match="public reference unavailable"):
        router(query, state, candidates)
    state.advance_to(1)
    with pytest.raises(ValueError, match="public reference unavailable"):
        router(query, state, candidates)
    state.advance_to(2)
    decision = router(query, state, candidates)
    assert 0 <= decision.selected < len(candidates)
    assert len(decision.scores) == len(candidates)
    assert router.last_public_source == "persistent_state"
    assert router.last_cost.public_state_reads == 1


def test_fallback_public_query_with_no_memory_and_errors_fail_closed():
    router = HardenedCEDRouter(QueryOnlyCED())
    state = TemporalPersistentState()
    q = Query("1 0 1 1 0 0 1 0\t", context=(1,1,0,0,0,0,0,0))
    got = router(q, state, _candidates())
    assert router.last_public_source == "query"
    assert got.provenance == "dev008.ced_router"
    with pytest.raises(ValueError):
        router(Query("\tbad", context=q.context), state, _candidates())
    with pytest.raises(ValueError):
        router(q, state, ())
    with pytest.raises(ValueError):
        router(dataclasses.replace(q, context=(1,0)), state, _candidates())


def test_hardened_loop_runs_full_execution_and_delayed_state():
    torch.manual_seed(4)
    model = QueryOnlyCED(mode="shared")
    model.eval()
    router = HardenedCEDRouter(model)
    benchmark = TemporalBenchmark(seed=11, dim=8, n_candidates=16, filler_candidates=4)
    benchmark.schedule_lookup_probe(write_step=0, delay=2, key="persistent:probe",
                                    value=(1,0,1,0,1,0,1,0))
    loop = HardenedLoop(router=router, benchmark=benchmark, repair=None,
                        learning_enabled=False)
    loop.step(0)
    loop.step(1)
    assert not benchmark.state.available("persistent:probe")
    loop.step(2)
    assert benchmark.state.available("persistent:probe")
    assert router.last_public_source == "persistent_state"
    assert loop.costs.executor_invocations == 3
    assert loop.costs.verifier_checks == 3
    assert loop.costs.candidates_routed == 4 + 4 + 16


def test_hardened_loop_index_preserves_retrieval_boundary():
    torch.manual_seed(13)
    model = QueryOnlyCED(mode="sparse", topk=2)
    model.eval()
    router = HardenedCEDRouter(model)
    benchmark = TemporalBenchmark(seed=20, dim=8, n_candidates=32, filler_candidates=4)
    benchmark.schedule_lookup_probe(write_step=0, delay=2, key="persistent:probe",
                                    value=(1,0,1,0,1,0,1,0))
    loop = HardenedLoop(router=router, benchmark=benchmark, index=ContentAddressIndex(),
                        index_k=4, repair=None, learning_enabled=False)
    loop.step(0)
    loop.step(1)
    loop.step(2)
    assert loop.costs.index_build_candidates > 0
    assert loop.costs.address_query_positions > 0
    assert loop.costs.candidates_routed <= 3 * 4
    assert router.last_cost.candidates <= 4
    assert loop.costs.executor_invocations == 3


def test_public_state_read_cannot_be_turned_into_oracle_metadata():
    class ReadOnlyView:
        def __init__(self):
            self.calls = 0
        def read(self, query):
            self.calls += 1
            return StateRead(keys=("k",), values=((0,1,1,0,0,1,0,1),),slot_used=(True,))
        def __getattr__(self, key):
            raise AssertionError(f"forbidden state access: {key}")
    state = ReadOnlyView()
    router = HardenedCEDRouter(QueryOnlyCED(mode="shared"))
    q = Query("\tk", context=(1,1,1,1,0,0,0,0))
    decision = router(q, state, _candidates())
    assert state.calls == 1
    assert len(decision.scores) == 7
