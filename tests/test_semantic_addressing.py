from tac_osm import Candidate, Query
from tac_osm.benchmark_v1 import generate_task
from tac_osm.semantic_addressing import (
    RepresentationAddressIndex,
    SemanticAddressConfig,
)


def _query_encoder(query, state):
    del state
    bits = query.text.partition("\t")[0]
    return tuple(float(x) for x in bits.split())


def _candidate_encoder(candidate, state):
    del state
    return tuple(float(x) for x in candidate.descriptor)


def test_semantic_address_index_is_bounded_by_probe_count_not_history():
    config = SemanticAddressConfig(dimensions=8, n_planes=8, seed=3)
    index = RepresentationAddressIndex.create(
        _query_encoder,
        _candidate_encoder,
        config=config,
    )
    for h in (64, 256):
        task = generate_task(
            100 + h,
            dim=8,
            n_candidates=h,
            relation="equality",
            validity="unique",
        )
        built = index.rebuild(task.candidates, context=task.query.context)
        hit = built.lookup(task.public(), k=8)
        assert hit.bucket_probes == 1
        assert hit.inspected_positions == 1
        assert hit.representation_dimension == 8
        assert len(hit.candidate_indices) <= 8


def test_semantic_addressing_never_receives_hidden_truth():
    captured = []

    def query_encoder(query, state):
        captured.append(query.text)
        del state
        return (1.0, 0.0)

    def candidate_encoder(candidate, state):
        del state
        return (float(candidate.descriptor[0]), float(candidate.descriptor[1]))

    task = generate_task(
        22,
        dim=8,
        n_candidates=16,
        relation="xor_parity",
        validity="unique",
    )
    index = RepresentationAddressIndex.create(
        query_encoder,
        candidate_encoder,
        config=SemanticAddressConfig(dimensions=2, n_planes=4, seed=7),
    )
    index = index.rebuild(task.candidates, context=task.query.context)
    index.lookup(task.public(), k=4)
    assert captured
    assert str(task.acceptable_actions) not in captured[-1]
    assert str(task.reference_bits) not in captured[-1]


def test_semantic_address_index_supports_hamming_one_probe_without_history_scan():
    config = SemanticAddressConfig(dimensions=8, n_planes=8, seed=5, hamming_radius=1)
    index = RepresentationAddressIndex.create(
        _query_encoder,
        _candidate_encoder,
        config=config,
    )
    task = generate_task(
        33,
        dim=8,
        n_candidates=64,
        relation="equality",
        validity="unique",
    )
    built = index.rebuild(task.candidates, context=task.query.context)
    hit = built.lookup(task.public(), k=4)
    assert hit.bucket_probes == 9
    assert hit.inspected_positions == 9
    assert len(hit.candidate_indices) <= 4
