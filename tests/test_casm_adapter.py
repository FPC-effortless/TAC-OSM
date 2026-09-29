from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from enum import Enum

import pytest

from tac_osm import Structure
from tac_osm import casm_adapter


class FakeOp(str, Enum):
    INPUT = "INPUT"
    NOT = "NOT"
    AND = "AND"
    OR = "OR"
    XOR = "XOR"


@dataclass(frozen=True)
class FakeNode:
    index: int
    op: FakeOp
    depth: int
    slot: int

    @property
    def arity(self) -> int:
        return {
            FakeOp.INPUT: 0,
            FakeOp.NOT: 1,
            FakeOp.AND: 2,
            FakeOp.OR: 2,
            FakeOp.XOR: 2,
        }[self.op]


@dataclass(frozen=True)
class FakeEdge:
    src: int
    dst: int
    port: int


@dataclass(frozen=True)
class FakeEpisode:
    nodes: tuple[FakeNode, ...]
    true_edges: tuple[FakeEdge, ...]
    candidate_edges: tuple[FakeEdge, ...]
    inputs: tuple[int, ...]
    output: int
    input_values: tuple[int, ...]
    target: int
    truth_table: dict[tuple[int, ...], int]
    active_count: int


def make_episode() -> FakeEpisode:
    # Candidate ordering deliberately contains two distractors. The CASM-S
    # gate vector is indexed by this exact order:
    #
    #   0: (0 -> 2, port 0)  [true]
    #   1: (1 -> 2, port 0)  [distractor]
    #   2: (0 -> 2, port 1)  [distractor]
    #   3: (1 -> 2, port 1)  [true]
    #
    # Node 3 is a padded node, matching the source generator's variable-size
    # fixed substrate. Only the first active_count nodes are executable.
    nodes = (
        FakeNode(0, FakeOp.INPUT, 0, 0),
        FakeNode(1, FakeOp.INPUT, 0, 1),
        FakeNode(2, FakeOp.XOR, 1, 2),
        FakeNode(3, FakeOp.INPUT, 0, 3),
    )
    candidate_edges = (
        FakeEdge(0, 2, 0),
        FakeEdge(1, 2, 0),
        FakeEdge(0, 2, 1),
        FakeEdge(1, 2, 1),
    )
    true_edges = (candidate_edges[0], candidate_edges[3])
    return FakeEpisode(
        nodes=nodes,
        true_edges=true_edges,
        candidate_edges=candidate_edges,
        inputs=(0, 1),
        output=2,
        input_values=(1, 0),
        target=3,
        truth_table={(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 0},
        active_count=3,
    )


def test_episode_serialization_matches_casm_s_field_names_and_edge_order():
    spec = casm_adapter.casm_graph_spec_from_episode(make_episode())

    assert spec["schema"] == casm_adapter.CASM_SCHEMA
    assert spec["kind"] == casm_adapter.CASM_KIND
    assert spec["active_count"] == 3
    assert spec["inputs"] == [0, 1]
    assert spec["output"] == 2

    assert [node["index"] for node in spec["nodes"]] == [0, 1, 2, 3]
    assert [node["op"] for node in spec["nodes"]] == [
        "INPUT", "INPUT", "XOR", "INPUT"
    ]

    edges = spec["candidate_edges"]
    assert [edge["index"] for edge in edges] == [0, 1, 2, 3]
    assert [(edge["src"], edge["dst"], edge["port"]) for edge in edges] == [
        (0, 2, 0),
        (1, 2, 0),
        (0, 2, 1),
        (1, 2, 1),
    ]


def test_serialized_spec_is_json_safe_and_contains_no_oracle_fields():
    spec = casm_adapter.casm_graph_spec_from_episode(make_episode())

    encoded = json.dumps(spec)
    assert json.loads(encoded) == spec

    forbidden = {
        "true_edges",
        "true_edge_set",
        "target",
        "input_values",
        "truth_table",
        "acceptable_actions",
    }
    assert forbidden.isdisjoint(spec)

    # Hidden information is absent recursively too. This catches a future
    # implementation that accidentally nests an oracle episode under metadata.
    encoded_lower = encoded.lower()
    for name in forbidden:
        assert name.lower() not in encoded_lower


def test_structure_helper_places_the_serialized_graph_in_structure_spec():
    structure = casm_adapter.casm_structure_from_episode(
        make_episode(),
        key="candidate-7",
        provenance="casm_s.test",
    )

    assert isinstance(structure, Structure)
    assert structure.key == "candidate-7"
    assert structure.provenance == "casm_s.test"
    assert structure.spec["kind"] == casm_adapter.CASM_KIND


def test_fake_casm_observes_the_same_edge_indexing_contract():
    observed = {}

    def fake_casm(spec, inputs):
        observed["spec"] = spec
        observed["inputs"] = tuple(inputs)

        edges = spec["candidate_edges"]
        assert all(edge["index"] == i for i, edge in enumerate(edges))

        # Mimic CASM-S forward(): one gate per candidate edge, aligned by
        # candidate_edges position. The fake learned router has selected the
        # two true positions internally; no true_edges field is supplied by
        # the serialization seam.
        gates = [1.0, 0.0, 0.0, 1.0]
        values = [0.0] * spec["active_count"]
        for input_position, node_index in enumerate(spec["inputs"]):
            values[node_index] = float(inputs[input_position])

        incoming = {}
        for edge, gate in zip(edges, gates):
            if gate > 0.5:
                incoming[edge["port"]] = edge["src"]

        a = values[incoming[0]]
        b = values[incoming[1]]
        values[2] = float(a != b)

        return {
            "output": values[spec["output"]],
            "gates": gates,
            "node_values": values,
            "provenance": "fake_casm_s",
        }

    adapter = casm_adapter.CasmExecutorAdapter(fake_casm)
    structure = casm_adapter.casm_structure_from_episode(
        make_episode(), key="candidate-7"
    )
    result = adapter.execute(structure, (1.0, 0.0))

    assert observed["inputs"] == (1.0, 0.0)
    assert observed["spec"] == structure.spec
    assert result.output == 1.0
    assert result.gates == (1.0, 0.0, 0.0, 1.0)
    assert result.node_values == (1.0, 0.0, 1.0)
    assert result.provenance == "fake_casm_s"


def test_adapter_accepts_tensor_like_outputs_without_importing_tensor_library():
    class Scalar:
        def item(self):
            return 1.0

    class Vector:
        def tolist(self):
            return [1.0, 0.0, 1.0]

    def fake_casm(spec, inputs):
        del inputs
        return {
            "output": Scalar(),
            "gates": [1.0, 0.0, 0.0, 1.0],
            "node_values": Vector(),
            "provenance": "tensor_like_fake",
        }

    adapter = casm_adapter.CasmExecutorAdapter(fake_casm)
    structure = casm_adapter.casm_structure_from_episode(
        make_episode(), key="tensor-like"
    )
    result = adapter.execute(structure, (1.0, 0.0))

    assert result.output == 1.0
    assert result.node_values == (1.0, 0.0, 1.0)


@pytest.mark.parametrize(
    ("field", "payload", "match"),
    [
        (
            "gates",
            {
                "output": 1.0,
                "gates": [1.0, 0.0],
                "node_values": [1.0, 0.0, 1.0],
            },
            "gates length",
        ),
        (
            "node_values",
            {
                "output": 1.0,
                "gates": [1.0, 0.0, 0.0, 1.0],
                "node_values": [1.0],
            },
            "node_values length",
        ),
    ],
)
def test_structured_result_cardinality_is_fail_closed(field, payload, match):
    del field

    adapter = casm_adapter.CasmExecutorAdapter(lambda spec, inputs: payload)
    structure = casm_adapter.casm_structure_from_episode(
        make_episode(), key="bad-cardinality"
    )
    with pytest.raises(ValueError, match=match):
        adapter.execute(structure, (1.0, 0.0))


def test_input_vector_cardinality_is_fail_closed():
    adapter = casm_adapter.CasmExecutorAdapter(
        lambda spec, inputs: {
            "output": 1.0,
            "gates": [1.0, 0.0, 0.0, 1.0],
            "node_values": [1.0, 0.0, 1.0],
        }
    )
    structure = casm_adapter.casm_structure_from_episode(
        make_episode(), key="bad-inputs"
    )
    with pytest.raises(ValueError, match="inputs length"):
        adapter.execute(structure, (1.0,))


def test_serialized_edge_index_tampering_is_rejected_before_execution():
    spec = casm_adapter.casm_graph_spec_from_episode(make_episode())
    spec["candidate_edges"][2]["index"] = 7

    calls = []

    def fake_casm(spec, inputs):
        calls.append((spec, inputs))
        return {
            "output": 1.0,
            "gates": [1.0, 0.0, 0.0, 1.0],
            "node_values": [1.0, 0.0, 1.0],
        }

    adapter = casm_adapter.CasmExecutorAdapter(fake_casm)
    with pytest.raises(ValueError, match="preserve list order"):
        adapter.execute(Structure(key="tampered", spec=spec), (1.0, 0.0))

    assert calls == []


def test_casm_adapter_module_has_no_torch_import():
    source = inspect.getsource(casm_adapter)
    assert "import torch" not in source
    assert "from torch" not in source


def test_typed_graph_spec_is_accepted_as_structure_spec():
    episode = make_episode()
    typed = casm_adapter.CasmGraphSpec.from_episode(episode)

    seen = {}

    def fake_casm(spec, inputs):
        seen["spec"] = spec
        return {
            "output": 1.0,
            "gates": [1.0, 0.0, 0.0, 1.0],
            "node_values": [1.0, 0.0, 1.0],
        }

    result = casm_adapter.CasmExecutorAdapter(fake_casm).execute(
        Structure(key="typed", spec=typed),
        (1.0, 0.0),
    )

    assert result.output == 1.0
    assert seen["spec"] == typed.to_dict()
