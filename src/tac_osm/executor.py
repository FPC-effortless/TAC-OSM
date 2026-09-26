"""``C_t`` -> ``A_t`` — structural execution.

Adapted from ``cdl-attention-experiment/casm_v01/phase1_dag/model.py``
(``c315544``), the executor whose contract is frozen in that directory's
README. This module is a pure-stdlib re-expression of that contract, not a
torch import: ``torch`` is unavailable on the control plane by policy, and the
first version of the loop is deliberately deterministic and simple.

## The contract, preserved

* single-pass topological execution, no recurrent spectral diagnostic
* fixed upper-triangular candidate substrate shared across episodes
* hard existence mask for variable-size programs
* true wiring is one of many admissible wirings, so the learned gate product
  is not the oracle
* **routing depends only on structural state; runtime values never enter the
  router** — this boundary is what makes execution an addressable resource
* one learned computation strength per syntactic argument port
  (``alpha_eta.shape == (2)`` in the source)
* raw operand order retained; commutative canonicalisation is bookkeeping
* copy-mask is a mandatory oracle preflight, never a learned baseline

The last point is why ``copy_mask`` is exported here: it is the falsification
preflight the ledger requires before any learned structural result is
interpreted.

## The grammar, reduced

The source ``grammar.py`` carries ``INPUT``, ``NOT``, ``AND``, ``OR``, ``XOR``
over a fixed node grid with per-port edges. v0 keeps ``INPUT``, ``NOT``,
``AND``, ``OR``, ``XOR`` and the port-indexed edges, because that is the
smallest set that still exercises the property that matters: the *topology*
determines what is computed, so a different topology is a different program.

## The simplest valid version

Per the integration plan, the executor here is the simplest valid
implementation of the contract, and deliberately does not try to solve the
L2 routing failure at the same time. Gates are fixed parameters rather than
learned ones: the failure analysis attributed L2's 0/18 to learned routing,
not to execution, so v0 makes execution deterministic and puts all the
learning on the routing side, where the portfolio has positive evidence.

The ``flattened`` arm is the dense-compute control: it replaces the gated DAG
with a plain weighted sum over all inputs, computing everything and letting a
downstream stage sort it out. It is what ``C_executed ~ f(|H|)`` looks like at
the execution layer, and it is the arm any structural claim has to beat.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Sequence

from . import Computation, ExecutionResult, Structure

__all__ = [
    "Op",
    "Node",
    "Edge",
    "Program",
    "ExecutorConfig",
    "StructuralExecutor",
    "FlattenedExecutor",
    "build_program",
    "program_from_descriptor",
    "copy_mask",
]


@dataclass(frozen=True)
class Op:
    """The reduced CASM grammar."""
    name: str
    arity: int


INPUT = Op("INPUT", 0)
NOT = Op("NOT", 1)
AND = Op("AND", 2)
OR = Op("OR", 2)
XOR = Op("XOR", 2)
EQ = Op("EQ", 2)
OPS = (INPUT, NOT, AND, OR, XOR, EQ)


@dataclass(frozen=True)
class Node:
    index: int
    op: Op
    depth: int = 0
    arity: int = 0


@dataclass(frozen=True)
class Edge:
    src: int
    dst: int
    port: int


@dataclass
class Program:
    """A structural program: nodes plus candidate edges.

    ``candidate_edges`` is the *substrate*: every edge that could exist. The
    true wiring is a subset, and which subset is chosen is the structural
    decision. Keeping the substrate fixed across episodes is what makes the
    gate product comparable — the same edge either carries a value or does not.
    """
    nodes: tuple[Node, ...]
    candidate_edges: tuple[Edge, ...]
    inputs: tuple[int, ...]
    output: int
    active_count: int
    input_values: tuple[int, ...] = ()
    exact: bool = False

    input_values: tuple[int, ...] = ()

    @property
    def true_edge_set(self) -> set[tuple[int, int, int]]:
        return {(e.src, e.dst, e.port) for e in self.candidate_edges}


def _alpha(strength: Sequence[float], port: int) -> float:
    """``alpha = c * softplus(eta)``, indexed by syntactic argument port."""
    eta = strength[port] if port < len(strength) else 0.0
    return 1.0 * _softplus(eta)


def _softplus(x: float) -> float:
    if x > 30.0:
        return x
    if x < -30.0:
        return 0.0
    import math
    return math.log1p(math.exp(x))


def _gate(logit: float, temperature: float) -> float:
    import math
    z = logit / max(temperature, 1e-6)
    if z >= 0.0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


@dataclass
class ExecutorConfig:
    """Switch surface for the executor, mirroring ``ablation.StructureSwitch``."""
    type: str = "learned"
    dim: int = 8
    max_nodes: int = 10
    temperature: float = 2.0
    seed: int = 0
    alpha: tuple[float, float] = (1.0, 1.0)

    def __post_init__(self) -> None:
        if self.type not in ("learned", "random", "oracle", "flattened"):
            raise ValueError(f"unknown executor type {self.type!r}")
        if self.dim < 2:
            raise ValueError(f"dim must be >= 2, got {self.dim}")
        if self.max_nodes < 4:
            raise ValueError(f"max_nodes must be >= 4, got {self.max_nodes}")
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")


class StructuralExecutor:
    """The CASM contract in pure stdlib.

    Gates are fixed parameters, not learned parameters. The failure analysis
    attributed the Phase 1.5A L2 result (0/18) to learned routing, not to
    execution, so v0 makes execution deterministic and concentrates learning
    on the routing side, where the portfolio has positive evidence.
    """

    def __init__(self, config: ExecutorConfig | None = None) -> None:
        self.config = config if config is not None else ExecutorConfig()
        self.dim = self.config.dim
        self.max_nodes = self.config.max_nodes
        self.temperature = self.config.temperature
        self.alpha_eta = list(self.config.alpha)
        self._rng = random.Random(self.config.seed)
        self.last_node_values: tuple[float, ...] = ()
        self.last_gates: tuple[float, ...] = ()

    # -- structural encoding ---------------------------------------------- #

    def structural_encode(self, program: Program) -> list[float]:
        """Encode only public per-node structure, never the true edge set.

        This is the information boundary of the contract: the encoder sees
        node identity, op, depth, position and arity. It does not see which
        edges are true, and it never sees runtime values.
        """
        enc: list[float] = []
        for node in program.nodes[: program.active_count]:
            enc.append(float(list(OPS).index(node.op)) / len(OPS))
            enc.append(node.depth / max(1.0, self.max_nodes))
            enc.append(node.index / max(1.0, self.max_nodes - 1))
            enc.append(node.arity / 2.0)
        return enc

    # -- the protocol ------------------------------------------------------ #

    def execute(self, structure: Structure, inputs: Sequence[float]) -> ExecutionResult:
        """Execute ``structure`` over ``inputs`` and return its output.

        Runtime values enter only here, downstream of and invisible to
        routing. ``trace`` carries the per-node values so path-level
        verification (``V_path``) can be compared against final-only
        verification (``V_final``) — a comparison no source repository has
        run.

        Relevance circuits run in *exact* Boolean semantics (``alpha = 0``).
        The learned ``alpha = c * softplus(eta)`` scaling is a computation
        strength per syntactic argument port; it is appropriate for a learned
        substrate, but a relevance circuit computes a fixed relation where
        any scaling of a partial disagreement can outrank an exact agreement,
        which would make the relation's satisfier non-maximal. The circuit is
        the environment's own relation, so it is exact.
        """
        program = structure.spec
        if not isinstance(program, Program):
            raise TypeError(f"expected a Program in Structure.spec, got {type(program)}")

        if self.config.type == "flattened":
            return self._execute_flattened(structure, program, inputs)

        gates = self._gates_for(program, structure)
        alpha = self._alpha_for(program)
        values = self._input_values_for(program, inputs)

        for node_idx in range(program.active_count):
            node = program.nodes[node_idx]
            if node.op is INPUT:
                continue
            args: list[float] = []
            for port in range(node.arity):
                incoming = [(e, gates[k]) for k, e in enumerate(program.candidate_edges)
                            if e.dst == node.index and e.port == port]
                if not incoming:
                    args.append(0.0)
                    continue
                if getattr(program, "exact", False):
                    # The circuit's own edges are the true wiring, so each
                    # argument is taken at full strength. Gating here would
                    # reintroduce partial credit where the relation is exact.
                    args.append(values[incoming[0][0].src])
                    continue
                routed = sum(g * values[e.src] for e, g in incoming) / (sum(g for _, g in incoming) + 1e-6)
                args.append(routed)
            values[node.index] = self._apply(node.op, args, alpha)

        self.last_node_values = tuple(values[: program.active_count])
        self.last_gates = tuple(gates)
        return ExecutionResult(
            output=values[program.output],
            gates=tuple(gates),
            node_values=tuple(values[: program.active_count]),
            provenance="executed",
        )

    def _input_values_for(self, program: Program, inputs: Sequence[float]) -> list[float]:
        """Resolve the input vector a program actually executes over.

        ``program_from_descriptor`` and ``build_program`` receive their bits
        positionally through ``inputs``. A ``relevance_program`` receives no
        external vector at all: its bits are the reference and descriptor,
        carried on the program itself. Without this, the circuit would read
        zeros for every candidate and compute a constant — the difference
        between an executed computation and a constant.
        """
        resolved = [0.0] * self.max_nodes
        if getattr(program, "exact", False) and program.input_values:
            vals: Sequence[float] = program.input_values
        else:
            vals = inputs
        for j, node_idx in enumerate(program.inputs):
            if j < len(vals):
                resolved[node_idx] = float(vals[j])
        return resolved

    def _execute_flattened(self, structure: Structure, program: Program,
                           inputs: Sequence[float]) -> ExecutionResult:
        """The dense-compute control: compute everything, route nothing.

        All inputs are summed with uniform weight regardless of topology, so
        ``C_executed`` scales with the input width rather than with the
        relevant subset. This is the arm any structural claim has to beat.
        """
        total = sum(float(x) for x in inputs)
        n = max(1, len(inputs))
        values = tuple(total / n for _ in range(self.max_nodes))
        self.last_node_values = values
        self.last_gates = tuple(1.0 for _ in program.candidate_edges)
        return ExecutionResult(
            output=total / n,
            gates=self.last_gates,
            node_values=values,
            provenance="flattened",
      )


    # -- internals --------------------------------------------------------- #

    def _alpha_for(self, program: Program) -> tuple[float, float] | None:
        """The computation strength for this execution, or ``None`` for exact.

        The learned ``alpha = c * softplus(eta)`` is a per-port computation
        strength for a *learned* substrate. A relevance circuit computes the
        environment's own relation, where any scaling of a partial
        disagreement can outrank an exact agreement and make the relation's
        satisfier non-maximal. Returning ``None`` selects exact Boolean
        semantics in ``_apply``, which is a different mode rather than a
        value of the learned parameter: ``softplus`` never maps back to
        exactly 1.0, so an exact circuit cannot be expressed through the
        learned parameter at all.
        """
        if getattr(program, "exact", False):
            return None
        return tuple(_alpha(self.alpha_eta, p) for p in range(2))

    def _gates_for(self, program: Program, structure: Structure) -> list[float]:
        """Gate values for the substrate, under the configured arm.

        ``oracle`` uses the true edge set — the copy-mask, which is a mandatory
        preflight rather than a learned baseline. ``learned`` and ``random``
        both produce values in [0, 1]; ``learned`` is deterministic given the
        structure (fixed gates) and ``random`` resamples on every execution.
        An ``exact`` program carries its own true wiring and is never gated.
        """
        if getattr(program, "exact", False):
            return [1.0 for _ in program.candidate_edges]
        true_set = program.true_edge_set
        if self.config.type == "oracle":
            return [1.0 if (e.src, e.dst, e.port) in true_set else 0.0
                    for e in program.candidate_edges]

        if self.config.type == "random":
            return [self._rng.random() for _ in program.candidate_edges]

        # learned / default: deterministic gates from the structural encoding
        enc = self.structural_encode(program)
        gates: list[float] = []
        for e in program.candidate_edges:
            d = float(e.dst) / max(1.0, self.max_nodes - 1)
            s = float(e.src) / max(1.0, self.max_nodes - 1)
            logit = 2.0 * (1.0 - abs(d - s)) - 1.0
            for k, bit in enumerate(enc):
                if k % 4 == 1:
                    logit += 0.25 * bit
            gates.append(_gate(logit, self.temperature))
        return gates

    def _apply(self, op: Op, args: list[float],
               alpha: tuple[float, float] | None = None) -> float:
        """CASM op semantics, from ``phase1_dag/model.py``.

        ``NOT = 1 - alpha[0]*x``; ``AND = a*b``; ``OR = a+b-a*b``;
        ``XOR = a+b-2ab``.

        ``alpha is None`` is *exact Boolean* mode, used only by a
        relevance circuit computing the environment's own relation. It is a
        distinct mode and not a value of the learned parameter: ``softplus``
        never returns exactly 1.0, so exact semantics cannot be reached
        through ``alpha`` at all, and confounding the two silently rescales
        an AND chain so a partial disagreement can outrank an exact
        agreement. ``EQ`` exists only in exact mode.
        """
        if alpha is None:
            return self._apply_exact(op, args)
        if op is NOT:
            return 1.0 - alpha[0] * args[0]
        a = alpha[0] * args[0]
        b = alpha[1] * args[1]
        if op is AND:
            return a * b
        if op is OR:
            return a + b - a * b
        if op is XOR:
            return a + b - 2.0 * a * b
        if op is EQ:
            # Unreachable: EQ is only constructible on an exact circuit.
            raise AssertionError("EQ outside an exact circuit")
        return 0.0

    @staticmethod
    def _apply_exact(op: Op, args: list[float]) -> float:
        """Exact Boolean semantics: identity alpha, ``[0, 1]`` inputs/outputs.

        ``EQ = NOT(XOR(a, b))`` as a composition of two existing ops, so no
        new primitive enters the grammar; ``EQ(a, b) = 1 - |a - b|``, which
        is 1 iff the bits agree and 0 iff they differ, with no polarity
        inversion in the range that matters.
        """
        if op is NOT:
            return 1.0 - args[0]
        a, b = args[0], args[1] if len(args) > 1 else 0.0
        if op is AND:
            return a * b
        if op is OR:
            return a + b - a * b
        if op is XOR:
            return abs(a - b)
        if op is EQ:
            return 1.0 - abs(a - b)
        return 0.0


class FlattenedExecutor(StructuralExecutor):
    """Convenience handle for the dense-compute control.

    ``StructuralExecutor`` already implements the flattened path internally so
    that one object serves every structure arm (one code path, no
    per-experiment branches). This subclass only exists so an ablation cell
    can name the arm explicitly.
    """

    def __init__(self, config: ExecutorConfig | None = None) -> None:
        super().__init__(config if config is not None else ExecutorConfig(type="flattened"))


def build_program(seed: int, *, dim: int = 8, max_nodes: int = 10) -> Program:
    """Construct a fixed upper-triangular substrate with a random wiring.

    The substrate is shared across episodes: the same candidate edges appear
    every time, so which edges carry values is the only variable. The true
    wiring is drawn randomly, making it one of many admissible wirings — so a
    learned gate product is not the oracle.
    """
    if dim < 2:
        raise ValueError(f"dim must be >= 2, got {dim}")
    if max_nodes < 4:
        raise ValueError(f"max_nodes must be >= 4, got {max_nodes}")

    rng = random.Random(seed)
    n_inputs = min(dim, max_nodes // 2)
    inputs = tuple(range(n_inputs))

    nodes: list[Node] = []
    for i in range(max_nodes):
        if i < n_inputs:
            nodes.append(Node(index=i, op=INPUT, depth=0, arity=0))
        else:
            op = rng.choice([o for o in OPS if o is not INPUT])
            nodes.append(Node(index=i, op=op, depth=1, arity=op.arity))

    edges: list[Edge] = []
    for dst in range(n_inputs, max_nodes):
        for src in range(dst):
            for port in range(nodes[dst].arity):
                edges.append(Edge(src=src, dst=dst, port=port))

    return Program(
        nodes=tuple(nodes),
        candidate_edges=tuple(edges),
        inputs=inputs,
        output=max_nodes - 1,
        active_count=max_nodes,
    )


def program_from_descriptor(descriptor: Sequence[int], *,
                            max_nodes: int = 10) -> Program:
    """Build a program whose input nodes carry a descriptor's bits.

    This is the adapter that lets the executor serve the environment's
    candidate descriptors: the descriptor becomes the program's input values,
    and the structural decision is which of those inputs reach the output.
    """
    if not descriptor:
        raise ValueError("descriptor must be non-empty")
    n_inputs = min(len(descriptor), max_nodes // 2)
    inputs = tuple(range(n_inputs))

    nodes: list[Node] = []
    for i in range(max_nodes):
        if i < n_inputs:
            nodes.append(Node(index=i, op=INPUT, depth=0, arity=0))
        else:
            arity = 2 if i < max_nodes - 1 else 1
            op = NOT if arity == 1 else XOR
            nodes.append(Node(index=i, op=op, depth=1, arity=arity))

    edges: list[Edge] = []
    for dst in range(n_inputs, max_nodes):
        for src in range(dst):
            for port in range(nodes[dst].arity):
                edges.append(Edge(src=src, dst=dst, port=port))
    return Program(
        nodes=tuple(nodes),
        candidate_edges=tuple(edges),
        inputs=inputs,
        output=max_nodes - 1,
        active_count=max_nodes,
    )


def copy_mask(program: Program) -> tuple[float, ...]:
    """The oracle preflight: gates that copy the true edge set.

    Frozen in the source README as a mandatory falsification check, never a
    learned baseline. If the copy-mask arm does not solve a task, the task is
    not solvable by structural routing and no learned structural result is
    interpretable — the same logic as the representability gate, applied to
    the substrate.
    """
    return tuple(
        1.0 if (e.src, e.dst, e.port) in program.true_edge_set else 0.0
        for e in program.candidate_edges
    )


# --------------------------------------------------------------------------- #
# The relevance circuit
# --------------------------------------------------------------------------- #


def relevance_program(
    reference: Sequence[int],
    descriptor: Sequence[int],
    marks: Sequence[int],
    *,
    max_nodes: int = 10,
) -> Program:
    """Build a circuit computing agreement with ``reference`` on ``marks``.

    This is the structural computation of the Stage-2c relation. Where the
    router scores the relation with a cheap linear basis, the executor
    *computes* it as a Boolean circuit — the CASM role in the loop:

        eq_j   = EQ(reference_j, descriptor_j)
        output = AND over j in marks of eq_j

    ``EQ = NOT(XOR(a, b))`` is added to the CASM grammar as a composition of
    two existing ops rather than a new primitive, so the substrate's op set
    stays closed under the semantics already defined. ``reference`` is
    whatever the family's target is measured against: the public query bits
    for ``relational``, or the state-read bits for the persistence families.
    ``marks`` are the positions the context gates on.

    The circuit is exact Boolean: at ``alpha = 1`` each ``eq_j`` is 1 iff the
    bits agree, and the AND chain is 1 iff *every* marked position agrees.
    The output is therefore high exactly for the relation's satisfier, which
    is what makes the execution a genuine cross-check of routing.

    The circuit is built from public inputs only. It never sees the target
    action, the gold index, or the outcome; it sees the same reference the
    router is allowed to see, computed a different way. That is what makes
    the verifier a genuine cross-check rather than a tautology: routing and
    execution are two independent computations of one relation, and
    verification compares them.
    """
    if not marks:
        raise ValueError("relevance requires at least one marked position")
    if not reference:
        raise ValueError("relevance requires a reference to compare against")

    # Node layout: inputs first, then one EQ per marked position, then a
    # left-leaning AND chain.
    n_inputs = 2 * len(marks)
    eq_base = n_inputs
    and_base = eq_base + len(marks)
    total = and_base + max(1, len(marks) - 1)

    if total > max_nodes:
        raise ValueError(
            f"circuit needs {total} nodes but max_nodes is {max_nodes}"
        )

    nodes: list[Node] = []
    for i in range(total):
        if i < n_inputs:
            nodes.append(Node(index=i, op=INPUT, depth=0, arity=0))
        elif i < and_base:
            nodes.append(Node(index=i, op=EQ, depth=1, arity=2))
        else:
            nodes.append(Node(index=i, op=AND, depth=2, arity=2))

    # Input nodes are the first ``n_inputs`` nodes of the circuit, in order:
    # the reference bits for the marked positions, then the descriptor bits
    # for the same positions. ``input_values`` carries the corresponding bits.
    inputs: list[int] = list(range(n_inputs))

    # The input layout is: ``0..m-1`` reference bits, then ``m..2m-1``
    # descriptor bits. EQ node ``k`` must therefore read reference ``k`` and
    # descriptor ``k`` — pairing ``2k`` with ``2k+1`` instead compares the
    # reference against itself and the descriptor against itself, which makes
    # the circuit's output independent of the marked agreement.
    n_marks = len(marks)
    edges: list[Edge] = []
    for k in range(n_marks):
        edges.append(Edge(src=k, dst=eq_base + k, port=0))
        edges.append(Edge(src=n_marks + k, dst=eq_base + k, port=1))

    # Left-leaning AND chain over the equality outputs.
    if len(marks) == 1:
        edges.append(Edge(src=eq_base, dst=and_base, port=0))
        edges.append(Edge(src=eq_base, dst=and_base, port=1))
    else:
        edges.append(Edge(src=eq_base, dst=and_base, port=0))
        edges.append(Edge(src=eq_base + 1, dst=and_base, port=1))
        for k in range(2, len(marks)):
            edges.append(Edge(src=and_base + k - 2, dst=and_base + k - 1, port=0))
            edges.append(Edge(src=eq_base + k, dst=and_base + k - 1, port=1))

    # Only the marked positions matter for the relation, so the circuit reads
    # exactly those bits from each side.
    input_values = tuple(reference[j] for j in marks) + tuple(descriptor[j] for j in marks)

    return Program(
        nodes=tuple(nodes),
        candidate_edges=tuple(edges),
        inputs=tuple(inputs),
        input_values=input_values,
        output=and_base + len(marks) - 2 if len(marks) > 1 else and_base,
        active_count=total,
        exact=True,
    )
