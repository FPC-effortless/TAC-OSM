"""``R_t`` — relevance routing.

## The arms, deliberately all present

The architecture document (§4.2) requires the learned router and the teacher
to coexist, and the distinction is the scientific point of this module:

* **``LearnedRelationalRouter``** — the runtime path. Adapted from
  ``stage2c_relational.py::RelationalBanditRouter`` (``42f9814``), the
  corrected key-gated form after ``f989430``. This is the strongest routing
  evidence in the portfolio (E2, 25/25 per-seed deltas positive), and its
  decisive property is that it is *cheap*: scoring one candidate costs a dot
  product, not an LM forward pass.
* **``CDLTeacher``** — a *re-declaration*, not an import. The real CDL
  computation requires ``transformers`` and a SmolLM2 forward pass per
  candidate at ``O(N · C_LM)``, which is structurally incompatible with the
  thesis this repository exists to test. It enters as a definition and a
  quality ceiling, never as runtime code. See ``provenance/COMPONENTS.md``.
* **``StaticTotalAgreementRouter``** — the control that matters. Stage 2b
  showed a fixed similarity scorer captures ~71% of available headroom for
  free. A routing result that does not beat static is not a routing result,
  so static is what every learned arm is measured against.
* **``OracleRouter``** — the ceiling. In the integrated setting the oracle
  reads ``Task.target_action``, which the real router cannot see; it exists
  to measure how much headroom outcome training leaves on the table.
* **``FullContextRouter``** — the dense baseline: ignore state entirely and
  route on total query agreement. This is the arm that computes
  ``C_executed ≈ f(|H|)`` instead of ``f(|R|)``.
* **``RandomRouter``** — chance.

## The persistence extension, and why its block is gated

The source router scores candidates against the *query*. The integrated loop
also has to route over candidates whose target lives in persistent state, so
the basis is extended with one keyed-agreement block per state slot: for each
slot ``s`` the store returns on this read, and each descriptor position ``j``,
the feature is

    slot_s_j = present(s) * [value(s)_j == descriptor_j]

This is what allows one router basis to serve the ``relational``,
``state_lookup``, and ``replay`` families without a per-family code path.

**But the un-gated slot block is not a sufficient basis.** The relevance
relation defines gold as *agreeing* with the reference on the marked
positions and *anti-matching* it elsewhere (prob ``1 - noise``), so on the
unmarked positions gold carries an anti-signal that is nearly deterministic
(measured agree-rate 0.04–0.06). An un-gated block sees that anti-signal and
inverts the score. Measured over 60 episodes, no single shared weight vector
separates gold from the best distractor for ``state_lookup`` (min margin
−0.2353) or ``replay`` (−2.3210), while ``relational`` separates easily
(+1.4754) — because the Stage-2c ``gated_agreement`` block zeroes every
unmarked position and the slot block does not. The gap is in the hypothesis
class, not the optimiser, so it is found by the §34 audit rather than by
training.

The fix is the exact structural analogue of the gated block with the state
read substituted for the query:

    slot_gated_s_j = present(s) * context_j * [value(s)_j == descriptor_j]

Measured solo it separates all three families at margin +2.0. Set as a fixed
analytic vector — ``gated_agreement = 1.0``, ``slot_gated = 2.0``, everything
else 0 — it separates all three families over 300 episodes each with **zero
ties and zero violations**. The ``replay`` family needs the ratio: it carries
public bits *and* an address, so both blocks are active, and the slot block
must dominate or the two cancel to an exact tie (186/300 episodes at equal
weights). ``slot_gated`` is therefore not a tuned constant but a required
asymmetry, and the gate is what restores a shared-weight solution where none
existed.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

from . import Candidate, PersistentState, Query, RoutingDecision, StateRead
from .state import Slot

__all__ = [
    "RouterConfig",
    "LearnedRelationalRouter",
    "StaticTotalAgreementRouter",
    "RandomRouter",
    "OracleRouter",
    "FullContextRouter",
    "CDLTeacher",
    "build_router",
    "basis_size",
    "features",
    "analytic_weights",
    "arm_types",
]

ARM_TYPES = ("learned", "static", "random", "oracle", "full_context")


def arm_types() -> tuple[str, ...]:
    """The router arms, matching ``ablation.RouterType``."""
    return ARM_TYPES


@dataclass
class RouterConfig:
    """Switch surface for the router, mirroring ``ablation.RouterSwitch``."""

    type: str = "learned"
    dim: int = 8
    lr: float = 0.05
    temperature: float = 0.5
    seed: int = 0
    max_state_slots: int = 4

    def __post_init__(self) -> None:
        if self.type not in ARM_TYPES:
            raise ValueError(
                f"unknown router type {self.type!r}; expected one of {ARM_TYPES}"
            )
        if self.dim < 4:
            raise ValueError(f"dim must be >= 4, got {self.dim}")
        if not 0.0 < self.lr:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if self.max_state_slots < 0:
            raise ValueError(f"max_state_slots must be >= 0, got {self.max_state_slots}")


# --------------------------------------------------------------------------- #
# Feature basis
# --------------------------------------------------------------------------- #


def basis_size(dim: int, max_slots: int) -> int:
    """Number of features produced by :func:`features`.

    Exposed so a caller can size a weight vector without computing features.

    ``1 + 4*dim`` is the Stage-2c basis (bias plus four agreement blocks).
    ``dim*max_slots`` is the un-gated slot block, retained because the audit
    that found the gap measures the same basis the router scores over, and
    ``dim*max_slots`` more is the gated slot block that restores
    representability.
    """
    return 1 + 4 * dim + 2 * dim * max_slots


def _bits(text: str) -> tuple[int, ...]:
    """Public query bits from a query's text, ignoring any state address.

    ``Query.text`` is ``"<bits>\t<address>"``; only the bits are parseable,
    and the address is not a signal for the feature basis.
    """
    part = text.partition("\t")[0]
    return tuple(int(b) for b in part.split()) if part.strip() else ()


def _agree(a: Sequence[int], i: int, b: Sequence[int], j: int) -> float:
    """Agreement feature, neutral (0.0) where either position is absent.

    A 0.0 is neither agreement nor disagreement, so a family that does not
    supply an input contributes no signal rather than contributing noise.
    """
    if i >= len(a) or j >= len(b):
        return 0.0
    return 1.0 if a[i] == b[j] else -1.0


def _pad(slots: Sequence[tuple[int, ...]], max_slots: int) -> list[tuple[int, ...]]:
    """Pad the state rows to a fixed width with empty rows.

    A fixed width is what makes the basis size independent of the store's
    occupancy, so the learned weights have a stable address space across
    reads that return different numbers of rows.
    """
    out = list(slots[:max_slots])
    while len(out) < max_slots:
        out.append(())
    return out


def features(
    query: Query,
    descriptor: Sequence[int],
    read: StateRead,
    dim: int,
    max_slots: int,
) -> list[float]:
    """The feature basis the learned router scores over.

    The first four blocks are the Stage-2c basis, adapted verbatim:

        bias
        q_j == c_j          (total query agreement — the static rule)
        x_j == c_j          (context-descriptor agreement)
        x_j == q_j          (mark/query agreement)
        x_j * (q_j == c_j)  (context-**gated** query agreement)

    The fifth is the persistence extension documented at the top of this
    module: keyed agreement against the state read, one block per slot.

    The state read is the *only* channel by which persistent state reaches
    routing. ``query`` supplies the public bits and the address; neither is
    the answer.
    """
    query_bits = _bits(query.text)
    context = tuple(query.context)

    # A family with no public bits (``state_lookup``) supplies an empty
    # ``query_bits``; a family with no state (``relational``) supplies an
    # empty read. Each block is neutral where its input is absent, so one
    # basis serves all three families without a per-family branch. Neutrality
    # is 0.0 for the gated block (the feature is switched off) and 0.0 for
    # the agreement blocks (neither agreement nor disagreement is asserted).
    feats: list[float] = [1.0]
    for j in range(dim):
        feats.append(_agree(query_bits, j, descriptor, j))
    for j in range(dim):
        feats.append(_agree(context, j, descriptor, j))
    for j in range(dim):
        feats.append(_agree(context, j, query_bits, j))
    for j in range(dim):
        gate = float(context[j]) if j < len(context) else 0.0
        feats.append(gate * _agree(query_bits, j, descriptor, j))

    # The state read supplies plain bit rows (``StateRead.values``); empty
    # rows mark absent slots. ``Slot`` never crosses the interface, so this
    # basis works against any store implementation.
    #
    # Two blocks, not one, and they are **contiguous**: every slot's
    # ``slot_agreement`` positions first, then every slot's ``slot_gated``
    # positions. ``_block_offsets`` addresses them by block, so an interleaved
    # layout would put ``analytic_weights`` in the wrong coordinates and the
    # gate would measure a different vector than the one it names.
    #
    # The un-gated block is the direct extension; the gated block is the one
    # that restores a shared-weight solution, because the un-gated block
    # carries the anti-signal on the unmarked positions. Both are gated by
    # slot presence; the second is gated by the mark vector too, exactly as
    # the Stage-2c gated block is.
    #
    # **Addressed slot only.** ``read`` places the addressed slot first, then
    # the rest of the occupied pool, and the pool *grows* over an episode —
    # after 40 tasks the read returns 40 vectors. A non-addressed slot's value
    # has no relation to this task, but its agreement features are still
    # signed, so it injects noise that grows with history and dominates the
    # signal. ``_addressed`` is 1.0 only for the slot the query names, so a
    # weight vector spanning multiple slots cannot pick up pool noise.
    address = _query_address(query)
    rows = _pad(read.values, max_slots)
    addressed = [
        (1.0 if (address and i < len(read.keys) and read.keys[i] == address) else 0.0)
        for i in range(len(rows))
    ]
    for i, row in enumerate(rows):
        present = addressed[i] if row else 0.0
        for j in range(dim):
            if not row:
                feats.append(0.0)
            else:
                feats.append(present * (1.0 if row[j] == descriptor[j] else -1.0))
    for i, row in enumerate(rows):
        present = addressed[i] if row else 0.0
        for j in range(dim):
            if not row:
                feats.append(0.0)
            else:
                gate = float(context[j]) if j < len(context) else 0.0
                feats.append(present * gate * (1.0 if row[j] == descriptor[j] else -1.0))
    return feats


def _query_address(query: Query) -> str:
    """The state address this query names, or '' when the task needs no state."""
    return query.text.partition("\t")[2]


# --------------------------------------------------------------------------- #
# The learned router
# --------------------------------------------------------------------------- #


class LearnedRelationalRouter:
    """Outcome-trained linear scorer over the Stage-2c basis.

    Adapted from ``stage2c_relational.py::RelationalBanditRouter``, with the
    persistence extension. Deliberately kept as a small linear model over a
    hand-designed basis so that the loop isolates "can relevance be learned
    from outcomes" rather than "is this model class expressive": setting every
    context-gated weight to +1 recovers the relation exactly, so the correct
    solution lives in the hypothesis class.

    Nothing here is given the gold index, the target action, the relation, or
    the answer. It sees the query, the candidates, and one state read.

    Training is REINFORCE on a scalar outcome, exactly as in the source: the
    gradient is ``lr * (reward - chance) * (1 - p_selected) * features``. The
    centring term is what keeps the update signed by *surprise* rather than by
    raw success, which matters because the environment's success rate is not
    0.5.
    """

    def __init__(self, config: RouterConfig | None = None) -> None:
        self.config = config if config is not None else RouterConfig()
        self.dim = self.config.dim
        self.max_state_slots = self.config.max_state_slots
        self.lr = self.config.lr
        self.temperature = self.config.temperature
        self.n = basis_size(self.dim, self.max_state_slots)
        self.w: list[float] = [0.0] * self.n
        self._rng = random.Random(self.config.seed)
        self._updates = 0

    # -- scoring ----------------------------------------------------------- #

    def _score_rows(self, query: Query, candidates: Sequence[Candidate],
                    read: StateRead) -> list[list[float]]:
        return [features(query, c.descriptor, read, self.dim, self.max_state_slots)
                for c in candidates]

    def _softmax(self, scores: Sequence[float]) -> list[float]:
        scaled = [s / max(self.temperature, 1e-6) for s in scores]
        m = max(scaled)
        exps = [math.exp(s - m) for s in scaled]
        z = sum(exps)
        return [x / z for x in exps] if z > 0 else [1.0 / len(scaled)] * len(scaled)

    # -- the protocol ------------------------------------------------------ #

    def route(self, query: Query, state: PersistentState,
              candidates: Sequence[Candidate]) -> RoutingDecision:
        """Select one candidate. Samples at the configured temperature."""
        read = state.read(query)
        rows = self._score_rows(query, candidates, read)
        scores = [sum(w * f for w, f in zip(self.w, row)) for row in rows]
        probs = self._softmax(scores)

        u = self._rng.random()
        acc = 0.0
        selected = len(probs) - 1
        for i, p in enumerate(probs):
            acc += p
            if u <= acc:
                selected = i
                break
        return RoutingDecision(selected=selected, scores=tuple(probs), provenance="learned")

    # -- learning ---------------------------------------------------------- #

    def update(self, query: Query, candidates: Sequence[Candidate],
               selected: int, reward: float, probs: Sequence[float],
               state: PersistentState) -> None:
        """REINFORCE on a scalar outcome. The gold index is never consulted."""
        read = state.read(query)
        row = features(query, candidates[selected].descriptor, read,
                       self.dim, self.max_state_slots)
        chance = 1.0 / max(1, len(candidates))
        centred = reward - chance
        grad = self.lr * centred * (1.0 - probs[selected])
        for k in range(self.n):
            self.w[k] += grad * row[k]
        self._updates += 1

    # -- diagnostics ------------------------------------------------------- #

    def weight_signature(self) -> dict[str, float]:
        """Block-wise weight means, so a learned rule can be read off.

        The Stage-2c result reported gated-agreement weights positive
        (1.29-1.47) and total-agreement weights negative (-1.25 to -1.72),
        with bias +4.00. That signature *is* the learned rule, and it is how
        a learned arm is distinguished from one that has merely memorised the
        candidate order.
        """
        blocks = _block_offsets(self.dim, self.max_state_slots)
        out: dict[str, float] = {}
        for name, (start, end) in blocks.items():
            block = self.w[start:end]
            out[name] = sum(block) / max(1, len(block))
        out["bias"] = self.w[0]
        return out

    @property
    def updates(self) -> int:
        return self._updates


def _block_offsets(dim: int, max_slots: int) -> dict[str, tuple[int, int]]:
    """Index ranges of each semantic block within a feature row."""
    blocks: dict[str, tuple[int, int]] = {}
    start = 1
    blocks["query_agreement"] = (start, start + dim); start += dim
    blocks["context_descriptor"] = (start, start + dim); start += dim
    blocks["context_query"] = (start, start + dim); start += dim
    blocks["gated_agreement"] = (start, start + dim); start += dim
    width = dim * max_slots
    blocks["slot_agreement"] = (start, start + width); start += width
    blocks["slot_gated"] = (start, start + width)
    return blocks


def analytic_weights(dim: int, max_slots: int) -> list[float]:
    """The fixed vector that separates gold on all three families.

    ``gated_agreement = 1.0`` and ``slot_gated = 2.0``, everything else zero.

    This is *not* a tuned hyperparameter and it is not a fitted value. It is
    the §34 evidence that the relation lives in the hypothesis class: one
    weight vector, shared across every episode, separates gold from every
    distractor on ``relational``, ``state_lookup``, and ``replay`` at margins
    +2.0, +4.0, and +2.0 with zero ties. ``slot_gated`` must exceed
    ``gated_agreement`` because ``replay`` activates both blocks at once and
    they otherwise cancel to an exact tie (186/300 episodes at equal weights);
    at 2:1 the tie vanishes entirely.

    Used by the representability gate and as the reference point a learned
    router's own block means are compared against.
    """
    n = basis_size(dim, max_slots)
    blocks = _block_offsets(dim, max_slots)
    w = [0.0] * n
    s, e = blocks["gated_agreement"]
    for j in range(s, e):
        w[j] = 1.0
    s, e = blocks["slot_gated"]
    for j in range(s, e):
        w[j] = 2.0
    return w


# --------------------------------------------------------------------------- #
# The control arms
# --------------------------------------------------------------------------- #


class StaticTotalAgreementRouter:
    """The fixed total-agreement rule from Stage 2b, unchanged.

    The key control. In Stage 2c the gold candidate deliberately anti-matches
    the query on the unmarked positions, so this rule is actively
    anti-correlated with relevance there — it scores ~0 where the learned
    router scores ~0.89. Carried over verbatim so that the comparison is
    against the *same* rule the portfolio has always used, not a modern
    rewrite of it.
    """

    def __init__(self, config: RouterConfig | None = None) -> None:
        self.config = config if config is not None else RouterConfig(type="static")
        self.dim = self.config.dim
        self._rng = random.Random(self.config.seed)

    def route(self, query: Query, state: PersistentState,
              candidates: Sequence[Candidate]) -> RoutingDecision:
        bits = _bits(query.text)
        scores: list[float] = []
        for c in candidates:
            agree = 0.0
            for j in range(min(self.dim, len(bits), len(c.descriptor))):
                if bits[j] == c.descriptor[j]:
                    agree += 1.0
            scores.append(agree)
        best = max(scores)
        tied = [i for i, s in enumerate(scores) if s == best]
        selected = tied[0] if len(tied) == 1 else self._rng.choice(tied)
        return RoutingDecision(selected=selected, scores=tuple(scores),
                               provenance="static")


class FullContextRouter:
    """The dense baseline: route on total agreement, ignoring state entirely.

    This is the arm that computes ``C_executed ≈ f(|H|)`` rather than
    ``f(|R|)``. It is the same rule as ``StaticTotalAgreementRouter``; the two
    are separate classes because they answer different questions in the
    ablation — static is a *routing* control (can a fixed rule do this?) and
    full-context is a *system* control (does persistence contribute anything
    a dense scorer cannot?). Keeping them apart stops a result for one from
    being read as a result for the other.
    """

    def __init__(self, config: RouterConfig | None = None) -> None:
        self.config = config if config is not None else RouterConfig(type="full_context")
        self.dim = self.config.dim
        self._rng = random.Random(self.config.seed)

    def route(self, query: Query, state: PersistentState,
              candidates: Sequence[Candidate]) -> RoutingDecision:
        bits = _bits(query.text)
        scores: list[float] = []
        for c in candidates:
            agree = 0.0
            for j in range(min(self.dim, len(bits), len(c.descriptor))):
                if bits[j] == c.descriptor[j]:
                    agree += 1.0
            scores.append(agree)
        best = max(scores)
        tied = [i for i, s in enumerate(scores) if s == best]
        selected = tied[0] if len(tied) == 1 else self._rng.choice(tied)
        return RoutingDecision(selected=selected, scores=tuple(scores),
                               provenance="full_context")


class RandomRouter:
    """Chance. Uniform over candidates, seeded for reproducibility."""

    def __init__(self, config: RouterConfig | None = None) -> None:
        self.config = config if config is not None else RouterConfig(type="random")
        self._rng = random.Random(self.config.seed)

    def route(self, query: Query, state: PersistentState,
              candidates: Sequence[Candidate]) -> RoutingDecision:
        selected = self._rng.randrange(len(candidates))
        chance = 1.0 / len(candidates)
        return RoutingDecision(selected=selected,
                               scores=tuple(chance for _ in candidates),
                               provenance="random")


class OracleRouter:
    """The ceiling: select the target the environment holds.

    In the integrated setting the oracle reads ``Task.target_action``, which
    no real router can see. It exists to measure how much headroom outcome
    training leaves on the table, and to run the copy-mask-style falsification
    preflight: if the oracle arm does not reach 1.0 the environment is
    ambiguous and no learned-arm result is interpretable.

    Bound to a task via :meth:`bind`, so the loop can hand it the task without
    exposing the target to anything else.
    """

    def __init__(self, config: RouterConfig | None = None) -> None:
        self.config = config if config is not None else RouterConfig(type="oracle")
        self._target: int | None = None

    def bind(self, target_action: int) -> None:
        self._target = target_action

    def route(self, query: Query, state: PersistentState,
              candidates: Sequence[Candidate]) -> RoutingDecision:
        if self._target is None:
            raise RuntimeError("OracleRouter is unbound; call bind(target_action)")
        target = self._target
        self._target = None
        scores = tuple(1.0 if i == target else 0.0 for i in range(len(candidates)))
        return RoutingDecision(selected=target, scores=scores, provenance="oracle")


class CDLTeacher:
    """The CDL relevance definition, re-declared rather than imported.

    The real computation is ``CDL(M, Q) = -NLL(Q | M)`` per candidate, requiring
    a SmolLM2 forward pass and ``transformers`` (source: ``app.py`` at
    ``c315544``). That is ``O(N · C_LM)`` — scoring every candidate with a full
    LM pass — which is structurally incompatible with the thesis that executed
    computation scales with the relevant subset rather than total history.

    So the teacher enters TAC-OSM as a *definition and a cost model*, never as
    runtime code. This class is the standing record of that exclusion: it is
    the only router whose ``route`` raises, because calling it at runtime would
    reintroduce the cost the whole architecture exists to avoid.

    For supervision, a cheap teacher surrogate is available in
    ``LearnedRelationalRouter`` trained to convergence on outcome reward — the
    same role CDL played in Stage B, without the per-candidate LM pass.
    """

    COST_PER_CANDIDATE = "one full LM forward pass"

    def __init__(self, config: RouterConfig | None = None) -> None:
        self.config = config if config is not None else RouterConfig(type="learned")

    def route(self, query: Query, state: PersistentState,
              candidates: Sequence[Candidate]) -> RoutingDecision:
        raise NotImplementedError(
            "CDLTeacher is a teacher definition and cost model, not a runtime "
            "router: scoring N candidates requires N full LM forward passes "
            "(O(N * C_LM)), which is structurally incompatible with the "
            "C_executed ~ f(|R|) thesis. Use LearnedRelationalRouter at runtime."
        )


def build_router(config: RouterConfig) -> LearnedRelationalRouter | object:
    """Construct the router for a config. One construction path for all arms.

    This is the single point at which an ablation switch becomes an object, so
    there is no per-experiment construction code anywhere else in the loop.
    """
    arm = config.type
    if arm == "learned":
        return LearnedRelationalRouter(config)
    if arm == "static":
        return StaticTotalAgreementRouter(config)
    if arm == "random":
        return RandomRouter(config)
    if arm == "oracle":
        return OracleRouter(config)
    if arm == "full_context":
        return FullContextRouter(config)
    raise ValueError(f"unreachable router arm: {arm!r}")
