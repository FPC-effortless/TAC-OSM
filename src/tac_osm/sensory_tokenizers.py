"""Seven *untrained* reversible sensory tokenizer mechanisms (SENSE-TOK-001).

These are deterministic mechanism prototypes, NOT trained tokenizers, CASM-S,
CDL teacher implementations, or evidence of intelligence. They consume only
observed bytes and, for query-conditioned arms, a supplied predecision query.
All byte spans are half-open; no target, reward, hidden graph or label enters.

The patch ledger is lossless. Additional hierarchy/graph metadata is counted
separately; do not equate fewer patches with fewer bytes stored or lower FLOPs.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from hashlib import sha256
from math import log2
from typing import Literal

Variant = Literal["fixed", "entropy", "sensory", "hierarchy", "relevance", "structure", "hybrid"]
VARIANTS: tuple[Variant, ...] = ("fixed", "entropy", "sensory", "hierarchy", "relevance", "structure", "hybrid")


@dataclass(frozen=True)
class Patch:
    start: int
    end: int
    payload: bytes
    salience: float = 0.0

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start or len(self.payload) != self.end - self.start:
            raise ValueError("invalid patch offset/payload")


@dataclass(frozen=True)
class Relation:
    src: int
    dst: int
    kind: str


@dataclass(frozen=True)
class Tokenization:
    variant: Variant
    length: int
    source_sha256: str
    patches: tuple[Patch, ...]
    leaves: tuple[Patch, ...] = ()  # optional lower level; NOT free computation
    edges: tuple[Relation, ...] = ()
    scan_work: int = 0  # count of byte/patch visits, NOT measured FLOPs
    query_sha256: str | None = None

    def reconstruct(self) -> bytes:
        data = b"".join(p.payload for p in self.patches)
        if len(data) != self.length or sha256(data).hexdigest() != self.source_sha256:
            raise ValueError("lossless reconstruction invariant violated")
        return data

    @property
    def total_representation_nodes(self) -> int:
        return len(self.patches) + len(self.leaves) + len(self.edges)

    def validate(self) -> None:
        if self.length == 0:
            if self.patches or self.leaves or self.edges:
                raise ValueError("empty source must have empty representation")
            self.reconstruct()
            return
        self._validate_partition(self.patches, self.length)
        if self.leaves:
            self._validate_partition(self.leaves, self.length)
        for edge in self.edges:
            if not (0 <= edge.src < edge.dst < len(self.patches)):
                raise ValueError("edge must be forward, in-range, and non-oracle")
            if edge.kind not in ("next", "repeat", "same_class"):
                raise ValueError("unknown relation type")
        self.reconstruct()

    @staticmethod
    def _validate_partition(patches: tuple[Patch, ...], length: int) -> None:
        pos = 0
        for patch in patches:
            if patch.start != pos:
                raise ValueError("noncontiguous partition")
            pos = patch.end
        if pos != length:
            raise ValueError("partition does not cover source")


def _classes(b: int) -> int:
    if 65 <= b <= 90 or 97 <= b <= 122:
        return 1
    if 48 <= b <= 57:
        return 2
    if b in (9, 10, 13, 32):
        return 3
    if b >= 128:
        return 5
    return 4


def _patches(data: bytes, ends: list[int], scores: list[float] | None = None) -> tuple[Patch, ...]:
    if not data:
        return ()
    if not ends or ends[-1] != len(data) or ends != sorted(set(ends)):
        raise ValueError("boundaries must be unique, ascending, and cover input")
    pos = 0
    result: list[Patch] = []
    for k, end in enumerate(ends):
        result.append(Patch(pos, end, data[pos:end], float(scores[k]) if scores else 0.0))
        pos = end
    return tuple(result)


def _fixed(data: bytes, size: int) -> tuple[Patch, ...]:
    return _patches(data, list(range(size, len(data), size)) + ([len(data)] if data else []))


def _entropy(data: bytes, minimum: int, maximum: int, threshold: float = 6.4, window: int = 32) -> tuple[Patch, ...]:
    history: deque[int] = deque()
    counts: Counter[int] = Counter()
    ends: list[int] = []
    scores: list[float] = []
    start = 0
    for i, b in enumerate(data):
        # P(byte | preceding observed bytes), Laplace smoothed; no lookahead.
        surprise = -log2((counts[b] + 1) / (len(history) + 256))
        history.append(b)
        counts[b] += 1
        if len(history) > window:
            old = history.popleft()
            counts[old] -= 1
            if counts[old] == 0:
                del counts[old]
        current_size = i + 1 - start
        if current_size >= maximum or (current_size >= minimum and surprise >= threshold):
            ends.append(i + 1)
            scores.append(surprise)
            start = i + 1
    if start < len(data):
        ends.append(len(data))
        scores.append(0.0)
    return _patches(data, ends, scores)


def _sensory(data: bytes, minimum: int, maximum: int) -> tuple[Patch, ...]:
    # Fly-inspired only as an analogy: local class contrast + repetition novelty.
    ends: list[int] = []
    scores: list[float] = []
    start = 0
    for i, b in enumerate(data):
        last = data[i - 1] if i else b
        class_jump = int(_classes(last) != _classes(b))
        local_change = int(b != last and i >= 2 and last == data[i - 2])
        salience = class_jump * 2 + local_change
        length = i + 1 - start
        if length >= maximum or (length >= minimum and salience >= 2):
            ends.append(i + 1)
            scores.append(float(salience))
            start = i + 1
    if start < len(data):
        ends.append(len(data))
        scores.append(0.0)
    return _patches(data, ends, scores)


def _hierarchy(data: bytes, leaves: tuple[Patch, ...], maximum: int) -> tuple[Patch, ...]:
    if not leaves:
        return ()
    ends: list[int] = []
    total = 0
    for leaf in leaves:
        # Oversized atomic leaves form their own top-level patch.
        if total and total + len(leaf.payload) > maximum:
            ends.append(leaf.start)
            total = 0
        total += len(leaf.payload)
    ends.append(len(data))
    return _patches(data, ends)


def _query_grams(query: bytes) -> set[bytes]:
    lower = query.lower()
    return {lower[i:i + 2] for i in range(max(0, len(lower) - 1))}


def _relevance(data: bytes, base: tuple[Patch, ...], query: bytes, maximum: int) -> tuple[Patch, ...]:
    if not base:
        return ()
    qgrams = _query_grams(query)
    # Deliberately simple visible-feature approximation, NOT CDL inference.
    # Query is fixed prior to processing. Offline relevance may resegment
    # existing units and therefore must NOT be used as a causal LM tokenizer.
    ends: list[int] = []
    scores: list[float] = []
    pending = bytearray()
    pos = 0
    def flush() -> None:
        nonlocal pos
        if pending:
            pos += len(pending)
            ends.append(pos)
            scores.append(0.0)
            pending.clear()

    for leaf in base:
        s = leaf.payload.lower()
        grams = {s[i:i + 2] for i in range(max(0, len(s) - 1))}
        relevant = bool(grams & qgrams) if qgrams else bool(s and s in query.lower())
        if relevant:
            flush()
            for i in range(0, len(leaf.payload), 2):
                end = pos + min(i + 2, len(leaf.payload))
                ends.append(end)
                scores.append(1.0)
            pos += len(leaf.payload)
        else:
            if pending and len(pending) + len(leaf.payload) > maximum:
                flush()
            if len(leaf.payload) > maximum:
                flush()
                for i in range(0, len(leaf.payload), maximum):
                    end = pos + min(i + maximum, len(leaf.payload))
                    ends.append(end)
                    scores.append(0.0)
                pos += len(leaf.payload)
            else:
                pending.extend(leaf.payload)
    flush()
    return _patches(data, ends, scores)


def _relations(patches: tuple[Patch, ...]) -> tuple[Relation, ...]:
    relations: list[Relation] = []
    seen: dict[bytes, int] = {}
    classes: dict[int, int] = {}
    for i, patch in enumerate(patches):
        if i > 0:
            relations.append(Relation(i - 1, i, "next"))
        prior = seen.get(patch.payload)
        if prior is not None and prior != i - 1:
            relations.append(Relation(prior, i, "repeat"))
        seen[patch.payload] = i
        kind = _classes(patch.payload[0])
        prior_class = classes.get(kind)
        if prior_class is not None and prior_class != i - 1 and prior_class != prior:
            relations.append(Relation(prior_class, i, "same_class"))
        classes[kind] = i
    return tuple(relations)


def tokenize(
    data: bytes,
    *,
    variant: Variant,
    query: bytes | None = None,
    min_patch: int = 2,
    max_patch: int = 24,
    fixed_size: int = 8,
) -> Tokenization:
    """Pure, deterministic, byte-preserving mechanism prototype.

    Input is raw observed bytes. For relevance/hybrid, query must be a separate
    predecision input; no evaluator gold/target is accepted by this API.
    """
    if not isinstance(data, bytes) or (query is not None and not isinstance(query, bytes)):
        raise TypeError("data and optional query must be bytes")
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant: {variant}")
    if not (1 <= min_patch <= max_patch) or fixed_size < 1:
        raise ValueError("invalid patch sizes")
    if variant in ("relevance", "hybrid") and query is None:
        raise ValueError("query required for query-conditioned variant")
    if variant not in ("relevance", "hybrid") and query is not None:
        raise ValueError("unexpected query: must not introduce extra information")
    leaves: tuple[Patch, ...] = ()
    edges: tuple[Relation, ...] = ()
    query_bytes = query or b""
    visits = len(data)
    if variant == "fixed":
        patches = _fixed(data, fixed_size)
    elif variant == "entropy":
        patches = _entropy(data, min_patch, max_patch)
    elif variant == "sensory":
        patches = _sensory(data, min_patch, max_patch)
    elif variant == "hierarchy":
        leaves = _sensory(data, min_patch, max_patch)
        patches = _hierarchy(data, leaves, max_patch * 2)
        visits += len(leaves)
    elif variant == "relevance":
        base = _sensory(data, min_patch, max_patch)
        patches = _relevance(data, base, query_bytes, max_patch)
        visits += len(base) + len(query_bytes) + sum(len(p.payload) for p in base)
    elif variant == "structure":
        patches = _sensory(data, min_patch, max_patch)
        edges = _relations(patches)
        visits += len(patches)
    else:
        # Hybrid reuses both local causal signals, then passes the patches
        # through query-conditioned refinement and bounded structural links.
        ent = _entropy(data, min_patch, max_patch)
        sen = _sensory(data, min_patch, max_patch)
        boundaries = sorted({p.end for p in ent} | {p.end for p in sen})
        leaves = _patches(data, boundaries)
        patches = _relevance(data, leaves, query_bytes, max_patch)
        edges = _relations(patches)
        visits += len(data) + len(leaves) + len(query_bytes) + sum(len(p.payload) for p in leaves) + len(patches)
    result = Tokenization(
        variant=variant,
        length=len(data),
        source_sha256=sha256(data).hexdigest(),
        patches=patches,
        leaves=leaves,
        edges=edges,
        scan_work=visits,
        query_sha256=sha256(query_bytes).hexdigest() if variant in ("relevance", "hybrid") else None,
    )
    result.validate()
    return result
