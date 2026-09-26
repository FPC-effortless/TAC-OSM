"""Model-state integrity — the "did we actually load a trained model" gate.

## Why this module exists

TACOSM-HS-001 produced a false scientific conclusion by exactly this route:

    learn=False
    + weights never copied into the evaluation router
    -> w = [0.0] * n
    -> softmax over identical scores = uniform
    -> entropy ln(H)
    -> routing@1 read as ~1/40 and interpreted as "the router's decision is
      meaningless"

The measurement looked plausible. It was not. A zeroed parameter vector
produces a *smooth, bounded, sensible-looking* distribution, so an evaluation
against untrained weights is not detectable from its outputs — only from the
parameters themselves.

This is the same error class as the ``dd8f63c`` representability failure that
motivated ``representability.py``: a broken mechanism reports a plausible
number while nothing in the pipeline objects. There the failure was in the
feature basis and was caught by asserting the basis could express the
relation. Here the failure is in the *model state* and is caught by asserting
the state is not the state the model was born with.

## The gate

For any evaluation that consumes a trained model:

1. snapshot the parameters at *training* end;
2. load them at *evaluation* start;
3. assert the loaded parameter hash is not the untrained hash;
4. record the snapshot hash in the run manifest.

A failure raises :class:`IntegrityError` rather than returning a flag, for the
same reason a failed representability gate is a build failure: an evaluation
run against untrained weights must not be able to complete and emit a number.

## What this deliberately does not do

It does not verify the weights are *good*, only that they are not the
initialisation. A hash cannot distinguish a well-trained router from a
poorly-trained one; that is what the baseline and the ablation matrix are for.
The norm and the block-mean signature are recorded so a reader can inspect
them, but only the untrained check is enforced, because it is the only one
with a threshold that is not a modelling judgement.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass, field
from typing import Any, Sequence

__all__ = [
    "IntegrityError",
    "UNTRAINED_HASH",
    "untrained_hash_for",
    "parameter_hash",
    "is_untrained",
    "parameter_norm",
    "Checkpoint",
    "snapshot_router",
    "assert_trained",
    "checkpoint_to_dict",
    "checkpoint_from_dict",
    "manifest_to_dict",
]


class IntegrityError(RuntimeError):
    """A model-state integrity gate failed.

    Raised rather than flagged: an evaluation that ran against untrained
    weights has produced a number, and a number that cannot be un-produced is
    worse than no number at all.
    """


# --------------------------------------------------------------------------- #
# Parameter hashing
# --------------------------------------------------------------------------- #


def _canon(x: float) -> float:
    """Canonicalise one float so equivalent states hash alike.

    ``-0.0`` and ``0.0`` are numerically equal but have distinct bit patterns,
    and the analytic weights that the representability gate uses are exactly
    zero across several whole blocks, so a sign-bit difference would make two
    identical states hash differently. Mapping signed to unsigned zero makes
    the hash a function of the *values*, not of the arithmetic that produced
    them.
    """
    if x == 0.0:
        return 0.0
    return x


def parameter_hash(weights: Sequence[float]) -> str:
    """A content hash of a parameter vector, stable across processes.

    ``hash(tuple(...))`` is salted per interpreter run and ``list`` is
    unhashable, so neither can identify a checkpoint. This packs each value in
    little-endian IEEE-754 and digests the concatenation: two vectors hash
    alike exactly when they are bit-identical after :func:`_canon`.

    Floats are hashed as bits rather than as decimal text because a text
    encoding would make the hash depend on the formatting, and ``repr`` of a
    float is not stable across Python versions.

    The length is prefixed because a hash that ignored it would collide a
    vector with its own truncation, and ``load_weights`` exists precisely to
    refuse a basis of the wrong width: a hash that cannot see the width could
    not support that check.
    """
    digest = hashlib.sha256()
    digest.update(struct.pack("<Q", len(weights)))
    for w in weights:
        digest.update(struct.pack("<d", _canon(float(w))))
    return digest.hexdigest()


def untrained_hash_for(weights: Sequence[float]) -> str:
    """The hash an all-zero vector *of this width* would have.

    A content hash cannot be width-independent — packing more zeros extends
    the byte string — so there is no single ``UNTRAINED_HASH`` constant that
    matches a router of every ``dim`` and ``max_state_slots``. Instead the
    manifest records the untrained hash *at the checkpoint's own width*, so a
    reader's comparison is exact.

    The gate itself does not use this: :func:`is_untrained` is the
    width-independent predicate and it is what :func:`assert_trained`
    enforces. This function is for the manifest, so a reader can confirm the
    comparison rather than trusting that it happened.
    """
    return parameter_hash([0.0] * len(weights))


#: The hash of the minimal zero vector — the one-parameter degenerate case.
#: Retained as a named constant for documentation and for the empty-basis
#: corner; **not** the value a checkpoint of real width compares against. Use
#: :func:`untrained_hash_for` for that.
UNTRAINED_HASH = parameter_hash([0.0])


def is_untrained(weights: Sequence[float]) -> bool:
    """True when the vector is identically zero, i.e. nothing was ever learned."""
    return all(w == 0.0 for w in weights)


def parameter_norm(weights: Sequence[float]) -> float:
    """L2 norm. Recorded, never enforced: the right threshold is a modelling call."""
    return math.sqrt(sum(float(w) * float(w) for w in weights))


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Checkpoint:
    """A point-in-time record of a router's learned state.

    Frozen so a checkpoint can be compared for equality and used as a key; a
    mutable checkpoint could be edited after the hash was recorded, which
    would defeat the gate.

    ``hash`` is derived from ``weights`` and is the field an evaluation
    compares against. ``n_updates`` and ``signature`` are provenance: they let
    a reader see *how much* training produced the state and *what rule* it
    encodes, but they are not part of the identity, so a re-run that trains to
    the same weights reports the same checkpoint.
    """

    arm: str
    weights: tuple[float, ...]
    hash: str
    n_updates: int = 0
    signature: dict[str, float] = field(default_factory=dict)
    norm: float = 0.0
    source: str = ""

    def __post_init__(self) -> None:
        if self.hash != parameter_hash(self.weights):
            raise IntegrityError(
                "checkpoint hash does not match its weights: the record was "
                "constructed with a hash that does not describe the state it "
                "claims to describe"
            )

    @property
    def untrained(self) -> bool:
        return is_untrained(self.weights)


def snapshot_router(router: Any) -> Checkpoint | None:
    """Record the learned state of a router, or ``None`` if it has none.

    Returns ``None`` for an arm with no ``w`` attribute — ``static``,
    ``random``, ``oracle``, ``full_context`` carry no learned parameters, so
    there is nothing to gate. ``None`` is a *legible* absence rather than an
    omitted one: the caller records it, so a manifest says "no learned
    parameters" instead of saying nothing and looking like a missing field.

    Reads ``w`` and ``updates`` by ``getattr`` rather than ``isinstance`` on
    ``LearnedRelationalRouter`` so the gate does not become a special case for
    one router class: any future learned arm with a ``w`` vector is covered by
    the same path.
    """
    w = getattr(router, "w", None)
    if w is None:
        return None
    weights = tuple(float(x) for x in w)
    arm = getattr(getattr(router, "config", None), "type", "learned")
    sig: dict[str, float] = {}
    signature = getattr(router, "weight_signature", None)
    if callable(signature):
        try:
            sig = {k: float(v) for k, v in signature().items()}
        except Exception:
            sig = {}
    return Checkpoint(
        arm=arm,
        weights=weights,
        hash=parameter_hash(weights),
        n_updates=int(getattr(router, "updates", 0)),
        signature=sig,
        norm=parameter_norm(weights),
        source=getattr(router, "_integrity_source", "training"),
    )


def assert_trained(checkpoint: Checkpoint | None, *, context: str = "") -> None:
    """Fail loudly if the model state an evaluation is about to use is empty.

    ``checkpoint`` may be ``None``, which is the "no learned parameters" case
    and is not an error. The error case is a checkpoint that *exists* and is
    the untrained vector — that is the exact TACOSM-HS-001 failure, and it is
    only detectable here, because a zeroed scorer's outputs look reasonable.

    ``context`` names the evaluation in the message, so a failing gate in one
    arm of a sweep identifies the arm rather than the sweep.
    """
    if checkpoint is None:
        return
    if checkpoint.untrained:
        raise IntegrityError(
            f"evaluation would run against an untrained parameter vector "
            f"({context or 'unnamed evaluation'}): all weights are zero, so "
            "the scorer is uniform and every metric it produces is a "
            "statement about chance, not about the model. The weights were "
            "never copied from training, or training never ran. "
            "(TACOSM-HS-001 produced a false conclusion this way.)"
        )


# --------------------------------------------------------------------------- #
# Manifest serialisation
# --------------------------------------------------------------------------- #
#
# Stage A of the roadmap defines the full RunManifest. This is deliberately the
# narrow slice of it that the integrity gate needs, so an evaluation can
# record its model state *now* rather than waiting for the research kernel.
# Stage A supersedes this dict; it does not invalidate the hashes.


def checkpoint_to_dict(checkpoint: Checkpoint | None) -> dict[str, Any]:
    """Serialise a checkpoint, including ``None`` as an explicit reason.

    The weights travel with the record so a manifest is *self-verifying*: a
    reader recomputes the hash and confirms the state, rather than trusting a
    hex string. Carrying the parameters costs little — the v0.1 basis is
    ``1 + 6*dim + 2*dim*max_slots`` floats — and it is what makes a manifest
    auditable instead of merely descriptive.
    """
    if checkpoint is None:
        return {"parameter_hash": None, "reason": "arm carries no learned parameters"}
    return {
        "arm": checkpoint.arm,
        "parameter_hash": checkpoint.hash,
        "weights": list(checkpoint.weights),
        "n_updates": checkpoint.n_updates,
        "weight_signature": dict(checkpoint.signature),
        "parameter_norm": checkpoint.norm,
        "untrained": checkpoint.untrained,
        "source": checkpoint.source,
    }


def checkpoint_from_dict(data: dict[str, Any]) -> Checkpoint | None:
    """Reconstruct a checkpoint, restoring the ``None`` absence.

    Uses the *stored* hash rather than recomputing one, so ``__post_init__``
    verifies the record against the state it claims — a manifest whose hash
    disagrees with its weights is rejected rather than silently re-hashed.
    """
    if data.get("parameter_hash") is None:
        return None
    weights = tuple(float(x) for x in data.get("weights", ()))
    if not weights:
        return None
    return Checkpoint(
        arm=str(data.get("arm", "learned")),
        weights=weights,
        hash=str(data.get("parameter_hash")),
        n_updates=int(data.get("n_updates", 0)),
        signature={k: float(v) for k, v in data.get("weight_signature", {}).items()},
        norm=float(data.get("parameter_norm", parameter_norm(weights))),
        source=str(data.get("source", "")),
    )


def manifest_to_dict(
    *,
    experiment_id: str,
    arm: str,
    checkpoint: Checkpoint | None,
    commit: str = "",
    seed: int | None = None,
    n_steps: int | None = None,
) -> dict[str, Any]:
    """The model-state portion of a run manifest.

    ``untrained_hash`` is recorded alongside the observed hash so a reader can
    confirm the comparison rather than trusting that it happened, which is the
    difference between a gate and a log line.
    """
    entry = checkpoint_to_dict(checkpoint)
    entry["untrained_hash"] = (
        untrained_hash_for(checkpoint.weights) if checkpoint is not None else None
    )
    entry["gate"] = "passed" if checkpoint is None or not checkpoint.untrained else "FAILED"
    return {
        "experiment_id": experiment_id,
        "commit": commit,
        "arm": arm,
        "seed": seed,
        "n_steps": n_steps,
        "model_state": entry,
    }
