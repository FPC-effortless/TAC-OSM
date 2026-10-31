"""SENSE-TOK-002: prefix-only streaming segmentation with availability times.

This is an untrained engineering prototype, not a CASM executor or CDL teacher.
A model predicting byte x[t] must request its view before observing x[t].
For T2b, the byte triggering an earlier boundary is seen at the commit time.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from hashlib import sha256
from math import log2
from typing import Literal

from .sensory_tokenizers import Patch, _classes

StreamVariant = Literal["fixed", "entropy", "sensory", "sensory_v2"]
STREAM_VARIANTS: tuple[StreamVariant, ...] = ("fixed", "entropy", "sensory", "sensory_v2")


@dataclass(frozen=True)
class PatchCommit:
    patch: Patch
    available_after: int
    requires_eos: bool = False
    reason: str = ""


@dataclass(frozen=True)
class PrefixView:
    observed: int
    committed: tuple[PatchCommit, ...]
    pending: bytes
    finalized: bool
    data_sha256: str

    def validate(self) -> None:
        consumed = 0
        chunks = []
        for c in self.committed:
            p = c.patch
            if p.start != consumed or not (p.end <= c.available_after <= self.observed):
                raise ValueError("invalid or prematurely visible patch commit")
            if c.requires_eos and not self.finalized:
                raise ValueError("EOS-only commit exposed before EOS")
            consumed = p.end
            chunks.append(p.payload)
        if consumed + len(self.pending) != self.observed:
            raise ValueError("incorrect observed byte count")
        if self.finalized and self.pending:
            raise ValueError("post-EOS pending data")
        chunks.append(self.pending)
        if sha256(b"".join(chunks)).hexdigest() != self.data_sha256:
            raise ValueError("prefix hash mismatch")

    @property
    def complete_byte_payload(self) -> bytes:
        return b"".join(c.patch.payload for c in self.committed) + self.pending


class CausalPatchStream:
    """Incremental, prefix-only segmentation; no implicit EOS in snapshots."""

    def __init__(
        self, *, variant: StreamVariant, min_patch: int = 2,
        max_patch: int = 24, fixed_size: int = 8,
        entropy_threshold: float = 6.4, entropy_window: int = 32,
    ) -> None:
        if variant not in STREAM_VARIANTS:
            raise ValueError("unknown variant")
        if not (1 <= min_patch <= max_patch) or fixed_size < 1 or entropy_window < 1:
            raise ValueError("invalid patch/window limits")
        if not (0 < entropy_threshold <= 9):
            raise ValueError("invalid surprise threshold")
        self.variant = variant
        self.min_patch = min_patch
        self.max_patch = max_patch
        self.fixed_size = fixed_size
        self.entropy_threshold = entropy_threshold
        self.entropy_window = entropy_window
        self._pending = bytearray()
        self._committed: list[PatchCommit] = []
        self._observed = bytearray()
        self._history: deque[int] = deque()
        self._counts: Counter[int] = Counter()
        self._finalized = False
        self._work_visits = 0
        self._delayed_commits = 0

    @property
    def observed_count(self) -> int:
        return len(self._observed)

    @property
    def work_visits(self) -> int:
        """Instrumented byte visits, NOT FLOPs or GPU energy."""
        return self._work_visits

    @property
    def delayed_commits(self) -> int:
        return self._delayed_commits

    def _commit(self, *, available_after: int, reason: str, salience: float = 0.0,
                requires_eos: bool = False) -> PatchCommit:
        if not self._pending:
            raise ValueError("cannot commit an empty patch")
        end = len(self._observed)
        start = end - len(self._pending)
        event = PatchCommit(
            Patch(start, end, bytes(self._pending), float(salience)),
            available_after=available_after, requires_eos=requires_eos, reason=reason,
        )
        if available_after > end:
            self._delayed_commits += 1
        self._committed.append(event)
        self._pending.clear()
        return event

    def observe(self, chunk: bytes) -> tuple[PatchCommit, ...]:
        """Append observed input; caller must predict each target before observing it."""
        if self._finalized:
            raise ValueError("cannot observe after EOS")
        if not isinstance(chunk, bytes):
            raise TypeError("observation must be raw bytes")
        created: list[PatchCommit] = []
        for b in chunk:
            at = len(self._observed)
            t = at + 1
            prev = self._observed[-1] if at else None
            prevprev = self._observed[-2] if at >= 2 else None
            class_jump = bool(at and _classes(prev) != _classes(b))
            if self.variant == "sensory_v2" and self._pending and (
                len(self._pending) >= self.max_patch or
                (len(self._pending) >= self.min_patch and class_jump)
            ):
                created.append(self._commit(available_after=t, reason="before_class_or_max"))
            surprise = 0.0
            if self.variant == "entropy":
                surprise = -log2((self._counts[b] + 1) / (len(self._history) + 256))
                self._work_visits += 1
            self._observed.append(b)
            self._pending.append(b)
            self._work_visits += 1
            if self.variant == "fixed":
                if len(self._pending) >= self.fixed_size:
                    created.append(self._commit(available_after=t, reason="fixed"))
            elif self.variant == "entropy":
                if len(self._pending) >= self.max_patch or (
                    len(self._pending) >= self.min_patch and surprise >= self.entropy_threshold
                ):
                    created.append(self._commit(available_after=t, reason="surprise", salience=surprise))
                self._history.append(b)
                self._counts[b] += 1
                if len(self._history) > self.entropy_window:
                    old = self._history.popleft()
                    self._counts[old] -= 1
                    if self._counts[old] == 0:
                        del self._counts[old]
            elif self.variant == "sensory":
                transient = bool(at >= 2 and b != prev and prev == prevprev)
                salience = 2 * int(class_jump) + int(transient)
                if len(self._pending) >= self.max_patch or (
                    len(self._pending) >= self.min_patch and salience >= 2
                ):
                    created.append(self._commit(available_after=t, reason="after_class_or_max", salience=salience))
        return tuple(created)

    def snapshot(self) -> PrefixView:
        """Read only what has been observed, without a fabricated end token."""
        v = PrefixView(
            observed=self.observed_count,
            committed=tuple(self._committed),
            pending=bytes(self._pending),
            finalized=self._finalized,
            data_sha256=sha256(self._observed).hexdigest(),
        )
        v.validate()
        return v

    def finish(self) -> tuple[PatchCommit, ...]:
        """End-of-sequence is a separate observed event, not inferred from prefix."""
        if self._finalized:
            raise ValueError("EOS already observed")
        self._finalized = True
        if self._pending:
            return (self._commit(available_after=self.observed_count, reason="eos", requires_eos=True),)
        return ()


def tokenize_stream(data: bytes, *, variant: StreamVariant, **kwargs) -> PrefixView:
    """Offline complete-string view; NEVER feed this into an earlier-byte prediction."""
    stream = CausalPatchStream(variant=variant, **kwargs)
    stream.observe(data)
    stream.finish()
    return stream.snapshot()
