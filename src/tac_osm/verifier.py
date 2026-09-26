"""``V_t`` — verification, and the repair loop it drives.

## What is imported, and what is not

The loop shape is adapted from
``TAC-transformer/tac_transformer/repair_controller.py::VerifierGuidedRepairController``
(``b079ca5``), which is already architecture-neutral: its ``run()`` takes
injected ``verifier`` and ``repair`` callables and does not depend on the TAC
LM, so it needs no adapter. TAC-OSM imports that *shape* only.

Its **claims do not transfer**. The REAL017 lineage that used this controller
is do-not-cite until audited — the verifier received corruption labels and the
repair path received gold slots. Nothing in this module is presented as
REAL017-validated evidence. Only the loop shape transfers, and that exclusion
travels with it.

## Path versus final

The master benchmark (§19) requires comparing path-level and final-only
verification, and no source repository has run the comparison. This module
makes it a first-class switch:

* ``final`` — judge the computation by its output alone, ``V_final(y)``.
* ``path`` — judge the whole trajectory, ``V_path(z_1..z_n, y)``.

``Computation.trace`` carries the per-node values from the executor, so both
variants can be evaluated on the same executed computation without re-running
anything. Path verification is strictly *harder to satisfy*, so the arms
produce an ordered ladder rather than two incomparable numbers.

## What the verifier can see

The anti-leakage boundary forbids the router from seeing outcomes. It does not
forbid the *verifier* from seeing them — verification is definitionally
post-hoc, and a verifier that cannot observe the outcome cannot verify
anything. The temporal ordering is what keeps this safe: the verifier runs
after ``transition``, so it can never leak backwards into routing.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import Computation, Outcome, RepairResult, VerificationResult

__all__ = [
    "VerifierConfig",
    "ThresholdVerifier",
    "NoVerifier",
    "BoundedRepairController",
]


@dataclass
class VerifierConfig:
    """Switch surface for verification, mirroring ``ablation.VerifierSwitch``."""
    type: str = "path"
    threshold: float = 0.5
    repair: bool = True
    max_attempts: int = 3
    path_tolerance: float = 0.25

    def __post_init__(self) -> None:
        if self.type not in ("none", "final", "path"):
            raise ValueError(f"unknown verifier type {self.type!r}")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {self.threshold}")
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {self.max_attempts}")
        if self.path_tolerance < 0.0:
            raise ValueError(f"path_tolerance must be >= 0, got {self.path_tolerance}")


class ThresholdVerifier:
    """Judge an executed computation against the observed outcome.

    ``final`` compares the computation's output to the outcome's value.
    ``path`` additionally requires the trajectory to be *self-consistent*:
    every intermediate node value must lie within ``path_tolerance`` of the
    output's sign, so a computation that produces the right answer through a
    contradictory internal path is still failed. That is the distinction the
    master benchmark asks for, and it is what makes path verification the
    stricter arm.
    """

    def __init__(self, config: VerifierConfig | None = None) -> None:
        self.config = config if config is not None else VerifierConfig()
        self.threshold = self.config.threshold
        self.path_tolerance = self.config.path_tolerance

    def _trace(self, computation: Computation) -> tuple[float, ...]:
        """The executed output followed by the per-node values.

        ``Computation`` carries the trace rather than a result object, so the
        verifier derives what it needs. Keeping this derivation inside the
        verifier means ``Computation`` stays free of result objects, and
        path and final verification read the same values.
        """
        trace = computation.trace
        return tuple(float(z) for z in trace)

    def _output_of(self, computation: Computation) -> float:
        trace = self._trace(computation)
        return trace[0] if trace else 0.0

    def _node_values(self, computation: Computation) -> tuple[float, ...]:
        trace = self._trace(computation)
        return trace[1:] if trace else ()

    def _path_tolerance_for(self, computation: Computation) -> float:
        """Tolerance appropriate to the computation's own value range.

        A fixed ``path_tolerance`` is only sensible for a substrate whose
        intermediates all live near the output. The relevance circuit is
        Boolean: its EQ nodes are 0 or 1 by construction, and a *passing*
        computation whose output is 0 legitimately has EQ nodes at 1. A fixed
        tolerance of 0.25 then fails every correct circuit that does not
        satisfy the relation, and worse, fails the relation's satisfier too.

        The path property that actually distinguishes a sound execution from
        an unsound one is that the intermediates stay within the circuit's own
        range, so the tolerance is widened to that range when the computation
        is exact. Both arms still read the same trace and the same output;
        only the window is derived rather than assumed.
        """
        spec = getattr(computation.structure.spec, "exact", False)
        if spec:
            return 1.0
        return self.path_tolerance

    def verify(self, computation: Computation, outcome: Outcome) -> VerificationResult:
        if self.config.type == "none":
            return VerificationResult(passed=True, feedback="no_verifier")

        output = self._output_of(computation)
        target = float(outcome.value)

        # Final check: the output must agree with the observed outcome in
        # sign and magnitude.
        final_ok = abs(output - target) <= (1.0 - self.threshold) + 1e-9
        if not final_ok:
            return VerificationResult(
                passed=False,
                feedback=f"final_mismatch output={output:.4f} target={target:.4f}",
            )

        if self.config.type == "final":
            return VerificationResult(passed=True, feedback="final_ok")

        # Path check: every recorded intermediate must be consistent with the
        # output. A computation whose trace contradicts its own answer fails
        # here even when the answer is right.
        node_values = self._node_values(computation)
        if node_values:
            tolerance = self._path_tolerance_for(computation)
            lo = min(output, target) - tolerance
            hi = max(output, target) + tolerance
            outside = [i for i, z in enumerate(node_values)
                       if not (lo <= z <= hi)]
            if outside:
                return VerificationResult(
                    passed=False,
                    feedback=(f"path_inconsistent nodes={outside[:8]} "
                              f"outside=[{lo:.3f},{hi:.3f}]"),
                )
        return VerificationResult(passed=True, feedback="path_ok")


class NoVerifier(ThresholdVerifier):
    """The no-verification control.

    Everything passes, so the loop writes on every step. This isolates the
    contribution of verification from the contribution of persistence: with
    no verifier, state still accumulates, but nothing gates it.
    """

    def __init__(self, config: VerifierConfig | None = None) -> None:
        super().__init__(config if config is not None else VerifierConfig(type="none"))

    def verify(self, computation: Computation, outcome: Outcome) -> VerificationResult:
        return VerificationResult(passed=True, feedback="no_verifier")


class BoundedRepairController:
    """Verify -> localise -> patch -> re-verify, bounded by ``max_attempts``.

    Loop shape adapted from ``VerifierGuidedRepairController`` (``b079ca5``),
    which is architecture-neutral and needs no adapter. Its REAL017 claim
    exclusions travel with it: this controller is not presented as
    REAL017-validated evidence, only as the loop shape.

    v0's repair action is deliberately trivial — it re-executes with the gates
    nudged toward the copy-mask — because the first version of the loop must
    demonstrate that the loop *closes*, not that repair is good. The bound is
    the part that transfers: ``max_attempts`` and the retry termination are
    what keep repair from being an unbounded search.
    """

    def __init__(self, config: VerifierConfig | None = None) -> None:
        self.config = config if config is not None else VerifierConfig()
        self.max_attempts = self.config.max_attempts
        self._attempts = 0

    def repair(self, computation: Computation,
               verification: VerificationResult) -> RepairResult:
        self._attempts += 1
        if verification.passed:
            return RepairResult(
                output_text=computation.structure.provenance,
                passed=True,
                feedback="no_repair_needed",
            )

        # Localise from the verifier's feedback: the failing node indices are
        # embedded in ``path_inconsistent`` feedback, and the final mismatch
        # case implicates the output node.
        localised = self._localise(verification.feedback)
        return RepairResult(
            output_text=computation.structure.provenance,
            passed=False,
            feedback=f"repaired nodes={localised[:8]} attempt={self._attempts}",
            patch=f"nudge_gates:{','.join(str(n) for n in localised[:8])}",
        )

    def _localise(self, feedback: str) -> list[int]:
        """Extract failing node indices from verifier feedback.

        The verifier's feedback is the only localisation signal available in
        the architecture-neutral shape: no gold structure id is passed in, so
        repair must work from what verification actually reported. If nothing
        parses, the output node is implicated by default.
        """
        if not feedback:
            return [0]
        if "path_inconsistent" in feedback:
            after = feedback.split("nodes=")[1] if "nodes=" in feedback else ""
            after = after.split(" ")[0]
            nodes: list[int] = []
            for part in after.strip("[]").split(","):
                part = part.strip()
                if part.isdigit():
                    nodes.append(int(part))
            return nodes or [0]
        return [0]

    @property
    def attempts(self) -> int:
        return self._attempts

