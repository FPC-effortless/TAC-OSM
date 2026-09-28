"""Machine-readable measurement evidence: the common record envelope.

## Why this module exists

``contract.py`` made the *specification* of an experiment machine-readable.
This module makes the *outcome* machine-readable, which is the other half of
the same check. A result that exists only as terminal output exists only as
long as the terminal's scrollback does; a reader with the record can compare
an outcome against the contract's decision rule without re-running anything.

## Common envelope, experiment-specific measurements

The envelope — provenance, design, gate, endpoints, decision rule, audit,
per-seed values — is shared, because it is what a reader or a later check
needs from *every* result. What goes inside ``endpoints``, ``audit`` and
``per_seed`` is the experiment's own: F3 keys its endpoints by arm and level,
MATCHED-001 keys its by ``h_train x h_eval``, and neither is forced into the
other's shape. Forcing them would produce a third schema that serves neither.

## The instrument does not arbitrate

No record carries a ``status``. The status of an experiment is a scientific
judgement recorded against the pre-registration by whoever reads the decision
rule; having the instrument fill it in would make the *measurement* the
*arbiter*. See the contract row in ``docs/EVIDENCE_REGISTER.md``: a compliant
run is a well-specified run, not a supported hypothesis.

## Layer 0 — infrastructure, not evidence

This module is a Layer 0 *protocol* import (see
``docs/EVIDENCE_REGISTER.md``): a way of recording, not a result. It
establishes no scientific claim, and no number becomes more trustworthy by
being written down here; it becomes *checkable*, which is a different and
narrower thing.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

__all__ = [
    "Provenance",
    "Design",
    "GateCell",
    "Gate",
    "MeasurementRecord",
    "contract_fingerprint",
    "contract_path_for",
    "report_smoke",
    "git_commit",
    "contract_deviations",
    "write_record",
    "results_dir_for",
]


# --------------------------------------------------------------------------- #
# Identification: what executable specification produced this number?
# --------------------------------------------------------------------------- #
#
# The purpose of a record is that a reader can answer that question without
# reconstructing a CLI invocation from shell history. Three identifiers do it:
#
# * the experiment id — which pre-registration this run claims to implement;
# * a content hash of that contract — which *version* of it, exactly. Contracts
#   have no version field, because a version field is a thing a human forgets
#   to bump; a hash of the committed file is exact and needs no maintenance;
# * the git commit the script ran from — the implementation, including any
#   change the contract does not and cannot describe.
#
# All three degrade to a stated non-value rather than raising. A measurement
# must not fail because ``git`` is unavailable or the checkout is a tarball;
# a missing identifier is reported as missing, which is the honest result.


def git_commit() -> str:
    """The commit the measurement script is running from.

    ``"unavailable"`` rather than an exception: a run in a tarball or a
    shallow checkout still produced real numbers, and the honest record says
    the commit is unknown rather than refusing to be written.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=15,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    if out.returncode != 0 or not out.stdout.strip():
        return "unavailable"
    return out.stdout.strip()


def contract_fingerprint(path: Path) -> str:
    """A content hash of one contract file.

    Identifies the exact specification a run was checked against. Two runs
    whose fingerprints differ were checked against different designs, however
    similar the printed tables look — which is the situation a reader most
    needs to be able to detect, and the one a contract *id* alone cannot show.
    """
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def contract_deviations(
    *,
    contract_steps: int,
    contract_eval_steps: int,
    contract_h_levels: Sequence[int],
    contract_seeds: Sequence[int],
    contract_arms: Sequence[str],
    steps: int,
    eval_steps: int,
    h_levels: Sequence[int],
    seeds: Sequence[int],
    arms: Sequence[str],
) -> list[dict[str, Any]]:
    """Every way a run is off its registered design.

    Shared by all three measurement scripts, because it was the most-duplicated
    and most silently-divergent piece of the smoke report: three scripts each
    listing five dimensions by hand is three chances to list four and call it
    complete. The five are the quantities the contract pins, and a reader needs
    all five to judge whether a reported number is an artefact of the deviation.

    Empty for a compliant run, which is what makes the record answer the
    question "was this the registered design?" with a field rather than with
    the absence of a NOTE.
    """
    out: list[dict[str, Any]] = []
    pairs = (
        ("steps", steps, contract_steps),
        ("eval_steps", eval_steps, contract_eval_steps),
        ("levels", list(h_levels), list(contract_h_levels)),
        ("seeds", sorted(seeds), sorted(contract_seeds)),
        ("arms", list(arms), list(contract_arms)),
    )
    for label, got, want in pairs:
        if got != want:
            out.append({"dimension": label, "run": got, "registered": want})
    return out


@dataclass(frozen=True)
class Provenance:
    """What produced the record: the specification, the implementation, the run.

    ``recorded_at`` makes the record non-reproducible byte-for-byte, which is
    correct: a timestamp is not evidence, but its *absence* is what lets a
    re-run be silently presented as the original run. The records are
    gitignored and regenerable from deterministic seeds, so it costs nothing.
    """

    experiment_id: str
    contract_source: str
    contract_sha256: str
    git_commit: str
    script: str
    python: str
    recorded_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "contract_source": self.contract_source,
            "contract_sha256": self.contract_sha256,
            "git_commit": self.git_commit,
            "script": self.script,
            "python": self.python,
            "recorded_at": self.recorded_at,
        }


# --------------------------------------------------------------------------- #
# The envelope
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Design:
    """The design as it actually ran, registered or not.

    ``contract_checked`` is the fact that distinguishes a measurement from a
    smoke test, and it is recorded rather than implied by the absence of a
    printed NOTE — a NOTE is prose, and prose is what this module exists to
    replace.
    """

    steps: int
    eval_steps: int
    seeds: tuple[int, ...]
    h_levels: tuple[int, ...]
    k_levels: tuple[int, ...]
    arms: tuple[str, ...]
    smoke: bool
    contract_checked: bool
    deviations: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "eval_steps": self.eval_steps,
            "seeds": list(self.seeds),
            "h_levels": list(self.h_levels),
            "k_levels": list(self.k_levels),
            "arms": list(self.arms),
            "smoke": self.smoke,
            "contract_checked": self.contract_checked,
            "contract_deviations": [dict(d) for d in self.deviations],
        }


@dataclass(frozen=True)
class GateCell:
    """One cell of the reproduction gate, with the tolerance actually used.

    ``tol`` is recorded because it is the spread *floored at 0.02* (see
    :func:`tac_osm.measurement.verdicts.materiality_threshold`), and the floor
    is a pre-registered decision a reader needs to see rather than infer. A
    gate with a zero spread would otherwise demand a bit-exact reproduction,
    which is stronger than the protocol promises.
    """

    cell: str
    metric: str
    published: float
    observed: float
    diff: float
    tol: float
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell": self.cell,
            "metric": self.metric,
            "published": self.published,
            "observed": self.observed,
            "diff": self.diff,
            "tol": self.tol,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class Gate:
    """The reproduction gate: whether the frozen baseline still reproduces.

    A failed gate is recorded, not swallowed. A run that failed the gate still
    produced measurements, and the record is what a later reader uses to see
    that the run was invalid and why — deleting the evidence would delete the
    proof that it was invalid.
    """

    name: str
    tolerance: str
    passed: bool
    cells: tuple[GateCell, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tolerance": self.tolerance,
            "passed": self.passed,
            "cells": [c.to_dict() for c in self.cells],
        }


@dataclass(frozen=True)
class MeasurementRecord:
    """One published measurement, as data.

    The envelope fields are typed; ``endpoints``, ``audit`` and ``per_seed``
    are the experiment's own payload and are carried as-is. Forcing a common
    schema on them would move the experiment's structure out of the record and
    into the shared module, which is where the next drift would then live.
    """

    provenance: Provenance
    design: Design
    gate: Gate
    endpoints: dict[str, Any]
    decision_rule: tuple[dict[str, Any], ...]
    audit: dict[str, Any]
    per_seed: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "provenance": self.provenance.to_dict(),
            "design": self.design.to_dict(),
            "gate": self.gate.to_dict(),
            "endpoints": self.endpoints,
            "decision_rule": [dict(d) for d in self.decision_rule],
            "audit": self.audit,
            "per_seed": self.per_seed,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False) + "\n"


def contract_path_for(script_file: str, experiment_id: str) -> Path:
    """``contracts/<id>.json`` at the repository root, resolved from a script.

    The script's own view of the contracts directory: one level up from
    ``scripts/``. It agrees with ``contract._contracts_dir()``, which resolves
    the same directory from inside the package, because both locate the
    repository root. Two resolvers is deliberate — the alternative is a script
    importing a private helper from the contract module — and a disagreement
    between them is loud rather than silent: the fingerprint becomes
    ``"missing"`` and ``load_contract`` raises, so neither can quietly point
    somewhere else.
    """
    here = Path(script_file).resolve().parent
    return here.parent / "contracts" / f"{experiment_id}.json"


def report_smoke(contract: object, experiment_id: str, *, steps: int,
                 eval_steps: int, h_levels: Sequence[int], seeds: Sequence[int],
                 arms: Sequence[str]) -> list[dict[str, Any]]:
    """Print a smoke run's deviation from the registered design, in full.

    ``--smoke`` is the one declared way to run something that is not the
    registered design: it weakens no check, it states out loud that the run is
    not a result. What makes that statement worth anything is that the
    deviation is printed — every dimension the contract pins, so the output
    cannot be mistaken for a measurement. This is the shared implementation of
    that report, which was copy-pasted across the three scripts with only the
    experiment id differing: three copies of a five-dimension list is three
    chances to list four and call it complete.

    Returns the deviations for the record, so the printed report and the
    written record come from one comparison rather than two that can disagree.
    """
    deviations = contract_deviations(
        contract_steps=contract.steps,
        contract_eval_steps=contract.eval_steps,
        contract_h_levels=contract.h_levels,
        contract_seeds=contract.seeds,
        contract_arms=[a.name for a in contract.arms],
        steps=steps,
        eval_steps=eval_steps,
        h_levels=h_levels,
        seeds=seeds,
        arms=arms,
    )
    print("=" * 72)
    print("SMOKE TEST — declared with --smoke; this is not a measurement")
    print("=" * 72)
    print("The contract check is skipped on the declared deviation, and the")
    print("deviation is printed here so this output cannot be read as a")
    print(f"result of {experiment_id}:")
    if deviations:
        for d in deviations:
            print(f"  {d['dimension']}: run has {d['run']}, "
                  f"registered {d['registered']}")
    else:
        print("  no deviation found: the registered design is in force")
    print()
    return deviations


def results_dir_for(script_file: str) -> Path:
    """``results/`` at the repository root, resolved from a script's own path.

    Not from the CWD, because a contract lookup that depended on the caller's
    working directory would silently write somewhere unexpected — the same
    reasoning as ``contract._contracts_dir``.
    """
    here = Path(script_file).resolve().parent
    return here.parent / "results"


def write_record(record: MeasurementRecord, out_path: Path) -> None:
    """Write a record atomically.

    The temp file and ``replace`` mean a reader never observes a half-written
    record, and a failed write does not leave a truncated file that parses as
    an incomplete result — which is the failure mode most likely to be read as
    a real one.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(record.to_json(), encoding="utf-8")
    tmp.replace(out_path)


def now() -> str:
    """UTC now, in the one form the record uses.

    Isolated so a test can pin it, and so every record agrees on what a
    timestamp looks like — three scripts inventing three formats is three
    chances for a parser to read one as the other.
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _python_version() -> str:
    return ".".join(str(p) for p in sys.version_info[:3])
