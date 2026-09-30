"""Machine-readable experiment contracts.

## Why this module exists

Every experiment in this repository is pre-registered: the arms, the
endpoints, the decision rule and the interpretation order are committed in a
document *before* the run. That is what makes the results falsifiable rather
than tunable.

Pre-registration in prose has one weakness. It is checked by reading, and
reading is how spec drift happens: a run that used ``--steps 400`` against a
registered ``500`` differs from the registered design, and nothing in the
output of that run says so. The prose cannot be compared with the run, because
prose has no fields.

This module makes the contract a *thing* — a JSON document with named fields
that a program can compare against the run that claims to implement it. It is
part (a) of the research-integrity work: a spec drift detector.

## The three things a contract pins, and only these

1. **What is held constant** — the registered levels, seeds and schedule
   length. A run that changes these is a different experiment and must say so.
2. **What is being tested** — the named arms, and the intervention surface
   each one acts on, so "which knob did this arm touch" is a field and not a
   paragraph a reader has to locate.
3. **What counts** — the endpoints, which one is primary, the materiality
   threshold, and the decision rule's branches with their recorded
   consequences attached *before* the result is known.

It deliberately does **not** pin outcomes. A contract that encoded the
expected result would be a foregone conclusion, not a pre-registration. The
``status`` field records the outcome after the run, and it lives in the
contract only so the *distance between the registered design and the reported
outcome* is one lookup rather than two documents.

## Why JSON and not YAML

The interface package is deliberately zero-dependency pure Python (see
``pyproject.toml``): importing a serialisation library here would make the
ablation surface depend on a parser. JSON is in the standard library, and a
contract that cannot be read without installing something is a contract that
will stop being read. The cost is no comments in the file; the fields carry
their own ``description``.

## How a contract is used

A measurement script loads its contract, and the run's own constants are
compared against the registered ones before any measurement is reported:

    contract = load_contract("TACOSM-SURROGATE-001")
    contract.require_levels(h_levels)
    contract.require_seeds(seeds)
    contract.require_steps(steps)

A mismatch raises :class:`ContractError` rather than printing a warning, for
the same reason the model-state integrity gate raises rather than flags: a run
that does not match its pre-registration must not be able to complete and emit
a number that reads as though it did. A smoke test that deliberately shortens
the schedule constructs its own contract or uses ``expect_strict=False``.

## Layer 0 — this module is infrastructure, not evidence

This module is a Layer 0 *protocol* import (see
``docs/EVIDENCE_REGISTER.md``): a way of measuring, not a measurement. It
inherits the notion that a design should be fixed before the run from general
experimental practice, and the local adaptation — a contract schema that names
the interpretation order and the decision-rule branches as fields — is what
this repository contributes. The module itself establishes no scientific
claim, and no number in ``results/`` becomes more trustworthy by being written
down here; it becomes *checkable*, which is a different and narrower thing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

__all__ = [
    "ContractError",
    "ArmSpec",
    "EndpointSpec",
    "DecisionBranch",
    "AmendmentSpec",
    "ExperimentContract",
    "load_contract",
    "validate",
]


class ContractError(RuntimeError):
    """An experiment contract is violated, malformed, or self-inconsistent.

    Raised rather than flagged. A run whose design does not match its
    pre-registration has produced a number under conditions it did not
    declare, which is exactly the situation a silent warning permits to
    become a published result.
    """


# --------------------------------------------------------------------------- #
# The schema
# --------------------------------------------------------------------------- #
#
# Every field below is a thing a reader would otherwise have to hunt for
# across several documents. Nothing here predicts a result; the fields that
# describe the outcome (``status``, ``result_note``) are filled after the run
# and are readable as *post-hoc additions to a fixed design*, which is what
# distinguishes a registered outcome from a fitted one.


@dataclass(frozen=True)
class ArmSpec:
    """One named arm of the experiment.

    The arm is a *name*, not a parameter set: ``contract.json`` records that
    ``analytic_margin`` means "the analytic margin, unclipped", and
    ``arm_reward`` in ``rewards.py`` looks the name up and raises on an
    unknown one. There is deliberately no path from a contract to a hook with
    caller-chosen clip bounds, because that path is how a registered
    intervention becomes a tunable one.
    """

    name: str
    intervention: str
    acts_on: str
    knob: str = ""
    value: str = ""
    schedule: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = {
            "name": self.name,
            "intervention": self.intervention,
            "acts_on": self.acts_on,
        }
        if self.knob:
            d["knob"] = self.knob
        if self.value:
            d["value"] = self.value
        if self.schedule:
            d["schedule"] = self.schedule
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "ArmSpec":
        _require_keys(d, ("name", "intervention", "acts_on"), "arm")
        return ArmSpec(
            name=str(d["name"]),
            intervention=str(d["intervention"]),
            acts_on=str(d["acts_on"]),
            knob=str(d.get("knob", "")),
            value=str(d.get("value", "")),
            schedule=str(d.get("schedule", "")),
        )


@dataclass(frozen=True)
class EndpointSpec:
    """A registered endpoint and its role in the decision rule.

    ``primary`` is a boolean and not a ranking, because the decision rule is
    a pair — a primary endpoint and a secondary one — and the pair's joint
    outcome is what the branches describe. A run that made ``recall@16``
    primary would be answering a different question than the one registered.
    """

    name: str
    role: str
    primary: bool = False
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = {"name": self.name, "role": self.role, "primary": self.primary}
        if self.description:
            d["description"] = self.description
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "EndpointSpec":
        _require_keys(d, ("name", "role", "primary"), "endpoint")
        return EndpointSpec(
            name=str(d["name"]),
            role=str(d["role"]),
            primary=bool(d["primary"]),
            description=str(d.get("description", "")),
        )


@dataclass(frozen=True)
class DecisionBranch:
    """One branch of the decision rule, with its consequence pre-committed.

    The consequence is the load-bearing part. A decision rule without an
    attached consequence is a threshold that can be re-interpreted after the
    result is known; a branch that says *what it licenses* before the run
    cannot be, because the reader can see the commitment predates the
    outcome.
    """

    condition: str
    licenses: str
    does_not_license: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = {"condition": self.condition, "licenses": self.licenses}
        if self.does_not_license:
            d["does_not_license"] = self.does_not_license
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "DecisionBranch":
        _require_keys(d, ("condition", "licenses"), "decision branch")
        return DecisionBranch(
            condition=str(d["condition"]),
            licenses=str(d["licenses"]),
            does_not_license=str(d.get("does_not_license", "")),
        )


@dataclass(frozen=True)
class AmendmentSpec:
    """A change to a pre-registration, recorded as a change.

    A pre-registration that can be edited after the fact is not a
    pre-registration, so an amendment is not an edit: it is a *visible*
    change, carrying what it replaced, why, and what it did not touch. A
    reader who disagrees with an amendment can still see the design they were
    promised, because the old definition is a field and not an overwritten
    value.

    Amendments are expected to be rare, and the one that exists was found by
    the instrument rather than by reading: a primary endpoint whose
    definition made it arm-independent had to be amended before any
    confirmatory run, because the decision rule could not have fired on any
    data.
    """

    id: str
    applies_to: str
    old_definition: str
    new_definition: str
    rationale: str
    discovered: str = ""
    affected: str = ""
    no_result_under_previous_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "applies_to": self.applies_to,
            "old_definition": self.old_definition,
            "new_definition": self.new_definition,
            "rationale": self.rationale,
        }
        if self.discovered:
            d["discovered"] = self.discovered
        if self.affected:
            d["affected"] = self.affected
        if self.no_result_under_previous_version:
            d["no_result_under_previous_version"] = self.no_result_under_previous_version
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "AmendmentSpec":
        _require_keys(
            d,
            ("id", "applies_to", "old_definition", "new_definition", "rationale"),
            "amendment",
        )
        return AmendmentSpec(
            id=str(d["id"]),
            applies_to=str(d["applies_to"]),
            old_definition=str(d["old_definition"]),
            new_definition=str(d["new_definition"]),
            rationale=str(d["rationale"]),
            discovered=str(d.get("discovered", "")),
            affected=str(d.get("affected", "")),
            no_result_under_previous_version=str(
                d.get("no_result_under_previous_version", "")),
        )


@dataclass(frozen=True)
class ExperimentContract:
    """A machine-readable pre-registration.

    Frozen so a loaded contract cannot be edited between the check and the
    report — the same reasoning as a ``Checkpoint``: a contract that could be
    mutated after validation would defeat the point of validating it.
    """

    experiment_id: str
    title: str
    question: str
    hypothesis: str
    arms: tuple[ArmSpec, ...]
    endpoints: tuple[EndpointSpec, ...]
    decision_rule: tuple[DecisionBranch, ...]
    interpretation_order: tuple[str, ...]
    h_levels: tuple[int, ...]
    seeds: tuple[int, ...]
    steps: int
    eval_steps: int
    m_levels: tuple[int, ...] = ()
    k_levels: tuple[int, ...] = ()
    held_constant: tuple[str, ...] = ()
    reproduction_baseline: dict[str, dict[str, float]] = field(default_factory=dict)
    status: str = "pre-registered"
    result_note: str = ""
    result_url: str = ""
    layer: str = ""
    #: Changes to the registered design, each carrying the definition it
    #: replaced. Empty for an unamended pre-registration.
    amendments: tuple[AmendmentSpec, ...] = ()

    # ------------------------------------------------------------------ #
    # The checks a run performs against its contract
    # ------------------------------------------------------------------ #

    def require_levels(self, h_levels: Sequence[int], *, strict: bool = True) -> None:
        """The candidate-count levels must match the registered ones.

        ``strict`` requires the full registered set. A subset is permitted
        only with ``strict=False`` and is a smoke test, not a result: the
        reproduction gate is then informative rather than binding, and the
        run prints that note itself.
        """
        got = tuple(int(h) for h in h_levels)
        if strict:
            if got != self.h_levels:
                raise ContractError(
                    f"{self.experiment_id}: H levels {got} do not match the "
                    f"registered {list(self.h_levels)}. A run that changes the "
                    "population levels is a different experiment and needs its "
                    "own contract."
                )
            return
        missing = [h for h in self.h_levels if h not in got]
        if missing:
            raise ContractError(
                f"{self.experiment_id}: this run is a subset — the registered "
                f"levels {missing} are absent, so the reproduction gate and the "
                "decision rule cannot both be evaluated. Declare this as a smoke "
                "test or run the registered levels."
            )

    def require_seeds(self, seeds: Sequence[int], *, strict: bool = True) -> None:
        """The seed set must cover the registered one.

        Fewer seeds widen the spread and therefore the materiality threshold,
        which silently weakens the decision rule. The check is on *coverage*,
        not order, because seeds are an unordered set in the statistics.
        """
        got = {int(s) for s in seeds}
        want = set(self.seeds)
        if strict:
            if got != want:
                raise ContractError(
                    f"{self.experiment_id}: seeds {sorted(got)} do not match the "
                    f"registered {sorted(want)}. The seed spread is the "
                    "materiality threshold, so a different seed set makes the "
                    "decision rule a different rule."
                )
            return
        if not want.issubset(got):
            missing = sorted(want - got)
            raise ContractError(
                f"{self.experiment_id}: seeds {missing} are absent from this run. "
                "The spread the decision rule reads is measured over the "
                "registered seeds, and a subset does not estimate it."
            )

    def require_m_levels(self, m_levels: Sequence[int]) -> None:
        """Require registered persistent-state population levels exactly."""
        got = tuple(int(m) for m in m_levels)
        want = tuple(self.m_levels)
        if got != want:
            raise ContractError(
                f"{self.experiment_id}: M levels {got} do not match the "
                f"registered {list(want)}."
            )

    def require_k_levels(self, k_levels: Sequence[int]) -> None:
        """Require the registered K/delay levels exactly when a contract uses them."""
        got = tuple(int(k) for k in k_levels)
        want = tuple(self.k_levels)
        if got != want:
            raise ContractError(
                f"{self.experiment_id}: K levels {got} do not match the "
                f"registered {list(want)}."
            )

    def require_eval_steps(self, eval_steps: int) -> None:
        """The evaluation length must be the registered one.

        ``eval_steps`` sets how many tasks each endpoint is averaged over, so
        a shorter evaluation widens the sampling error on every endpoint at
        no visible cost: the tables look identical, and only the standard
        error changed. A smoke test that shortens it is not reporting the
        registered endpoints' values.
        """
        if int(eval_steps) != self.eval_steps:
            raise ContractError(
                f"{self.experiment_id}: eval_steps={eval_steps} but the "
                f"registered evaluation length is {self.eval_steps}. Every "
                "endpoint is an average over this many tasks, so a different "
                "length changes the sampling error on all of them at once. Use "
                "the registered length, or record this run as a smoke test."
            )

    def require_steps(self, steps: int) -> None:
        """The schedule length must be the registered one.

        This is the single most common form of spec drift in this repository:
        a shortened run that no longer matches the pre-registered schedule.
        For F2 and F3 the schedules anneal over the registered length, so a
        shorter run explores *more* than registered and is not the registered
        intervention at all.
        """
        if int(steps) != self.steps:
            raise ContractError(
                f"{self.experiment_id}: steps={steps} but the registered schedule "
                f"length is {self.steps}. The schedules anneal over the "
                "registered length, so a different step count is a different "
                "intervention. Use the registered length, or record this run as "
                "a smoke test."
            )

    def require_arms(self, arms: Sequence[str]) -> None:
        """The arm names must be exactly the registered set.

        An extra arm is a new experiment that the decision rule does not
        cover; a missing arm leaves a branch of the rule unevaluable. Both are
        errors rather than warnings because the decision rule is stated over
        the full arm set.
        """
        got = [str(a) for a in arms]
        want = [a.name for a in self.arms]
        if got != want:
            raise ContractError(
                f"{self.experiment_id}: arms {got} do not match the registered "
                f"{want}. The decision rule is stated over the registered arms, "
                "so an added or removed arm makes the rule unevaluable as "
                "written."
            )

    def primary_endpoint(self) -> str:
        """The endpoint the decision rule reads first.

        Raises if there is not exactly one: a contract with two primaries is
        answering two questions at once, and one with none has no decision
        rule to evaluate.
        """
        primaries = [e.name for e in self.endpoints if e.primary]
        if len(primaries) != 1:
            raise ContractError(
                f"{self.experiment_id}: expected exactly one primary endpoint, "
                f"found {primaries}. The decision rule reads one endpoint first "
                "and the rest as secondary context."
            )
        return primaries[0]

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "title": self.title,
            "question": self.question,
            "hypothesis": self.hypothesis,
            "arms": [a.to_dict() for a in self.arms],
            "endpoints": [e.to_dict() for e in self.endpoints],
            "decision_rule": [b.to_dict() for b in self.decision_rule],
            "interpretation_order": list(self.interpretation_order),
            "h_levels": list(self.h_levels),
            "seeds": list(self.seeds),
            "steps": self.steps,
            "eval_steps": self.eval_steps,
        }
        if self.m_levels:
            d["m_levels"] = list(self.m_levels)
        if self.k_levels:
            d["k_levels"] = list(self.k_levels)
        if self.held_constant:
            d["held_constant"] = list(self.held_constant)
        if self.reproduction_baseline:
            d["reproduction_baseline"] = {
                str(k): dict(v) for k, v in self.reproduction_baseline.items()
            }
        if self.status and self.status != "pre-registered":
            d["status"] = self.status
        if self.result_note:
            d["result_note"] = self.result_note
        if self.result_url:
            d["result_url"] = self.result_url
        if self.layer:
            d["layer"] = self.layer
        if self.amendments:
            d["amendments"] = [a.to_dict() for a in self.amendments]
        return d

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False) + "\n"

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "ExperimentContract":
        _require_keys(
            d,
            (
                "experiment_id", "title", "question", "hypothesis",
                "arms", "endpoints", "decision_rule", "interpretation_order",
                "h_levels", "seeds", "steps", "eval_steps",
            ),
            "contract",
        )
        return ExperimentContract(
            experiment_id=str(d["experiment_id"]),
            title=str(d["title"]),
            question=str(d["question"]),
            hypothesis=str(d["hypothesis"]),
            arms=tuple(ArmSpec.from_dict(a) for a in d["arms"]),
            endpoints=tuple(EndpointSpec.from_dict(e) for e in d["endpoints"]),
            decision_rule=tuple(DecisionBranch.from_dict(b) for b in d["decision_rule"]),
            interpretation_order=tuple(str(x) for x in d["interpretation_order"]),
            h_levels=tuple(int(h) for h in d["h_levels"]),
            m_levels=tuple(int(m) for m in d.get("m_levels", ())),
            seeds=tuple(int(s) for s in d["seeds"]),
            steps=int(d["steps"]),
            eval_steps=int(d["eval_steps"]),
            k_levels=tuple(int(k) for k in d.get("k_levels", ())),
            held_constant=tuple(str(x) for x in d.get("held_constant", ())),
            reproduction_baseline={
                str(k): {str(m): float(v) for m, v in cell.items()}
                for k, cell in d.get("reproduction_baseline", {}).items()
            },
            status=str(d.get("status", "pre-registered")),
            result_note=str(d.get("result_note", "")),
            result_url=str(d.get("result_url", "")),
            layer=str(d.get("layer", "")),
            amendments=tuple(
                AmendmentSpec.from_dict(a) for a in d.get("amendments", ())
            ),
        )

    # ------------------------------------------------------------------ #
    # Cross-field consistency — the invariants a reader cannot check by eye
    # ------------------------------------------------------------------ #

    def check_consistency(self) -> list[str]:
        """Self-inconsistencies that a schema alone cannot express.

        Returns a list of problems; empty means the contract is internally
        sound. These are the invariants that matter for falsifiability:

        * exactly one primary endpoint (the decision rule reads one first);
        * every arm named in the decision rule is a registered arm, so a rule
          cannot silently cover an intervention that was never specified;
        * the interpretation order is non-empty, because an empty order is a
          contract that permits any post-hoc reading;
        * the levels are strictly increasing, because a repeated level is a
          duplicated condition that a reader would read as two;
        * ``steps`` is positive, because a zero-length schedule is not a
          schedule;
        * every amendment is *complete*: an amendment that omits its old
          definition, its new one or its rationale is not an amendment but a
          silent edit, because a reader cannot compare two designs when one of
          them is not there;
        * every amendment id is unique, because two amendments sharing an id
          are one change recorded twice and a reader cannot tell which applies.

        The function does not raise on the first problem, because a contract
        with two defects is best described once, in full.
        """
        problems: list[str] = []

        try:
            self.primary_endpoint()
        except ContractError as exc:
            problems.append(str(exc))

        arm_names = {a.name for a in self.arms}
        if not arm_names:
            problems.append(f"{self.experiment_id}: no arms registered")

        for branch in self.decision_rule:
            for arm in _arm_names_in(branch.condition, arm_names):
                if arm not in arm_names:
                    problems.append(
                        f"{self.experiment_id}: decision rule references arm "
                        f"{arm!r} which is not registered"
                    )

        if not self.interpretation_order:
            problems.append(
                f"{self.experiment_id}: interpretation_order is empty — a "
                "contract without a committed order permits a post-hoc reading"
            )
        elif len(set(self.interpretation_order)) != len(self.interpretation_order):
            problems.append(
                f"{self.experiment_id}: interpretation_order repeats an entry"
            )

        if len(set(self.h_levels)) != len(self.h_levels):
            problems.append(
                f"{self.experiment_id}: h_levels repeats a level"
            )
        elif list(self.h_levels) != sorted(self.h_levels):
            problems.append(
                f"{self.experiment_id}: h_levels is not strictly increasing"
            )
        if self.m_levels:
            if len(set(self.m_levels)) != len(self.m_levels):
                problems.append(
                    f"{self.experiment_id}: m_levels repeats a level"
                )
            elif list(self.m_levels) != sorted(self.m_levels):
                problems.append(
                    f"{self.experiment_id}: m_levels is not strictly increasing"
                )
        if self.steps <= 0:
            problems.append(f"{self.experiment_id}: steps must be positive")
        if self.eval_steps <= 0:
            problems.append(f"{self.experiment_id}: eval_steps must be positive")

        seen_ids: set[str] = set()
        for am in self.amendments:
            if not am.id:
                problems.append(
                    f"{self.experiment_id}: an amendment has no id, so it "
                    "cannot be referred to by any later record"
                )
            elif am.id in seen_ids:
                problems.append(
                    f"{self.experiment_id}: amendment id {am.id!r} is used by "
                    "two amendments"
                )
            else:
                seen_ids.add(am.id)

        return problems


# --------------------------------------------------------------------------- #
# Loading and validation
# --------------------------------------------------------------------------- #


def _contracts_dir() -> Path:
    """``contracts/`` at the repository root.

    Resolved from this file rather than from the CWD, because the measurement
    scripts are invoked from the repository root and a contract lookup that
    depended on the caller's working directory would silently fail to find a
    contract that exists.

    The module lives at ``src/tac_osm/contract.py``, so the repository root is
    two levels up from the file and one above the ``src`` tree: the package's
    own parent is ``src``, which is not the root.
    """
    here = Path(__file__).resolve().parent
    return here.parent.parent / "contracts"


def _require_keys(d: dict[str, Any], keys: Sequence[str], label: str) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise ContractError(
            f"malformed {label}: missing required keys {missing}"
        )


def _arm_names_in(text: str, known_arm_names: set[str] | None = None) -> set[str]:
    """Return arm identifiers mentioned in decision-rule prose.

    Endpoint names often contain underscores too, so treating every snake_case
    token as an arm creates false contract failures. An identifier is an arm
    when it is a registered arm, or when the prose explicitly places the
    identifier next to the word 'arm'. The second rule preserves the existing
    negative control for an unregistered arm such as 'epsilon_greedy arm'.
    """
    import re

    known_arm_names = known_arm_names or set()
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)
    out: set[str] = set()
    for i, tok in enumerate(tokens):
        if tok in known_arm_names:
            out.add(tok)
            continue
        if len(tok) >= 4 and tok.lower() not in _COMMON_WORDS and "_" in tok:
            nearby = tokens[max(0, i - 2): i + 3]
            if "arm" in {x.lower() for x in nearby}:
                out.add(tok)
    return out

#: Words that appear in decision-rule prose and are not arm names. Keeping
#: this list small and obvious is deliberate: the check's job is to catch a
#: rule covering an unregistered arm, and an over-broad stoplist would make
#: it useless by complaining about every branch.
_COMMON_WORDS = frozenset({
    "does_not", "the_primary", "the_baseline", "the_materiality_threshold",
})


def load_contract(experiment_id: str, *, path: Path | None = None) -> ExperimentContract:
    """Load one contract by experiment id.

    ``path`` is for tests and for the rare case where a contract lives
    somewhere other than ``contracts/``; production code resolves the
    directory from this module's location.
    """
    p = path if path is not None else _contracts_dir() / f"{experiment_id}.json"
    if not p.is_file():
        raise ContractError(
            f"no contract found for {experiment_id!r} at {p}. Either the id is "
            "wrong, or the experiment has no machine-readable contract and "
            "needs one before its results can be checked."
        )
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ContractError(f"malformed JSON in {p}: {exc}") from exc
    return ExperimentContract.from_dict(raw)


def validate(path: Path | None = None) -> list[str]:
    """Validate every contract, returning each contract's problems.

    Runs both the schema checks (``from_dict`` raises on a bad file) and the
    cross-field checks. A contract that fails validation is a pre-registration
    that cannot be checked against a run, which is as useless as having no
    contract, so the failure is reported with the file's name.

    This is the entry point the test suite calls: one assertion over all
    contracts, so a contract added later cannot be forgotten.
    """
    directory = path if path is not None else _contracts_dir()
    if not directory.is_dir():
        raise ContractError(
            f"no contracts directory at {directory} — the experiment contracts "
            "are missing, so no pre-registration in this repository is "
            "machine-checkable."
        )
    all_problems: list[str] = []
    for p in sorted(directory.glob("*.json")):
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            all_problems.append(f"{p.name}: malformed JSON: {exc}")
            continue
        try:
            contract = ExperimentContract.from_dict(raw)
        except ContractError as exc:
            all_problems.append(f"{p.name}: {exc}")
            continue
        for problem in contract.check_consistency():
            all_problems.append(f"{p.name}: {problem}")
    return all_problems


def load_all(path: Path | None = None) -> dict[str, ExperimentContract]:
    """Every contract keyed by experiment id. For the index and for audits."""
    directory = path if path is not None else _contracts_dir()
    if not directory.is_dir():
        raise ContractError(f"no contracts directory at {directory}")
    out: dict[str, ExperimentContract] = {}
    for p in sorted(directory.glob("*.json")):
        raw = json.loads(p.read_text(encoding="utf-8"))
        c = ExperimentContract.from_dict(raw)
        out[c.experiment_id] = c
    return out
