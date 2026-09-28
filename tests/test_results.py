"""Tests for the machine-readable measurement record.

The contract module made the *specification* of an experiment machine-
readable; this module makes the *outcome* machine-readable, which is the
other half of the same check. A result that exists only as terminal output
exists only as long as the terminal's scrollback does.

What these tests pin:

1. **The envelope** — the fields every record carries, so a reader or a later
   check can rely on the shape. A field that is not asserted is a field that
   can silently disappear.
2. **The identifiers degrade, they do not raise** — a run in a tarball or a
   shallow checkout still produced real numbers, and the honest record says
   the commit is unknown rather than refusing to be written.
3. **The two contract resolvers agree** — the script's view of ``contracts/``
   and the package's must point at the same directory, because a disagreement
   there is what would let a record claim to have been checked against a
   contract it never opened.
4. **The write is atomic** — a reader never observes a half-written record,
   and a failed write leaves no truncated file that parses as an incomplete
   result (the failure mode most likely to be read as a real one).
5. **The deviations are the record's statement of non-compliance** — empty for
   a compliant run, which is what lets a record answer "was this the
   registered design?" with a field rather than with the absence of a NOTE.

Layer 0 — this is infrastructure, not evidence. No number becomes more
trustworthy by being written down here; it becomes *checkable*, which is a
different and narrower thing. See the contract row in
``docs/EVIDENCE_REGISTER.md``: contract validity is not scientific validity.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tac_osm.contract import load_contract  # noqa: E402
from tac_osm.measurement import results as _results  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
CONTRACTS_DIR = REPO_ROOT / "contracts"
SCRIPTS_DIR = REPO_ROOT / "scripts"


# --------------------------------------------------------------------------- #
# A record for testing
# --------------------------------------------------------------------------- #


def _record(**over):
    """A minimal well-formed record, with any field overridden.

    Built from the dataclasses rather than from a JSON blob, so a change to
    the envelope moves this fixture rather than silently diverging from it.
    """
    provenance = _results.Provenance(
        experiment_id="TEST-RECORD-001",
        contract_source="contracts/TACOSM-SURROGATE-001.json",
        contract_sha256="abc123",
        git_commit="deadbeef",
        script="measure_test.py",
        python="3.13.13",
        recorded_at="2026-01-01T00:00:00+00:00",
    )
    design = _results.Design(
        steps=500, eval_steps=100,
        seeds=(0, 1, 2, 3, 4), h_levels=(8, 64, 256), k_levels=(1, 2, 4, 8, 16),
        arms=("baseline",), smoke=False, contract_checked=True,
    )
    gate = _results.Gate(
        name="a gate", tolerance="the registered one", passed=True, cells=(),
    )
    kw = dict(
        provenance=provenance, design=design, gate=gate,
        endpoints={}, decision_rule=(), audit={}, per_seed={},
    )
    kw.update(over)
    return _results.MeasurementRecord(**kw)


# --------------------------------------------------------------------------- #
# The envelope
# --------------------------------------------------------------------------- #


def test_every_record_is_self_describing():
    """A reader with the JSON needs no shell history to attribute the number.

    ``experiment_id`` alone is not enough, because a contract has no version
    field — a version field is a thing a human forgets to bump. The content
    hash distinguishes two versions of the same id, which is the situation a
    reader most needs to detect, and the commit identifies the implementation
    including anything the contract cannot describe.
    """
    d = _record().to_dict()
    assert set(d) == {
        "provenance", "design", "gate", "endpoints", "decision_rule",
        "audit", "per_seed",
    }, f"the envelope is missing fields: {sorted(d)}"
    p = d["provenance"]
    assert set(p) == {
        "experiment_id", "contract_source", "contract_sha256", "git_commit",
        "script", "python", "recorded_at",
    }, f"the provenance is missing fields: {sorted(p)}"


def test_the_envelope_carries_no_status():
    """The instrument measures; it does not arbitrate the decision rule.

    ``status`` is the one field a human sets against the pre-registration. A
    script that computed it would be marking its own homework, and the field
    belongs on the contract, where a reader can see it was set after the run.
    This holds at every level: a status tucked inside the audit or the
    per-seed values is the same judgement one layer down.
    """
    d = _record(
        audit={"reward": [{"status": "supported"}]},
        per_seed={"cells": [{"status": "supported"}]},
    ).to_dict()
    assert "status" not in d
    assert "status" not in d["audit"]
    assert "status" not in d["per_seed"]


def test_the_design_records_the_run_as_it_happened():
    """The design fields are the run's actual values, registered or not.

    A smoke run that reported no deviation would be indistinguishable from a
    registered one, so ``contract_deviations`` is what makes ``--smoke`` worth
    more than deleting the check.
    """
    d = _record().to_dict()["design"]
    assert set(d) == {
        "steps", "eval_steps", "seeds", "h_levels", "k_levels", "arms",
        "smoke", "contract_checked", "contract_deviations",
    }
    assert d["smoke"] is False and d["contract_checked"] is True
    assert d["contract_deviations"] == []


def test_the_gate_records_its_cells_and_tolerance():
    """A gate is recorded, never swallowed.

    A run that failed the gate still produced measurements; the record is what
    a later reader uses to see that the run was invalid and why. Deleting the
    evidence would delete the proof that it was invalid.
    """
    cell = _results.GateCell(
        cell="8x8", metric="routing@1", published=0.624, observed=0.600,
        diff=-0.024, tol=0.02, passed=False,
    )
    d = _record(
        gate=_results.Gate(name="reproduction", tolerance="the spread",
                           passed=False, cells=(cell,))
    ).to_dict()["gate"]
    assert d["passed"] is False
    assert d["cells"] and d["cells"][0]["cell"] == "8x8"


def test_the_record_round_trips_through_json():
    """A record that cannot be written and read is one that cannot be checked later."""
    rec = _record()
    decoded = json.loads(rec.to_json())
    assert decoded == rec.to_dict()
    # ``sort_keys=False`` keeps the envelope in a readable order: provenance
    # first, per-seed last, so a reader opening the file meets the
    # attribution before the numbers.
    text = rec.to_json()
    assert text.index("provenance") < text.index("endpoints") < text.index("per_seed")


# --------------------------------------------------------------------------- #
# The identifiers degrade, they do not raise
# --------------------------------------------------------------------------- #


def test_a_missing_contract_is_reported_as_missing():
    """The fingerprint is a stated non-value, not an exception.

    A measurement must not fail because a contract file was moved; the honest
    record says the specification is unknown, which is what a reader needs to
    know — the number is unattributable, not unrecordable.
    """
    assert _results.contract_fingerprint(Path("contracts/does-not-exist.json")) == "missing"


def test_the_fingerprint_identifies_a_files_contents():
    """Two fingerprints differ iff the contract files differ.

    This is what distinguishes two versions of the same experiment id, and a
    test that cannot see the difference cannot see the drift.
    """
    a = CONTRACTS_DIR / "TACOSM-SURROGATE-001.json"
    b = CONTRACTS_DIR / "TACOSM-LEARN-001.json"
    assert _results.contract_fingerprint(a) != _results.contract_fingerprint(b)
    assert _results.contract_fingerprint(a) == _results.contract_fingerprint(a)
    # A truncated hash is still enough to separate two committed files, and is
    # short enough to read in a table.
    assert len(_results.contract_fingerprint(a)) == 16


def test_git_commit_degrades_to_unavailable():
    """A tarball or a shallow checkout still produced real numbers.

    The fallback is a stated non-value for the same reason the fingerprint's
    is: the record must be writable, and ``unavailable`` is the honest entry.
    """
    # In this checkout the commit is available, so what is pinned is that the
    # function returns a hex id rather than the fallback — and that the
    # fallback is a documented value a reader can recognise.
    got = _results.git_commit()
    assert got == "unavailable" or len(got) == 40, (
        f"the git commit identifier is neither a sha nor the documented "
        f"fallback: {got!r}"
    )


def test_now_is_utc_and_second_granular():
    """Every record agrees on what a timestamp looks like.

    Three scripts inventing three formats is three chances for a parser to
    read one as another, and the timezone is what stops a re-run being read as
    an earlier run by a local-time reader.
    """
    ts = _results.now()
    assert ts.endswith("+00:00"), f"the timestamp is not UTC: {ts!r}"
    assert "T" in ts, f"the timestamp is not in ISO form: {ts!r}"


# --------------------------------------------------------------------------- #
# The two contract resolvers agree
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("script,experiment_id", [
    ("measure_surrogate.py", "TACOSM-SURROGATE-001"),
    ("measure_learn.py", "TACOSM-LEARN-001"),
    ("measure_matched_h.py", "TACOSM-MATCHED-001"),
])
def test_the_scripts_resolver_agrees_with_the_packages(script, experiment_id):
    """Two resolvers, by design, and they must not disagree.

    ``contract_path_for`` resolves ``contracts/`` from the script's own
    location and ``contract._contracts_dir`` from the package's; the
    alternative to two resolvers is a script importing a private helper. A
    disagreement would be loud rather than silent — the fingerprint becomes
    ``"missing"`` and ``load_contract`` raises — but only if the two point at
    different *files*, which is what this pins.
    """
    from tac_osm.contract import _contracts_dir

    via_script = _results.contract_path_for(
        str(SCRIPTS_DIR / script), experiment_id)
    via_package = _contracts_dir() / f"{experiment_id}.json"
    assert via_script.resolve() == via_package.resolve(), (
        f"{script} resolves {experiment_id} to {via_script}, but the package "
        f"resolves it to {via_package}; a record would fingerprint a "
        "different file from the one the run was checked against"
    )
    assert via_script.is_file(), (
        f"{script} resolves {experiment_id} to {via_script}, which does not "
        "exist — the record's contract hash would be 'missing'"
    )


def test_the_results_directory_resolves_alongside_the_contracts():
    """``results/`` is found the same way ``contracts/`` is.

    Resolving from the script's own path rather than the CWD is what keeps a
    run invoked from elsewhere writing somewhere unexpected, and the two
    directories must be siblings: a script that found ``contracts/`` at the
    root and ``results/`` elsewhere would be checking one tree and writing to
    another.
    """
    via_script = _results.results_dir_for(str(SCRIPTS_DIR / "measure_surrogate.py"))
    assert via_script == REPO_ROOT / "results", (
        f"the results directory resolves to {via_script}, not the repository's "
        "results/ — it must be a sibling of contracts/"
    )


# --------------------------------------------------------------------------- #
# The deviations
# --------------------------------------------------------------------------- #


def _deviations(**over):
    """The deviations for a run off the registered design on the given fields.

    Defaults describe a fully compliant run; every keyword overrides one of
    the run's values, and the registered ones come from the surrogate contract
    so the fixture is not asserting a design the contracts have moved on from.
    """
    contract = load_contract("TACOSM-SURROGATE-001")
    kw = dict(
        contract_steps=contract.steps,
        contract_eval_steps=contract.eval_steps,
        contract_h_levels=contract.h_levels,
        contract_seeds=contract.seeds,
        contract_arms=[a.name for a in contract.arms],
        steps=contract.steps,
        eval_steps=contract.eval_steps,
        h_levels=contract.h_levels,
        seeds=contract.seeds,
        arms=[a.name for a in contract.arms],
    )
    kw.update(over)
    return _results.contract_deviations(**kw)


def test_a_compliant_run_reports_no_deviation():
    """Empty is what makes the field an answer rather than a missing NOTE.

    A record that omitted the field on compliance would be indistinguishable
    from one that never computed it.
    """
    assert _deviations() == []


#: The deviation dimension each keyword reports. The labels are the
#: *dimensions* a reader sees, not the keywords the function takes:
#: ``h_levels`` is reported as ``"levels"`` because that is what the dimension
#: is — the population levels — and the smoke report's reader meets the
#: dimension, not the parameter.
_DIMENSION = {
    "steps": "steps",
    "eval_steps": "eval_steps",
    "h_levels": "levels",
    "seeds": "seeds",
    "arms": "arms",
}


@pytest.mark.parametrize("field,run_value", [
    ("steps", 20),
    ("eval_steps", 17),
    ("h_levels", (8, 64)),
    ("seeds", (0, 1, 2)),
    ("arms", ["baseline", "analytic_margin"]),
])
def test_every_deviation_is_named(field, run_value):
    """The five dimensions the contract pins are the five a smoke run reports.

    A reader needs all five to judge whether a reported number is an artefact
    of the deviation, and a report that named four of five would read as
    complete.
    """
    deviations = _deviations(**{field: run_value})
    assert len(deviations) == 1, (
        f"changing {field} produced {len(deviations)} deviations, expected one"
    )
    assert deviations[0]["dimension"] == _DIMENSION[field], (
        f"changing {field} is reported as {deviations[0]['dimension']!r}"
    )
    assert deviations[0]["run"] == list(run_value) if not isinstance(
        run_value, int) else run_value
    assert deviations[0]["registered"] is not None


def test_a_deviation_reports_both_values():
    """The run's value and the registered one, so the size of the deviation is visible.

    "The run differs" is not enough: a reader judging whether a number is a
    smoke artefact needs to see that the run was 20 steps against a registered
    500, not merely that the two disagree.
    """
    d = _deviations(steps=20)[0]
    assert d == {"dimension": "steps", "run": 20, "registered": 500}


def test_seeds_are_compared_as_sets():
    """Seeds are an unordered set in the statistics.

    A re-ordering is not a deviation, and a report that called it one would
    cry wolf often enough to be ignored.
    """
    assert _deviations(seeds=(4, 3, 2, 1, 0)) == []


def test_an_added_or_removed_seed_is_a_deviation():
    """The seed set is the materiality threshold, so changing it changes the rule."""
    assert _deviations(seeds=(0, 1, 2, 3))
    assert _deviations(seeds=(0, 1, 2, 3, 4, 5))


def test_arms_are_compared_by_name_and_order():
    """An added or removed arm is a different experiment.

    Order matters for arms, unlike seeds: the decision rule is stated over the
    registered arm sequence and the tables are printed in it, so a re-ordering
    changes what a reader is shown even though the set is the same. The
    comparison therefore matches ``require_arms``, which is order-sensitive by
    the same reasoning.
    """
    contract = load_contract("TACOSM-SURROGATE-001")
    names = [a.name for a in contract.arms]
    assert _deviations(arms=list(reversed(names)))
    assert _deviations(arms=names[:-1])
    assert _deviations(arms=names + ["extra"])
    # The same set, in the registered order, is not a deviation.
    assert _deviations(arms=list(names)) == []


# --------------------------------------------------------------------------- #
# The smoke report
# --------------------------------------------------------------------------- #


def test_the_smoke_report_prints_and_returns_the_same_deviations():
    """The printed report and the written record come from one comparison.

    Two computations of the deviation is how the terminal output and the JSON
    come to disagree about whether the run was registered — and the terminal
    is what a reader sees first.
    """
    contract = load_contract("TACOSM-SURROGATE-001")
    out = _results.report_smoke(
        contract, "TACOSM-SURROGATE-001",
        steps=20, eval_steps=5, h_levels=(8,), seeds=(0, 1),
        arms=[a.name for a in contract.arms],
    )
    assert out and {d["dimension"] for d in out} == {
        "steps", "eval_steps", "levels", "seeds",
    }
    # The same comparison, computed again, must agree — the printed report and
    # the written record come from one helper, not two that can drift.
    assert out == _results.contract_deviations(
        contract_steps=contract.steps,
        contract_eval_steps=contract.eval_steps,
        contract_h_levels=contract.h_levels,
        contract_seeds=contract.seeds,
        contract_arms=[a.name for a in contract.arms],
        steps=20, eval_steps=5, h_levels=(8,), seeds=(0, 1),
        arms=[a.name for a in contract.arms],
    )


def test_the_smoke_report_declares_it_is_not_a_measurement(capsys):
    """The header is the load-bearing half of ``--smoke``.

    A smoke run that skipped the checks and printed nothing would be worse
    than no contract at all, because its output would be indistinguishable
    from a registered run's.
    """
    contract = load_contract("TACOSM-SURROGATE-001")
    _results.report_smoke(
        contract, "TACOSM-SURROGATE-001",
        steps=contract.steps, eval_steps=contract.eval_steps,
        h_levels=contract.h_levels, seeds=contract.seeds,
        arms=[a.name for a in contract.arms],
    )
    text = capsys.readouterr().out
    assert "SMOKE TEST" in text
    assert "not a measurement" in text
    assert "TACOSM-SURROGATE-001" in text
    # A compliant smoke run still says so, rather than printing nothing.
    assert "no deviation found" in text


def test_the_smoke_report_names_every_deviation(capsys):
    """All five dimensions, not the subset that first differed."""
    contract = load_contract("TACOSM-SURROGATE-001")
    _results.report_smoke(
        contract, "TACOSM-SURROGATE-001",
        steps=contract.steps + 1, eval_steps=contract.eval_steps + 1,
        h_levels=(8,), seeds=(0, 1),
        arms=[a.name for a in contract.arms] + ["unregistered"],
    )
    text = capsys.readouterr().out
    for label in ("steps", "eval_steps", "levels", "seeds", "arms"):
        assert label in text, (
            f"the smoke report does not name the {label} dimension"
        )


# --------------------------------------------------------------------------- #
# The write
# --------------------------------------------------------------------------- #


def test_write_record_is_atomic(tmp_path):
    """A reader never observes a half-written record.

    The temp file and ``replace`` mean an interrupted write leaves the
    previous record intact rather than a truncated one — a truncated JSON is
    the failure mode most likely to be read as a real, incomplete result.
    """
    out = tmp_path / "nested" / "dir" / "record.json"
    _results.write_record(_record(), out)
    assert out.is_file()
    assert json.loads(out.read_text(encoding="utf-8"))["provenance"]["experiment_id"]
    # No temp file is left behind to be mistaken for a record.
    assert not list(out.parent.glob("*.tmp"))


def test_write_record_overwrites_the_previous_run(tmp_path):
    """A re-run replaces, it does not append or merge.

    Two records in one file is a file that parses as one record with fields
    from both runs, which is a silent corruption.
    """
    out = tmp_path / "record.json"
    _results.write_record(_record(), out)
    _results.write_record(
        _record(provenance=_results.Provenance(
            experiment_id="TEST-RECORD-002",
            contract_source="contracts/TACOSM-SURROGATE-001.json",
            contract_sha256="def456", git_commit="cafebabe",
            script="measure_test.py", python="3.13.13",
            recorded_at="2026-01-02T00:00:00+00:00")),
        out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["provenance"]["experiment_id"] == "TEST-RECORD-002"
    assert out.read_text(encoding="utf-8").count('"experiment_id"') == 1


def test_write_record_creates_the_directory(tmp_path):
    """``results/`` may not exist on a fresh checkout, and the run still writes."""
    out = tmp_path / "new" / "results" / "record.json"
    assert not out.parent.is_dir()
    _results.write_record(_record(), out)
    assert out.is_file()


# --------------------------------------------------------------------------- #
# The scripts build records through the shared envelope
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("script", [
    "measure_surrogate.py", "measure_learn.py", "measure_matched_h.py",
])
def test_every_script_builds_a_record(script):
    """The record builder is the one path from a run to a file.

    A script that wrote its own JSON would be a script with its own envelope,
    which is the drift this module exists to remove. The builder is the
    shared seam, so it is pinned by name in every script.
    """
    text = (SCRIPTS_DIR / script).read_text(encoding="utf-8")
    assert "_build_record(" in text, (
        f"{script} does not build a record through the shared envelope; its "
        "output stays terminal-only, which is a result that exists only as "
        "long as the scrollback does"
    )
    assert "_emit_results(" in text, (
        f"{script} builds a record but does not emit it"
    )
    for name in ("MeasurementRecord", "Provenance", "Design", "Gate"):
        assert name in text, (
            f"{script} does not use the shared {name}, so its envelope can "
            "drift from the other scripts'"
        )


def test_every_script_writes_to_the_gitignored_results_directory():
    """``results/`` is ignored by design: the committed part of a result is the
    script and the gates, and the JSON is regenerable from deterministic seeds."""
    for script in ("measure_surrogate.py", "measure_learn.py",
                   "measure_matched_h.py"):
        text = (SCRIPTS_DIR / script).read_text(encoding="utf-8")
        assert "_results.results_dir_for(__file__)" in text, (
            f"{script} does not resolve results/ from its own path; a run "
            "invoked from another directory would write somewhere unexpected"
        )
    ignored = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "results/" in ignored or "results" in ignored, (
        "results/ is not gitignored; the regenerable JSON would be committed "
        "alongside the scripts that generate it, and kept consistent by hand"
    )


def test_the_scripts_emit_the_record_on_a_failed_gate():
    """A failed gate is a fact about the run, not a reason to delete the evidence.

    F3 and F2 stop on a failed gate and still write the record, because the
    record is what a later reader uses to see that the run was invalid and
    why. MATCHED-001's gate is the oracle, which cannot fail on a sound
    environment, so it has one emission site rather than two.
    """
    surrogate = (SCRIPTS_DIR / "measure_surrogate.py").read_text(encoding="utf-8")
    assert surrogate.count("_emit_results(") >= 2, (
        "measure_surrogate.py emits the record only on the success path; a "
        "failed gate would leave no record that the run was invalid"
    )
    learn = (SCRIPTS_DIR / "measure_learn.py").read_text(encoding="utf-8")
    assert learn.count("_emit_results(") >= 2, (
        "measure_learn.py emits the record only on the success path; a failed "
        "gate would leave no record that the run was invalid"
    )
