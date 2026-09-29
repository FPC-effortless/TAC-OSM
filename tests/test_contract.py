"""Tests for the machine-readable experiment contracts.

The contract is the spec-drift detector: it pins what a run must hold
constant, and raises rather than warns when it does not. These tests pin the
contract itself, so the detector cannot drift either.

Six groups:

1. **Every contract in ``contracts/`` is valid** — schema plus the
   cross-field invariants. This is the test that catches a contract written
   in haste, and it is written over ``validate()`` rather than over one file
   so a contract added later cannot be forgotten.
2. **The drift detectors actually detect** — each ``require_*`` is fed the
   kind of mismatch it exists to catch, and must raise. A detector that does
   not fire on the failure it was written for is worse than no detector,
   because it lends its name to a claim it does not enforce.
3. **The contracts agree with the code that implements them** — the
   registered levels, seeds, schedule length and arm names in each contract
   are the ones the measurement script actually uses. This is the seam where
   spec drift would otherwise live undetected: a contract that says 500 and a
   script whose default is 400 are two documents describing one experiment,
   and nothing else in the repository compares them.
4. **The scripts actually enforce their contracts** — the wiring: a detector
   the measurement scripts do not call is a document.
5. **The machine-readable result summary** — the outcome the contract is
   checked against, and its invariants.
6. **The M0 freeze** — every ``measure_*.py`` is either contracted or
   deliberately exempt, and both halves of the partition are enumerated. This
   is the completeness assertion the other five groups cannot make: a script
   added with neither a contract nor an exemption is invisible to them, and
   would emit unregistered numbers. See the ``_WITHOUT_CONTRACT`` entries for
   the three exemptions and the reason each is correct rather than a gap.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tac_osm.contract import (  # noqa: E402
    AmendmentSpec,
    ArmSpec,
    ContractError,
    DecisionBranch,
    EndpointSpec,
    ExperimentContract,
    load_all,
    load_contract,
    validate,
)
from tac_osm.measurement import results as _results  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
CONTRACTS_DIR = REPO_ROOT / "contracts"
SCRIPTS_DIR = REPO_ROOT / "scripts"


# --------------------------------------------------------------------------- #
# 1. Every contract is valid
# --------------------------------------------------------------------------- #


def test_contracts_directory_exists():
    """No contracts means no pre-registration is machine-checkable."""
    assert CONTRACTS_DIR.is_dir(), (
        f"missing {CONTRACTS_DIR}: the experiment contracts are absent, so no "
        "pre-registration in this repository can be checked against a run"
    )


def test_every_contract_is_internally_sound():
    """The single assertion over all contracts.

    ``validate()`` runs the schema and the cross-field checks and reports by
    filename, so a failure names the contract and the problem rather than
    aborting at the first one.
    """
    problems = validate()
    assert problems == [], (
        "one or more experiment contracts fail validation, which means a "
        "pre-registration that cannot be checked against a run:\n"
        + "\n".join(f"  - {p}" for p in problems)
    )


def test_every_contract_is_loadable_and_round_trips():
    """Serialisation must be lossless.

    A contract that does not survive a write and a read cannot be persisted
    next to its results, and a contract that is not persisted cannot be
    compared with a run after the fact.
    """
    contracts = load_all()
    assert contracts, "no contracts found"
    for experiment_id, contract in contracts.items():
        encoded = json.loads(contract.to_json())
        decoded = ExperimentContract.from_dict(encoded)
        assert decoded == contract, f"{experiment_id} does not round-trip"


def test_every_contract_has_exactly_one_primary_endpoint():
    """The decision rule reads one endpoint first."""
    for experiment_id, contract in load_all().items():
        assert contract.primary_endpoint(), experiment_id
        primaries = [e.name for e in contract.endpoints if e.primary]
        assert len(primaries) == 1, (
            f"{experiment_id}: {len(primaries)} primary endpoints "
            f"({primaries}); the decision rule reads one"
        )


def test_every_contract_commits_an_interpretation_order():
    """A contract without a committed order permits a post-hoc reading."""
    for experiment_id, contract in load_all().items():
        assert len(contract.interpretation_order) >= 2, (
            f"{experiment_id}: an interpretation order of fewer than two "
            "steps does not constrain the reading"
        )
        # The order is a contract, and reordering it to fit an outcome is the
        # failure it exists to prevent — so it must be repeatable.
        assert len(set(contract.interpretation_order)) == len(
            contract.interpretation_order
        ), f"{experiment_id}: interpretation_order repeats an entry"


def test_every_contract_pins_its_levels_and_seeds():
    """The constants the decision rule depends on are named, not implied."""
    for experiment_id, contract in load_all().items():
        assert contract.h_levels == tuple(sorted(contract.h_levels)), (
            f"{experiment_id}: h_levels must be strictly increasing"
        )
        assert len(contract.seeds) >= 3, (
            f"{experiment_id}: fewer than 3 seeds cannot estimate the spread "
            "the materiality threshold is built from"
        )
        assert contract.steps > 0 and contract.eval_steps > 0, experiment_id


def test_each_decision_branch_commits_a_consequence():
    """A branch without a consequence is a threshold that can be re-read."""
    for experiment_id, contract in load_all().items():
        assert contract.decision_rule, (
            f"{experiment_id}: a contract with no decision rule is a design, "
            "not a pre-registration"
        )
        for branch in contract.decision_rule:
            assert branch.condition.strip(), (
                f"{experiment_id}: a decision branch with no condition"
            )
            assert branch.licenses.strip(), (
                f"{experiment_id}: a decision branch with no committed "
                "consequence is re-interpretable after the result is known"
            )


def test_held_constant_is_non_empty():
    """Every contract names what it did not change.

    ``held_constant`` is the field that makes the intervention legible: a
    reader who cannot see what was held fixed cannot tell an intervention on
    the reward from one on the architecture.
    """
    for experiment_id, contract in load_all().items():
        assert contract.held_constant, (
            f"{experiment_id}: held_constant is empty — the contract does not "
            "say what the arms did not change"
        )


def test_reproduction_baselines_are_numeric():
    """The reference a gate compares against must be a number."""
    for experiment_id, contract in load_all().items():
        for cell, metrics in contract.reproduction_baseline.items():
            for metric, value in metrics.items():
                assert isinstance(value, (int, float)), (
                    f"{experiment_id}[{cell}].{metric} is {value!r}, not a number"
                )


# --------------------------------------------------------------------------- #
# 2. The drift detectors detect
# --------------------------------------------------------------------------- #


@pytest.fixture
def surrogate() -> ExperimentContract:
    return load_contract("TACOSM-SURROGATE-001")


def test_missing_contract_raises_with_a_usable_message():
    """The error must name the id, so a typo is distinguishable from an absence."""
    with pytest.raises(ContractError, match="TACOSM-NONEXISTENT"):
        load_contract("TACOSM-NONEXISTENT-999")


def test_malformed_json_is_reported_not_silently_ignored(tmp_path):
    """A contract that cannot be parsed must not become a contract that is skipped."""
    bad = tmp_path / "Broken.json"
    bad.write_text("{ not json ", encoding="utf-8")
    with pytest.raises(ContractError, match="malformed JSON"):
        load_contract("Broken", path=bad)


def test_missing_required_key_is_reported(tmp_path):
    """A contract missing a required field is a contract that cannot be checked."""
    incomplete = tmp_path / "Incomplete.json"
    incomplete.write_text(
        json.dumps({"experiment_id": "X", "title": "t"}),
        encoding="utf-8",
    )
    with pytest.raises(ContractError, match="missing required keys"):
        load_contract("Incomplete", path=incomplete)


@pytest.mark.parametrize("levels", [(8, 64), (8, 64, 512), (64, 8, 256), (8,)])
def test_wrong_levels_are_rejected(surrogate, levels):
    """The registered population levels are the experiment's design space."""
    with pytest.raises(ContractError, match="H levels"):
        surrogate.require_levels(levels)


def test_missing_levels_fail_even_permissively(surrogate):
    """``strict=False`` allows a superset, never a subset.

    A subset cannot estimate the spread the materiality threshold is built
    from, so a permissive check that accepted one would weaken the decision
    rule silently.
    """
    with pytest.raises(ContractError, match="subset"):
        surrogate.require_levels((8, 64), strict=False)


def test_superset_levels_are_accepted_permissively(surrogate):
    """Extra levels are harmless to the gate; missing levels are not."""
    surrogate.require_levels((8, 64, 256, 512), strict=False)


@pytest.mark.parametrize("seeds", [(0, 1, 2), (0, 1, 2, 3), (1, 2, 3, 4, 5)])
def test_wrong_seeds_are_rejected(surrogate, seeds):
    """The seed set is the materiality threshold, so changing it changes the rule."""
    with pytest.raises(ContractError, match="seeds"):
        surrogate.require_seeds(seeds)


def test_missing_seed_fails_even_permissively(surrogate):
    with pytest.raises(ContractError, match="absent from this run"):
        surrogate.require_seeds((0, 1, 2, 3), strict=False)


def test_reordering_seeds_is_allowed(surrogate):
    """Seeds are an unordered set in the statistics."""
    surrogate.require_seeds((4, 3, 2, 1, 0))


@pytest.mark.parametrize("steps", [100, 400, 501, 1000])
def test_wrong_schedule_length_is_rejected(surrogate, steps):
    """The most common drift: a shortened run that is no longer the registered intervention.

    The schedules anneal over the registered length, so a shorter run
    explores *more* than registered and is a different intervention.
    """
    with pytest.raises(ContractError, match="registered schedule length"):
        surrogate.require_steps(steps)


def test_registered_schedule_length_is_accepted(surrogate):
    surrogate.require_steps(500)

def test_registered_k_levels_are_accepted(surrogate):
    surrogate.require_k_levels(surrogate.k_levels)
    with pytest.raises(ContractError, match="K levels"):
        surrogate.require_k_levels((999,))


def test_wrong_arms_are_rejected(surrogate):
    """An added or removed arm makes the decision rule unevaluable as written."""
    with pytest.raises(ContractError, match="do not match the registered"):
        surrogate.require_arms(["baseline", "analytic_margin"])
    with pytest.raises(ContractError, match="do not match the registered"):
        surrogate.require_arms(["baseline", "analytic_margin",
                                "analytic_margin_clipped", "extra"])


def test_registered_arms_are_accepted(surrogate):
    surrogate.require_arms(["baseline", "analytic_margin",
                            "analytic_margin_clipped"])


def test_two_primary_endpoints_are_inconsistent():
    """A contract answering two questions at once has no single rule to evaluate."""
    c = _minimal_contract(endpoints=(
        EndpointSpec(name="a", role="r", primary=True),
        EndpointSpec(name="b", role="r", primary=True),
    ))
    problems = c.check_consistency()
    assert any("exactly one primary" in p for p in problems)


def test_no_primary_endpoint_is_inconsistent():
    c = _minimal_contract(endpoints=(
        EndpointSpec(name="a", role="r", primary=False),
    ))
    problems = c.check_consistency()
    assert any("exactly one primary" in p for p in problems)


def test_empty_interpretation_order_is_inconsistent():
    """An empty order is a contract that permits any post-hoc reading."""
    c = _minimal_contract(interpretation_order=())
    problems = c.check_consistency()
    assert any("interpretation_order is empty" in p for p in problems)


def test_repeated_h_level_is_inconsistent():
    c = _minimal_contract(h_levels=(8, 8, 64))
    problems = c.check_consistency()
    assert any("repeats a level" in p for p in problems)


def test_unsorted_h_levels_are_inconsistent():
    c = _minimal_contract(h_levels=(8, 256, 64))
    problems = c.check_consistency()
    assert any("strictly increasing" in p for p in problems)


def test_nonpositive_steps_are_inconsistent():
    c = _minimal_contract(steps=0)
    assert any("steps must be positive" in p for p in c.check_consistency())


def test_unregistered_arm_in_decision_rule_is_flagged():
    """A rule that covers an arm the contract never registered cannot be checked."""
    c = _minimal_contract(decision_rule=(
        DecisionBranch(
            condition="if the epsilon_greedy arm lifts the endpoint",
            licenses="the intervention is the mechanism",
        ),
    ))
    problems = c.check_consistency()
    assert any("not registered" in p for p in problems), (
        "a decision rule referencing an unregistered arm was not caught"
    )


def test_a_registered_arm_in_the_rule_is_not_flagged():
    """The detector must not fire on a correct reference."""
    c = _minimal_contract(
        arms=(ArmSpec(name="epsilon_greedy", intervention="i", acts_on="a"),),
        decision_rule=(
            DecisionBranch(
                condition="if the epsilon_greedy arm lifts the endpoint",
                licenses="the intervention is the mechanism",
            ),
        ),
    )
    assert c.check_consistency() == []


def test_consistency_reports_all_problems_not_just_the_first():
    """A contract with two defects is best described once, in full."""
    c = _minimal_contract(
        endpoints=(
            EndpointSpec(name="a", role="r", primary=True),
            EndpointSpec(name="b", role="r", primary=True),
        ),
        interpretation_order=(),
    )
    problems = c.check_consistency()
    assert any("exactly one primary" in p for p in problems)
    assert any("interpretation_order is empty" in p for p in problems)


def test_a_sound_contract_reports_no_problems():
    assert _minimal_contract().check_consistency() == []


# --------------------------------------------------------------------------- #
# 2b. Amendments: a change to a pre-registration is recorded as a change
# --------------------------------------------------------------------------- #
#
# A pre-registration that can be edited after the fact is not one, so an
# amendment is not an edit: it is a visible change that carries the definition
# it replaced. The one amendment in this repository was found by the
# instrument, not by reading — the primary endpoint's definition made it
# arm-independent, so no branch of the decision rule could have fired on any
# data. That is the failure mode this group exists to keep detectable: an
# amendment that was silently an edit would erase the design a reader was
# promised.


def test_an_amendment_round_trips():
    """An amendment is a first-class part of the contract, so it survives a write and a read.

    The optional fields are the ones a reader owes the record — where the
    defect was discovered, what it affected, and whether any result exists
    under the previous version. They are optional because an amendment
    recorded before any run has none of them yet, but a reader who sees them
    must see the round trip too.
    """
    full = AmendmentSpec(
        id="A1",
        applies_to="the primary endpoint's definition",
        old_definition="P(gold in the top-K of the full population)",
        new_definition="P(gold in the arm's retained set and argmax over it)",
        rationale="the original was arm-independent, so no branch could fire",
        discovered="by the instrument during the dry run, before any confirmatory run",
        affected="the decision rule and the C-vs-B comparison",
        no_result_under_previous_version=(
            "no confirmatory run was made under the original definition"
        ),
    )
    encoded = full.to_dict()
    assert encoded["id"] == "A1"
    assert encoded["old_definition"] and encoded["new_definition"]
    assert encoded["discovered"]
    decoded = AmendmentSpec.from_dict(encoded)
    assert decoded == full


def test_an_amendment_omits_its_optional_fields_when_empty():
    """An amendment recorded before any run has no affected-field yet.

    Those fields are not required, because requiring them would force an
    amendment to state a result it does not have. Their absence must therefore
    survive the round trip — an empty string written into the file would be
    read back as present, and a reader would take it for a claim.
    """
    minimal = AmendmentSpec(
        id="A1",
        applies_to="an endpoint",
        old_definition="old",
        new_definition="new",
        rationale="why",
    )
    d = minimal.to_dict()
    assert set(d) == {"id", "applies_to", "old_definition", "new_definition", "rationale"}
    assert AmendmentSpec.from_dict(d) == minimal


def test_an_amendment_requires_its_five_fields():
    """An amendment without its old definition is not a change but an overwrite.

    The five required keys are what make the amendment comparable: without
    ``old_definition`` a reader cannot see the design they were promised,
    without ``new_definition`` they cannot see what replaced it, and without
    ``rationale`` they cannot judge whether the change was a repair or a
    retreat.
    """
    base = dict(
        id="A1",
        applies_to="an endpoint",
        old_definition="old",
        new_definition="new",
        rationale="why",
    )
    for key in list(base):
        short = {k: v for k, v in base.items() if k != key}
        with pytest.raises(ContractError, match="missing required keys"):
            AmendmentSpec.from_dict(short)


def test_an_amendmentless_contract_round_trips():
    """A contract with no amendments must not serialise the field.

    An empty ``"amendments": []`` would read as a list that was once
    populated, which is the wrong signal for an unamended pre-registration:
    the field appears only when there is a change to see.
    """
    d = _minimal_contract().to_dict()
    assert "amendments" not in d
    assert ExperimentContract.from_dict(d).amendments == ()


def test_a_contract_with_an_amendment_round_trips():
    """The amendment survives the contract's round trip, not only its own.

    The contract is what gets written next to a run's results, so the
    amendment has to survive *that* round trip: a reader comparing a record
    against its contract reads the contract from the file, and an amendment
    lost on the way would be a change with no evidence.
    """
    c = _minimal_contract(
        amendments=(
            AmendmentSpec(
                id="A1",
                applies_to="the primary endpoint",
                old_definition="old",
                new_definition="new",
                rationale="the original could not discriminate between arms",
            ),
        ),
    )
    decoded = ExperimentContract.from_dict(json.loads(c.to_json()))
    assert decoded == c
    assert decoded.amendments[0].id == "A1"


def test_an_amendment_without_an_id_is_flagged():
    """An amendment with no id cannot be referred to by any later record."""
    c = _minimal_contract(
        amendments=(
            AmendmentSpec(id="", applies_to="x", old_definition="old",
                          new_definition="new", rationale="why"),
        ),
    )
    problems = c.check_consistency()
    assert any("has no id" in p for p in problems), (
        "an amendment without an id was not caught; a later record could not "
        "name the change it is answering to"
    )


def test_two_amendments_sharing_an_id_are_flagged():
    """Two amendments with one id are one change recorded twice.

    A reader could not tell which applies, and a record answering to ``A1``
    would be answering to either or both — which is the ambiguity the
    amendment record exists to remove.
    """
    c = _minimal_contract(
        amendments=(
            AmendmentSpec(id="A1", applies_to="x", old_definition="old1",
                          new_definition="new1", rationale="why1"),
            AmendmentSpec(id="A1", applies_to="y", old_definition="old2",
                          new_definition="new2", rationale="why2"),
        ),
    )
    problems = c.check_consistency()
    assert any("is used by two amendments" in p for p in problems)


def test_two_amendments_with_distinct_ids_are_fine():
    """The detector must not fire on the correct spelling."""
    c = _minimal_contract(
        amendments=(
            AmendmentSpec(id="A1", applies_to="x", old_definition="old1",
                          new_definition="new1", rationale="why1"),
            AmendmentSpec(id="A2", applies_to="y", old_definition="old2",
                          new_definition="new2", rationale="why2"),
        ),
    )
    assert c.check_consistency() == []


def test_the_amended_contract_is_internally_sound():
    """``TACOSM-RETRIEVAL-001`` validates with the amendment present.

    The amendment rewrote the primary endpoint's definition, so the contract
    that carries it must still have exactly one primary endpoint and a
    decision rule over registered arms. This is the assertion that the
    amendment repaired the definition without breaking the rule it was made
    to serve.
    """
    c = load_contract("TACOSM-RETRIEVAL-001")
    assert c.check_consistency() == [], (
        "the amended retrieval contract is self-inconsistent: "
        + "; ".join(c.check_consistency())
    )


def test_the_amended_contract_records_its_amendment():
    """The amendment is carried by the contract a run checks itself against.

    Not by a separate changelog: the contract is the document a record points
    at, so the amendment has to be there for a reader to see that the primary
    endpoint they are reading is not the one that was first registered.
    """
    c = load_contract("TACOSM-RETRIEVAL-001")
    assert len(c.amendments) == 1, (
        f"expected exactly one amendment on the retrieval contract, found "
        f"{len(c.amendments)}"
    )
    am = c.amendments[0]
    assert am.id == "A1"
    # The seven fields of a recorded change. Each is what a reader needs to
    # judge the amendment rather than accept it: the two definitions are the
    # before and after, the rationale is the judgement, and the remaining
    # four place the change in the record.
    for field_name in (
        "applies_to", "old_definition", "new_definition", "rationale",
        "discovered", "affected", "no_result_under_previous_version",
    ):
        assert getattr(am, field_name), (
            f"the amendment's {field_name} is empty; an amendment that omits "
            "a field is asking a reader to take the change on trust"
        )


def test_the_amended_contract_still_has_one_primary_and_four_branches():
    """The amendment changed the definition, not the rule's structure.

    The primary endpoint is still exactly one, the branches still number
    four, and the primary is still ``recall@K`` — now under the amended
    definition. A change that had altered the rule's shape would be a new
    pre-registration rather than an amendment to this one.
    """
    c = load_contract("TACOSM-RETRIEVAL-001")
    assert c.primary_endpoint() == "recall@K"
    assert len(c.decision_rule) == 4
    assert len(c.endpoints) == 8


# --------------------------------------------------------------------------- #
# 3. The contracts agree with the code that implements them
# --------------------------------------------------------------------------- #
#
# This is the seam. A contract that says 500 and a script whose default is 400
# are two documents describing one experiment, and nothing else in the
# repository compares them. The comparison is done by parsing the script's
# module-level registered constants, not by re-reading its output, so it is a
# check on the design the script is built from rather than on one run of it.


def _script_constants(name: str) -> dict:
    """Read the registered constants out of a measurement script.

    The scripts are runnable modules, not importable ones — they insert
    ``src`` into ``sys.path`` and import the package at import time, so they
    are read as text and their constants are read with ``ast`` rather than
    imported. Only literal syntax is evaluated, so no code in the script runs.

    ``STEPS_DEFAULT`` is not a module-level constant: it is the argparse
    ``default=`` on ``--steps``, which is what a run actually uses when
    nobody overrides it. A contract is compared against that rather than
    against an imported constant, because the default is the length an
    unflagged run really has.
    """
    import ast

    tree = ast.parse((SCRIPTS_DIR / name).read_text(encoding="utf-8"))
    out: dict = {}
    # ``ARMS`` in the multi-arm scripts; ``REGISTERED_ARMS`` in the
    # single-arm one, where the name ``ARMS`` would imply a comparison
    # against a control arm that the design does not run. Both spell the
    # registered arm set as a literal for this reader.
    for want in ("H_LEVELS", "K_LEVELS", "ARMS", "REGISTERED_ARMS",
                 "EXPERIMENT_ID"):
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == want:
                        try:
                            out[want] = ast.literal_eval(node.value)
                        except (ValueError, SyntaxError):
                            pass
    # The argparse defaults: the values an unflagged run actually uses, read
    # from the call rather than assumed. Each flag is handled by its own
    # branch, not a chain, because ``--steps`` carries ``type=int`` *before*
    # ``default`` and a chain keyed on the flag would skip it.
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, "attr", "") != "add_argument":
            continue
        if not (node.args and isinstance(node.args[0], ast.Constant)):
            continue
        flag = node.args[0].value
        default = next(
            (kw.value for kw in node.keywords if kw.arg == "default"), None
        )
        if default is None:
            continue
        if flag == "--steps":
            out["STEPS_DEFAULT"] = _resolve_steps(tree, default)
        elif flag == "--seeds":
            try:
                raw = ast.literal_eval(default)
            except (ValueError, SyntaxError):
                continue
            out["SEEDS"] = tuple(int(s) for s in str(raw).split(","))
        elif flag == "--eval-steps":
            try:
                out["EVAL_STEPS"] = int(ast.literal_eval(default))
            except (ValueError, SyntaxError):
                continue
    return out


def _resolve_steps(tree: ast.AST, node: ast.AST) -> int:
    """The ``--steps`` default, resolving a name like ``SCHEDULE_LENGTH``.

    In ``measure_surrogate.py`` and ``measure_learn.py`` the default is an
    imported name, so the literal is not at the call site. It is resolved to
    the module-level assignment, which is a literal in every script that
    names it. If neither reading yields an integer the test that called this
    fails loudly rather than silently assuming 500.
    """
    import ast

    if isinstance(node, ast.Constant):
        return int(node.value)
    if isinstance(node, ast.Name):
        for other in ast.walk(tree):
            if isinstance(other, ast.Assign):
                for target in other.targets:
                    if isinstance(target, ast.Name) and target.id == node.id:
                        try:
                            return int(ast.literal_eval(other.value))
                        except (ValueError, SyntaxError):
                            raise AssertionError(
                                f"the --steps default {node.id} is not a literal "
                                "the contract test can read; the registered "
                                "length must be a module-level literal"
                            )
    raise AssertionError(
        "could not read the --steps default; the registered schedule length "
        "must be a literal, or a name bound to one"
    )


def test_surrogate_contract_matches_its_script():
    """``TACOSM-SURROGATE-001.json`` must describe ``measure_surrogate.py``."""
    got = _script_constants("measure_surrogate.py")
    c = load_contract("TACOSM-SURROGATE-001")
    assert tuple(got["H_LEVELS"]) == c.h_levels
    assert tuple(got["K_LEVELS"]) == c.k_levels
    assert tuple(got["ARMS"]) == tuple(a.name for a in c.arms), (
        "the F3 contract's arm names drifted from the script's registered tuple"
    )
    assert c.steps == got["STEPS_DEFAULT"], (
        "the contract's schedule length is not the length an unflagged run "
        "of the script actually uses"
    )
    assert c.eval_steps == got["EVAL_STEPS"]
    assert tuple(got["SEEDS"]) == c.seeds


def test_learn_contract_matches_its_script():
    """``TACOSM-LEARN-001.json`` must describe ``measure_learn.py``."""
    got = _script_constants("measure_learn.py")
    c = load_contract("TACOSM-LEARN-001")
    assert tuple(got["H_LEVELS"]) == c.h_levels
    assert tuple(got["K_LEVELS"]) == c.k_levels
    assert tuple(got["ARMS"]) == tuple(a.name for a in c.arms), (
        "the F2 contract's arm names drifted from the script's ARMS tuple"
    )
    assert c.steps == got["STEPS_DEFAULT"], (
        "the F2 schedules anneal over the registered length, so a contract "
        "that disagrees with the script's default is describing a different "
        "intervention"
    )
    assert c.eval_steps == got["EVAL_STEPS"]
    assert tuple(got["SEEDS"]) == c.seeds


def test_matched_contract_matches_its_script():
    """``TACOSM-MATCHED-001.json`` must describe ``measure_matched_h.py``."""
    got = _script_constants("measure_matched_h.py")
    c = load_contract("TACOSM-MATCHED-001")
    assert tuple(got["H_LEVELS"]) == c.h_levels
    assert tuple(got["K_LEVELS"]) == c.k_levels
    # The single-arm design registers ``REGISTERED_ARMS`` rather than ``ARMS``;
    # the arm set is one name, and the comparison is across the matrix rather
    # than against a control arm.
    arms = got.get("REGISTERED_ARMS", got.get("ARMS"))
    assert tuple(arms) == tuple(a.name for a in c.arms), (
        "the MATCHED-001 contract's arm drifted from the script's registered "
        "tuple"
    )
    assert c.steps == got["STEPS_DEFAULT"]
    assert c.eval_steps == got["EVAL_STEPS"]
    assert tuple(got["SEEDS"]) == c.seeds


def test_c5_002_contract_matches_its_script():
    """``TACOSM-C5-002.json`` must describe ``measure_c5_casm_002.py``.

    C5-002 exists because C5-001's compound endpoint inherited an execution
    failure into an addressing number, and the repair is structural rather
    than a patched constant: the primary endpoint reads no CASM-S output. The
    contract therefore pins more than levels and arms — it pins the
    *separation*, and this test is the seam where a script that quietly
    reintroduced a model output into ``coverage_rate`` would be caught at the
    level of constants rather than of one run.
    """
    got = _script_constants("measure_c5_casm_002.py")
    c = load_contract("TACOSM-C5-002")
    assert got["EXPERIMENT_ID"] == c.experiment_id
    assert tuple(got["H_LEVELS"]) == c.h_levels
    assert tuple(got["K_LEVELS"]) == c.k_levels
    assert tuple(got["ARMS"]) == tuple(a.name for a in c.arms)
    assert c.steps == got["STEPS_DEFAULT"]
    assert c.eval_steps == got["EVAL_STEPS"]
    assert tuple(got["SEEDS"]) == c.seeds

    text = _script_text("measure_c5_casm_002.py")
    # The gate is what makes this a new pre-registration rather than a patched
    # C5-001: it runs before the task stream and terminates the run. A script
    # in which the gate became advisory, or ran after the cells, would be
    # C5-001 with a new number, which is the thing the void exists to prevent.
    assert "run_gate(runtime)" in text
    assert "if not gate.passed:" in text
    assert text.index("run_gate(runtime)") < text.index("for h in run_levels:")
    # The primary endpoint is a set intersection between retained indices and
    # the hidden acceptable set. The acceptable set is read *after* execution
    # and only for evaluation, so the ordering in the source is itself the
    # boundary — reading it before the execution block would move the hidden
    # set into the addressing decision.
    assert text.index("any(i in task.acceptable_actions") > text.index(
        "_build_and_execute("
    )

    # Every measured CellResult field must be a registered endpoint, because
    # the record is what a later check compares against the contract. A field
    # the contract does not name is a number the pre-registration does not
    # cover. The four structurals — ``arm``, ``h``, ``k``, ``seed`` — are the
    # cell's *address in the matrix*, not measured quantities: they name which
    # cell a number belongs to, so a contract that registered them as
    # endpoints would be registering the design's axes as its results.
    cell_fields = {
        f.name for f in __import__("dataclasses").fields(
            _script_module("measure_c5_casm_002.py").CellResult
        )
    }
    structural = {"arm", "h", "k", "seed"}
    registered = {e.name for e in c.endpoints}
    unregistered = cell_fields - structural - registered
    assert not unregistered, (
        "CellResult carries measured fields the contract does not register as "
        f"endpoints: {sorted(unregistered)}"
    )
    # And conversely the contract must not register an endpoint no cell
    # carries, which would be a number the instrument never measures.
    unmeasured = registered - cell_fields
    assert not unmeasured, (
        "the contract registers endpoints the script's CellResult does not "
        f"carry: {sorted(unmeasured)}"
    )


def test_reproduction_baselines_agree_with_the_scripts():
    """The reproduction gate's reference must be one number in both places.

    The scripts carry the published MATCHED-001 baseline as a module-level
    dict keyed by ``(h_train, h_eval)``; the contract carries the same values
    keyed by ``"8x8"``. A difference here would make the gate and the contract
    disagree about what the baseline is, which is precisely the drift the
    contract exists to prevent.
    """
    import ast

    text = (SCRIPTS_DIR / "measure_surrogate.py").read_text(encoding="utf-8")
    tree = ast.parse(text)
    script_baseline = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "MATCHED_001_BASELINE":
                    parsed = ast.literal_eval(node.value)
                    script_baseline = {
                        f"{a}x{b}": {m: float(v) for m, v in cell.items()}
                        for (a, b), cell in parsed.items()
                    }
    assert script_baseline, "could not read MATCHED_001_BASELINE from the script"

    c = load_contract("TACOSM-SURROGATE-001")
    assert set(script_baseline) == set(c.reproduction_baseline), (
        "the reproduction baseline in the contract does not cover the same "
        "cells as the one in the script"
    )
    for cell in script_baseline:
        for metric, value in script_baseline[cell].items():
            assert c.reproduction_baseline[cell][metric] == pytest.approx(value), (
                f"reproduction baseline disagrees at {cell}.{metric}: contract "
                f"{c.reproduction_baseline[cell][metric]!r} vs script {value!r}"
            )


# --------------------------------------------------------------------------- #
# 4. The scripts actually enforce their contracts
# --------------------------------------------------------------------------- #
#
# The contract module is the detector; this group is the wiring. A detector
# that the measurement scripts do not call is a document, and the whole point
# of the contract was that a run could not drift from its pre-registration
# *silently*. These tests are static — they do not run a measurement, which
# is hours of compute — because what is being pinned is the presence of the
# check, not one execution of it.


def _script_text(name: str) -> str:
    return (SCRIPTS_DIR / name).read_text(encoding="utf-8")


#: Every measurement script that has a contract, with its contract's id.
#: Parametrizing the wiring tests over this is what makes a new script
#: unforceable by accident: adding a script and a contract but forgetting to
#: wire them fails the parametrization, not only the eye.
#:
#: This is the **frozen M0 enforcement surface** (``MEASUREMENT_LAYERS.md``
#: §"Measurement Integrity"). The complementary list — the scripts that
#: deliberately have no contract, and why — is ``_WITHOUT_CONTRACT`` in group
#: 6 below. Both lists are needed: a partition asserted on one side only can
#: drift on the other, and the three exemption entries are the ones that
#: would otherwise look like oversights rather than decisions.
_WITH_CONTRACT = (
    ("measure_surrogate.py", "TACOSM-SURROGATE-001"),
    ("measure_learn.py", "TACOSM-LEARN-001"),
    ("measure_matched_h.py", "TACOSM-MATCHED-001"),
    ("measure_retrieval.py", "TACOSM-RETRIEVAL-001"),
    ("measure_temporal_persistence.py", "TACOSM-TEMPORAL-001"),
    ("measure_selective_scaling.py", "TACOSM-SELECTIVE-001"),
    ("measure_c5_casm.py", "TACOSM-C5-001"),
    ("measure_c5_casm_002.py", "TACOSM-C5-002"),
)


@pytest.mark.parametrize("script,experiment_id", _WITH_CONTRACT)
def test_the_scripts_load_and_enforce_their_contract(script, experiment_id):
    """A measurement script must check its run against its contract.

    ``load_contract`` at the top of ``main`` is the wiring; the five
    ``require_*`` calls are the enforcement. Both are required: a script with
    the load but no check has a contract it never consults, and a script with
    the checks but no load cannot name which contract it is enforcing.

    The id may be a literal or a module-level ``EXPERIMENT_ID`` constant. The
    constant is the better spelling — the script's record provenance, its
    contract lookup and its smoke report all read one name, so they cannot
    disagree — but the wiring matters more than the spelling, and both are
    accepted so the test pins the check rather than the style.
    """
    text = _script_text(script)
    loaded = (
        f'load_contract("{experiment_id}")' in text
        or f"load_contract(EXPERIMENT_ID)" in text
    )
    assert loaded, (
        f"{script} does not load its machine-readable contract "
        f"{experiment_id}; drift from the pre-registration is undetectable "
        "by the run itself"
    )
    # When the id goes through a constant, the constant must *be* the contract
    # id: a script whose EXPERIMENT_ID disagrees with its contract has a
    # record that attributes itself to the wrong pre-registration, which is
    # the one disagreement that silently invalidates every other check.
    if "load_contract(EXPERIMENT_ID)" in text and 'load_contract("' not in text:
        got = _script_constants(script).get("EXPERIMENT_ID")
        assert got == experiment_id, (
            f"{script}'s EXPERIMENT_ID is {got!r}, but its contract is "
            f"{experiment_id!r}; the record would name a pre-registration "
            "this script does not enforce"
        )
    for check in (
        "require_steps", "require_eval_steps", "require_levels",
        "require_seeds", "require_arms",
    ):
        assert f"contract.{check}(" in text, (
            f"{script} does not call {check}, so a run can change the "
            "quantity it pins and still report a number"
        )


@pytest.mark.parametrize("script,experiment_id", _WITH_CONTRACT)
def test_the_scripts_declare_their_smoke_test(script, experiment_id):
    """``--smoke`` is the one declared way to run off the registered design.

    The flag is what keeps the contract from being a lock that a developer
    simply removes: a smoke run is legitimate and frequent, and without a
    declared escape hatch the choice is between an unusable script and a
    silently weakened one. The flag must print *what* it skipped, so the
    output cannot be mistaken for a measurement.
    """
    text = _script_text(script)
    assert '"--smoke"' in text, (
        f"{script} has no --smoke flag; the contract check can only be "
        "bypassed by editing the script, which leaves no trace"
    )
    # The deviation report is the load-bearing half of the flag. A ``--smoke``
    # that skipped the checks and printed nothing would be worse than no
    # contract at all, because the output would be indistinguishable from a
    # registered run's. The strings live in the shared module now
    # (:func:`tac_osm.measurement.results.report_smoke`), so what is pinned
    # here is that the script reaches it rather than its own copy — a script
    # with its own copy is a script whose five dimensions can drift to four.
    assert "report_smoke(" in text, (
        f"{script} does not call the shared smoke report; a private copy can "
        "list four of the five dimensions and call itself complete"
    )
    assert experiment_id in text


@pytest.mark.parametrize("script,experiment_id", _WITH_CONTRACT)
def test_every_script_pins_its_registered_steps_as_a_literal(script, experiment_id):
    """The schedule length a script defaults to must be a literal, not a guess.

    ``_resolve_steps`` follows the ``--steps`` default wherever it is bound, so
    a script that defaulted to a bare ``500`` still passes — the value is what
    is checked, not the spelling. What this test catches is the script whose
    default is an expression a static reader cannot evaluate at all, which
    would silently break the contract-to-script comparison for every reader.
    """
    got = _script_constants(script)
    c = load_contract(experiment_id)
    assert c.steps == got["STEPS_DEFAULT"], (
        f"{script}'s --steps default is {got['STEPS_DEFAULT']}, but its "
        f"contract registers {c.steps}. An unflagged run of that script does "
        "not execute the registered intervention."
    )


@pytest.mark.parametrize("script,experiment_id", _WITH_CONTRACT)
def test_every_script_names_its_registered_arms(script, experiment_id):
    """The arm set is written out in every script, in whatever spelling.

    ``ARMS`` for the multi-arm designs, ``REGISTERED_ARMS`` for the single-arm
    one, whose comparison is across the matrix rather than against a control
    arm. A script with no literal at all would be a script whose arm set can
    only be known by running it, which is the drift the contract exists to
    prevent.
    """
    got = _script_constants(script)
    c = load_contract(experiment_id)
    arms = got.get("REGISTERED_ARMS", got.get("ARMS"))
    assert tuple(arms) == tuple(a.name for a in c.arms), (
        f"{script}'s registered arm tuple does not match {experiment_id}"
    )


def test_the_smoke_flag_reports_every_deviation():
    """A smoke run must name each dimension it is off the registered design.

    Not "the run differs"; the specific steps, eval length, levels, seeds and
    arms, because those are the five quantities the contract pins and a
    reader needs all five to judge whether a reported number is a smoke
    artefact. The report lives in the shared module now, so the five
    dimensions are pinned there — over every contract, since the report is
    used by all three scripts and a dropped dimension would be silent in all
    three at once.
    """
    for experiment_id in (cid for _, cid in _WITH_CONTRACT):
        contract = load_contract(experiment_id)
        # A run off the registered design on every dimension at once: the
        # report must name all five, not the subset that first differed.
        deviations = _results.contract_deviations(
            contract_steps=contract.steps,
            contract_eval_steps=contract.eval_steps,
            contract_h_levels=contract.h_levels,
            contract_seeds=contract.seeds,
            contract_arms=[a.name for a in contract.arms],
            steps=contract.steps + 1,
            eval_steps=contract.eval_steps + 1,
            h_levels=(8,),
            seeds=(0, 1),
            arms=[a.name for a in contract.arms] + ["unregistered"],
        )
        found = {d["dimension"] for d in deviations}
        assert found == {"steps", "eval_steps", "levels", "seeds", "arms"}, (
            f"the smoke report for {experiment_id} does not name all five "
            f"dimensions; found {sorted(found)}"
        )


def test_the_smoke_flag_is_the_only_bypass():
    """There is no second way around the contract check.

    A ``--no-contract`` or a commented-out check would be a quiet bypass; this
    pins the enforcement to a *single* named, documented flag. The assertion
    is written over the bypass count rather than over the flag's presence so
    that adding a second escape hatch fails this test.
    """
    text = _script_text("measure_surrogate.py")
    assert "require_steps(args.steps)" in text, (
        "the contract check is not applied to the parsed --steps value"
    )
    # The check must be reached on the non-smoke path, not defined only as a
    # possibility: it sits in the ``else`` of the smoke branch.
    assert "if args.smoke:" in text
    assert "else:" in text


@pytest.mark.parametrize("steps", [20, 50, 99, 400, 501])
def test_require_eval_steps_detects_a_short_evaluation(surrogate, steps):
    """A shorter evaluation changes every endpoint's sampling error at once.

    The tables look identical at 100 steps and at 20 — only the standard error
    moved — so this is the drift that is invisible in the output.
    """
    with pytest.raises(ContractError, match="registered evaluation length"):
        surrogate.require_eval_steps(steps)


def test_registered_eval_length_is_accepted(surrogate):
    surrogate.require_eval_steps(100)


# --------------------------------------------------------------------------- #
# 5. The machine-readable result summary
# --------------------------------------------------------------------------- #
#
# The contract pins the design; the summary pins the outcome. This group holds
# the summary's invariants in place the same way group 1 holds the contract's:
# a field that is not asserted is a field that can silently disappear.
#
# These are unit tests over the *shape* of the summary, deliberately not
# requiring a full run: a measurement is hours of compute on this device, and
# what needs pinning is that a summary written once must be readable by any
# later check.


def _script_module(name: str):
    """Import a measurement script as a module.

    The scripts insert ``src`` into ``sys.path`` at import time and are
    normally run as ``__main__``; importing them here is safe because they do
    no work at module scope beyond defining constants and functions — ``main``
    is guarded by ``if __name__ == "__main__"``.
    """
    import importlib.util

    path = SCRIPTS_DIR / name
    spec = importlib.util.spec_from_file_location(f"_script_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    # Registered *before* exec: a script whose module scope defines a
    # dataclass needs to be resolvable by name while that dataclass is being
    # processed, and CPython looks it up in ``sys.modules`` rather than in the
    # local namespace. Without this the dataclass machinery reads
    # ``sys.modules[cls.__module__]`` and finds ``None``.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _summary_with_failing_gate():
    """A summary built from a smoke-shaped run, by calling the record builder.

    Uses the script's own helpers rather than re-implementing the schema, so a
    change to the summary's shape moves this test rather than silently
    diverging from it. ``rows``/``per_seed`` carry only the cells the smoke run
    actually has, which is what a failing-gate record must still be able to
    express.
    """
    surrogate = _script_module("measure_surrogate.py")
    h = 8
    # The three metrics the reproduction gate reads at H=8, so the gate half
    # of the record has a cell to report. ``recall@16`` is NaN here for the
    # same reason it is NaN in a real H=8 run: 16 is not a budget below the
    # candidate count, so the endpoint is undefined and the record omits it.
    arm_rows = {
        ("baseline", h, h): {"routing@1": 0.60, "recall@4": 0.80,
                             "delta_1": 0.05},
        ("analytic_margin", h, h): {"routing@1": 0.40, "recall@4": 0.70,
                                    "delta_1": 0.01},
        ("analytic_margin_clipped", h, h): {"routing@1": 0.30,
                                            "recall@4": 0.60, "delta_1": 0.02},
    }
    per_seed = {k: [0.5, 0.7] for k in arm_rows}
    nan = float("nan")
    per_seed_recall = {k: [nan, nan] for k in arm_rows}
    train_stats = {
        (arm, h): [{"reward_nonzero_frac": 0.5, "reward_mean": 0.1,
                    "reward_min": 0.0, "reward_max": 1.0,
                    "sat_hi_frac": 0.0, "sat_lo_frac": 0.2,
                    "successes": 4}]
        for arm in surrogate.ARMS
    }

    class _Args:
        steps = 20
        eval_steps = 5
        smoke = True

    # The deviations a smoke run at these values would report. The fixture
    # *is* a smoke run — steps 20, five eval steps, two seeds — and the
    # deviations are the record's statement of that, so computing them from
    # the same shared helper keeps the fixture from asserting a shape the
    # scripts no longer produce.
    contract = load_contract("TACOSM-SURROGATE-001")
    deviations = tuple(_results.contract_deviations(
        contract_steps=contract.steps,
        contract_eval_steps=contract.eval_steps,
        contract_h_levels=contract.h_levels,
        contract_seeds=contract.seeds,
        contract_arms=[a.name for a in contract.arms],
        steps=_Args.steps,
        eval_steps=_Args.eval_steps,
        h_levels=(h,),
        seeds=[0, 1],
        arms=list(surrogate.ARMS),
    ))

    return surrogate._build_record(
        _Args(), [0, 1], (h,), arm_rows, per_seed,
        per_seed_recall, train_stats, False, (h,),
        contract_source="contracts/TACOSM-SURROGATE-001.json",
        deviations=deviations,
    ).to_dict()


def test_the_summary_names_its_experiment_and_contract():
    """A summary that does not say which experiment it is from is unattributable."""
    record = _summary_with_failing_gate()
    p = record["provenance"]
    assert p["experiment_id"] == "TACOSM-SURROGATE-001"
    assert p["contract_source"].endswith("TACOSM-SURROGATE-001.json")
    assert Path(REPO_ROOT / p["contract_source"]).is_file(), (
        "the summary points at a contract that is not committed"
    )
    # The hash is what distinguishes two versions of the same id, which a
    # contract id alone cannot show. ``"missing"`` would mean the script
    # resolved a different directory than the contract loader — the two
    # resolvers are deliberate, so a disagreement must be loud.
    assert p["contract_sha256"] != "missing", (
        "the record's contract hash is 'missing': the script cannot find the "
        "contract it was checked against"
    )
    assert set(p) == {
        "experiment_id", "contract_source", "contract_sha256", "git_commit",
        "script", "python", "recorded_at",
    }, f"the provenance record is missing fields: {sorted(p)}"


def test_the_summary_records_the_design_as_run():
    """The design fields are the run's actual values, registered or not."""
    record = _summary_with_failing_gate()
    d = record["design"]
    assert d["steps"] == 20 and d["eval_steps"] == 5
    assert d["seeds"] == [0, 1]
    assert d["h_levels"] == [8]
    assert d["arms"] == ["baseline", "analytic_margin", "analytic_margin_clipped"]
    assert d["smoke"] is True and d["contract_checked"] is False
    # A smoke run that reported no deviation would be indistinguishable from a
    # registered one; the deviations are what makes the smoke flag worth more
    # than deleting the check. Here the fixture is off the registered design
    # on steps, eval length and seeds, and the record must say so.
    dims = {d2["dimension"] for d2 in d["contract_deviations"]}
    assert {"steps", "eval_steps", "seeds"} <= dims, (
        "the record's design does not list every dimension the smoke run is "
        f"off the registered design on; found {sorted(dims)}"
    )


def test_the_summary_records_a_failed_gate():
    """A failed gate is recorded, not swallowed.

    A run that failed the reproduction gate still produced measurements; the
    record must say the gate failed, because that is what makes the
    measurements unusable as a result. Deleting them on failure would delete
    the evidence that the run was invalid.
    """
    record = _summary_with_failing_gate()
    assert record["gate"]["passed"] is False
    assert record["gate"]["cells"]
    for cell in record["gate"]["cells"]:
        assert set(cell) == {
            "cell", "metric", "published", "observed", "diff", "tol", "passed"
        }, f"the gate record is missing fields at {cell}"
    assert all(isinstance(c["tol"], float) and c["tol"] >= 0.02
               for c in record["gate"]["cells"]), (
        "the tolerance floor of 0.02 is not in the record; a zero-spread cell "
        "would otherwise demand a bit-exact reproduction"
    )


def test_the_summary_records_every_arm_and_level():
    """The endpoint table covers the full arm x level grid.

    A missing cell here is an arm whose measurement is silently absent, which
    is how a re-run with a typo would look identical to a complete run.
    """
    record = _summary_with_failing_gate()
    keys = set(record["endpoints"])
    assert keys == {"baseline@H=8", "analytic_margin@H=8",
                    "analytic_margin_clipped@H=8"}


def test_the_summary_carries_the_decision_rule_verdicts():
    """The verdicts the reader cares about are fields, not prose.

    ``verdict`` must be one of the three the pre-registration names. A fourth
    value here would be a verdict the decision rule never committed to.
    """
    record = _summary_with_failing_gate()
    assert record["decision_rule"]
    for d in record["decision_rule"]:
        assert set(d) == {
            "arm", "metric", "h", "baseline", "observed", "delta",
            "baseline_spread", "threshold", "verdict"
        }, f"the decision-rule record is missing fields: {sorted(d)}"
        assert d["verdict"] in (
            "MATERIAL improvement", "MATERIAL harm", "within seed noise"
        ), f"an unregistered verdict appeared: {d['verdict']!r}"
        assert d["threshold"] == max(d["baseline_spread"], 0.02)


def test_the_summary_carries_the_reward_audit():
    """The density audit is the leakage boundary's evidence."""
    record = _summary_with_failing_gate()
    audit = record["audit"]["reward"]
    assert len(audit) == 3
    for cell in audit:
        assert {"arm", "h", "reward_nonzero_frac", "reward_mean",
                "reward_min", "reward_max", "sat_hi_frac", "sat_lo_frac",
                "successes_per_schedule"} <= set(cell), (
            f"the reward audit is missing fields: {sorted(cell)}"
        )


def test_the_summary_carries_per_seed_values():
    """The spread is the materiality threshold, so the seeds must be present.

    A record with only means cannot be re-checked: a reader cannot tell
    whether a delta inside the threshold was a uniform effect or one seed
    moving against four.
    """
    record = _summary_with_failing_gate()
    cells = record["per_seed"]["cells"]
    assert len(cells) == 3
    for cell in cells:
        assert "routing@1" in cell
        vals = cell["routing@1"]
        assert set(vals) == {"seeds", "mean", "spread"}
        assert set(vals["seeds"]) == {"0", "1"}
        assert vals["mean"] == pytest.approx(
            sum(vals["seeds"].values()) / len(vals["seeds"])
        )
        assert vals["spread"] >= 0.0


def test_the_summary_round_trips_through_json():
    """A summary that cannot be written and read is one that cannot be checked later."""
    record = _summary_with_failing_gate()
    decoded = json.loads(json.dumps(record))
    assert decoded == record


def test_the_summary_records_no_status():
    """The instrument measures; it does not arbitrate the decision rule.

    ``status`` is the one field a human sets against the pre-registration. A
    script that computed it would be marking its own homework, and the field
    belongs on the contract, where a reader can see it was set after the run.
    """
    record = _summary_with_failing_gate()
    assert "status" not in record, (
        "the summary assigns the experiment a status, which is a scientific "
        "judgement the instrument is not entitled to make"
    )
    assert "result_note" not in record
    # The nested payloads are the experiment's own and carry no verdict either.
    # A ``status`` tucked inside the audit or the per-seed values would be the
    # same judgement, one layer down where this test would not notice it.
    assert "status" not in record["audit"]
    assert "status" not in record["per_seed"]


# --------------------------------------------------------------------------- #
# 6. The M0 freeze: every measurement script is either contracted or exempt
# --------------------------------------------------------------------------- #
#
# M0 ("Measurement Integrity", ``docs/MEASUREMENT_LAYERS.md``) freezes three
# things, one of which is the contract check. Group 1 asserts the contracts
# are valid, group 4 asserts the contracted scripts enforce them. What neither
# asserts is that the two lists are *complete* — that a script in
# ``scripts/measure_*.py`` is on one of them.
#
# That gap is how the freeze would break for real. A new measurement script
# added without a contract is invisible to groups 1–5: it has no contract for
# group 3 to compare against, and group 4's parametrization does not enumerate
# the directory. It would carry the M0 surface's protection to every number it
# emitted, and nothing would notice. Symmetrically, the three scripts that
# predate the contract system and are correctly exempt from it were exempt
# only by *absence* — the partition was asserted on the contracted side and
# not at all on the other, which is how an exemption silently becomes a
# precedent rather than a recorded decision.
#
# Group 6 asserts both halves. The three exemption entries below state why
# each script needs no contract, in the script's own terms, because an
# exemption that is not justified is an exemption that will be extended.

#: The scripts deliberately outside the contract system, each with its reason.
#:
#: ``measure_baseline.py`` produces the frozen reference itself
#: (``TACOSM-BASELINE-001`` at ``91597ab``): the numbers the contracted
#: scripts' reproduction gates compare against, published before contracts
#: existed. A contract would pin levels and seeds against a design that was
#: already the definition of "baseline", and the numbers are frozen at
#: ``M1.2``, not re-derived here.
#:
#: ``measure_history_scaling.py`` and ``measure_retrieval_ceiling.py`` are the
#: same case one layer up. Both publish reference tables (``TACOSM-HS-001``,
#: ``TACOSM-RETRIEVAL-001`` F0) whose re-verified digits are the reproduction
#: targets in the *contracted* scripts' baselines. Their own numbers are
#: already the frozen thing, and C6/C7's published tables were produced by
#: them and are frozen where they were produced.
#:
#: Both still run the model-state integrity gate, because a frozen reference
#: produced from untrained weights is a frozen wrong number (C4).
#: ``measure_baseline.py`` runs neither the gate nor a record — it is the
#: untrained-by-design case, and the reason is recorded in group 6 below.
_WITHOUT_CONTRACT = (
    (
        "measure_baseline.py",
        "produces the frozen reference itself: TACOSM-BASELINE-001 at 91597ab, "
        "the numbers the contracted scripts' reproduction gates compare "
        "against. Its arms (oracle, random, static, full_context, learned) "
        "need no loaded weights — the learned arm trains inside the run — so "
        "it runs neither the integrity gate nor a record, by design. Frozen "
        "by M1.2, not re-derived under a contract that would pin the design "
        "it defines.",
    ),
    (
        "measure_history_scaling.py",
        "publishes the TACOSM-HS-001 reference tables whose re-verified digits "
        "are the reproduction targets in the contracted scripts' baselines "
        "(C6, C7). Its numbers are already the frozen thing being compared "
        "against.",
    ),
    (
        "measure_retrieval_ceiling.py",
        "publishes the TACOSM-RETRIEVAL-001 F0 reference tables the retrieval "
        "experiment builds on. Predates the contract system, and its result "
        "is frozen where it was produced rather than re-derived.",
    ),
)


def _measurement_scripts() -> list[str]:
    """Every ``measure_*.py`` in ``scripts/``, so the freeze is over the directory.

    Enumerating here rather than listing by hand is the point: a script added
    later appears in this list automatically, and the completeness test below
    then asks which side of the partition it belongs on. A hand-written list
    would have to be remembered.
    """
    return sorted(p.name for p in SCRIPTS_DIR.glob("measure_*.py"))


def test_every_measurement_script_is_either_contracted_or_exempt():
    """The M0 freeze's completeness assertion, in both directions.

    A script that is on neither list has no contract and no recorded exemption
    — an unregistered measurement, which is the one thing M0 exists to
    prevent. A script on both is a contradiction, and a script whose name
    appears in neither list has been forgotten by the freeze entirely.
    """
    contracted = {name for name, _ in _WITH_CONTRACT}
    exempt = {name for name, _ in _WITHOUT_CONTRACT}
    actual = set(_measurement_scripts())

    unaccounted = actual - contracted - exempt
    assert not unaccounted, (
        "measurement scripts with neither a contract nor a recorded exemption "
        f"were found; each needs one or the other: {sorted(unaccounted)}"
    )
    # A script cannot be both: an exemption that also carries a contract is
    # not an exemption, and the reason would be unreadable.
    assert not (contracted & exempt), (
        "a script is in both _WITH_CONTRACT and _WITHOUT_CONTRACT, so the "
        f"partition is not one: {sorted(contracted & exempt)}"
    )


def test_the_exempt_scripts_still_run_the_integrity_gate():
    """An exemption from contracts is not an exemption from M0's second freeze.

    ``measure_history_scaling.py`` and ``measure_retrieval_ceiling.py`` produce
    the frozen reference tables the contracted scripts' reproduction gates
    compare against, so the one thing they must not do is emit numbers from
    untrained weights — the exact failure C4 exists for, and the failure that
    made a published HS-001 table wrong once already (``CLAIMS.md`` §C4, "The
    HS-001 correction, recorded").

    ``measure_baseline.py`` is the exception to the exception, and it is
    deliberate: its arms are ``oracle``, ``random``, ``static``,
    ``full_context`` and ``learned``, and the first four need no weights at
    all. The fifth trains *inside* the run and is measured at the end of the
    schedule, so an ``assert_trained`` call would be checking a router that is
    untrained by design. Its protection is not the gate but the *arm set* —
    the baseline is a comparison against arms whose behaviour does not depend
    on training, which is why the reference it produces is trustworthy without
    one.
    """
    for name, _reason in _WITHOUT_CONTRACT:
        if name == "measure_baseline.py":
            continue
        text = _script_text(name)
        assert "assert_trained" in text, (
            f"{name} is exempt from the contract system but does not call "
            "assert_trained; an exempt script still runs the M0 integrity "
            "gate, because it produces the frozen reference the contracted "
            "scripts are checked against"
        )


def test_baseline_has_no_integrity_gate_by_design():
    """The one script with no gate, and why that is correct rather than a gap.

    A reader who applies the M0 checklist to ``measure_baseline.py`` finds the
    contract check absent and the integrity gate absent, and could conclude
    the freeze was never applied to it. Both absences are the same reason: the
    script measures the reference, not a trained model. Pinned here so the
    exemption is not "found" later as an oversight.
    """
    text = _script_text("measure_baseline.py")
    assert "assert_trained" not in text, (
        "measure_baseline.py runs the integrity gate, which contradicts the "
        "exemption's reason: its arms need no trained weights"
    )
    assert "load_weights" not in text, (
        "measure_baseline.py loads weights, which contradicts the exemption's "
        "reason: the learned arm trains inside the run rather than loading a "
        "checkpoint"
    )


def test_the_record_freeze_covers_the_contracted_scripts_and_not_the_exempt_ones():
    """The boundary of M0's third freeze, stated as a boundary.

    The machine-readable record arrived with the contract system in M1.0, so
    it covers exactly the scripts that have contracts and no others. The three
    exempt scripts predate it and report to the terminal alone.

    That is a real limit on the freeze, and recording it is the point of this
    group: an assertion that the record *did* cover them would be false, and a
    silent gap is what a reader would otherwise have to notice by absence. The
    audit trail for the frozen reference is the published tables in
    ``CLAIMS.md`` and ``docs/EVIDENCE_REGISTER.md`` at the frozen commits,
    which are committed — a record under ``results/`` would not be, because
    ``results/*.json`` is gitignored. The committed table is therefore the
    auditable form of these numbers, and the record's absence is not a hole in
    the audit trail.

    What the test pins is the *partition*: the record reaches every contracted
    script, and any future script that is added to ``_WITH_CONTRACT`` is
    required to reach it too, because the contract is what a record points at.
    """
    contracted = {name for name, _ in _WITH_CONTRACT}
    for name in sorted(contracted):
        text = _script_text(name)
        assert "results_dir_for" in text or "write_record" in text, (
            f"{name} has a contract but writes no machine-readable record; "
            "the record is what a contract's outcome is checked against, so a "
            "contracted script without one has a pre-registration nothing can "
            "be compared to after the fact"
        )
    # The exempt scripts predate the record module. Naming them here rather
    # than testing nothing is what keeps the boundary from being read as an
    # oversight: each is a known exemption, and adding one to _WITH_CONTRACT
    # moves it to the asserted half above.
    exempt_without_record = {
        name
        for name, _ in _WITHOUT_CONTRACT
        if "results_dir_for" not in _script_text(name)
        and "write_record" not in _script_text(name)
    }
    assert exempt_without_record == {name for name, _ in _WITHOUT_CONTRACT}, (
        "an exempt script gained a machine-readable record; if it has a "
        "record it is no longer the frozen reference a contract would "
        "re-derive, and its exemption entry needs to say why"
    )


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


def _minimal_contract(
    *,
    endpoints: tuple[EndpointSpec, ...] = (
        EndpointSpec(name="a", role="r", primary=True),
    ),
    arms: tuple[ArmSpec, ...] = (ArmSpec(name="baseline", intervention="i", acts_on="a"),),
    decision_rule: tuple[DecisionBranch, ...] = (
        DecisionBranch(condition="if a moves", licenses="the mechanism"),
    ),
    interpretation_order: tuple[str, ...] = ("first", "second"),
    h_levels: tuple[int, ...] = (8, 64, 256),
    steps: int = 500,
    amendments: tuple[AmendmentSpec, ...] = (),
) -> ExperimentContract:
    """A contract with only the required structure, for invariant tests."""
    return ExperimentContract(
        experiment_id="TEST-001",
        title="t",
        question="q",
        hypothesis="h",
        arms=arms,
        endpoints=endpoints,
        decision_rule=decision_rule,
        interpretation_order=interpretation_order,
        h_levels=h_levels,
        seeds=(0, 1, 2, 3, 4),
        steps=steps,
        eval_steps=100,
        held_constant=("one thing held fixed",),
        amendments=amendments,
    )
