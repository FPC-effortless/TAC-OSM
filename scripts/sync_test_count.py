#!/usr/bin/env python3
"""Keep the live test count in the frozen baseline's provenance table.

## Why this exists

`docs/TACOSM-BASELINE-001.md` records the size of the suite at the time the
baseline was frozen, next to the *current* size, because a frozen table and a
growing suite disagree and a reader is entitled to see both numbers at once.

Written by hand, that second number rots on every commit that adds a test —
it was stale within one commit of being written. A hand-maintained constant
in a document cannot stay correct, so this script substitutes the live count
for whatever the row currently carries.

## What it does not do

It touches one number, and only that number. The frozen columns — every
measured accuracy in the tables below the provenance block — are never
reached. The frozen value is a literal in the row's pattern, not something a
substitution fills in, so it stays `327` while the current count moves.

## Usage

    python scripts/sync_test_count.py          # rewrite in place, print the diff
    python scripts/sync_test_count.py --check  # exit 1 if the doc is stale

`--check` is the pre-commit form: it reports the number the doc would carry
without writing it, so a stale count fails the gate rather than being silently
refreshed after the fact. It is enforced in CI and pinned by a unit test in
`tests/test_integrity.py`, so a count that drifts fails a gate rather than
being noticed by a reader against the frozen table.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOC = REPO_ROOT / "docs" / "TACOSM-BASELINE-001.md"

#: The placeholder the doc carries. An HTML comment inside a markdown table is
#: still rendered as a row by most parsers, so the table's shape is unaffected
#: and `check_tables.py` still sees the right pipe count. It is also a string
#: no human would type as a count, which is the point: a stale hand-written
#: number looks correct, and this does not.
MARKER = "<!--TESTCOUNT-->"

#: The pinned generator the two gated test modules load from. When it is
#: absent those modules skip at import and are never collected; when it is
#: present — as on CI — they are collected like any other module. The count
#: has to land on the same number either way, so the correction below is
#: conditional on this directory rather than applied unconditionally.
GATED_SOURCE = REPO_ROOT / "third_party" / "cdl-attention-experiment"


def live_test_count() -> int:
    """The number of tests the suite collects.

    Counted by running pytest, because a static count is not available:
    parametrized tests multiply at collection time, so the number of collected
    cases is not the number of `def test_` lines in the sources.

    One collection is not enough on its own. Two modules skip at import time
    when the pinned generator under `third_party/` is absent —

        tests/test_graph_casm_selective_010.py
        tests/test_graph_casm_identifiability_011.py

    — and a module-level `pytest.skip` is not collected at all, so the count
    is 14 lower on a checkout without the generator than on CI, which checks
    it out. The doc has to carry one number that is right in both, so the
    collection is corrected by the count of cases the gate swallowed — and
    only when the generator is actually absent, or CI would double-count.
    """
    collected = _pytest_collected()
    if collected is None:
        raise RuntimeError("could not read a test count from pytest")
    if GATED_SOURCE.exists():
        return collected
    return collected + _gated_case_count()


def _gated_case_count() -> int:
    """Cases the collection misses because their modules skip at import.

    Every gated module guards on the pinned generator's directory and skips
    without it, so its cases are collected on CI and nowhere else. None is
    parametrized, so a module's case count is its declared test-function count.
    """
    total = 0
    for path in (REPO_ROOT / "tests").glob("test_*.py"):
        source = path.read_text(encoding="utf-8")
        if 'pytest.skip("pinned external generator is not checked out"' not in source:
            continue
        total += sum(
            1
            for line in source.splitlines()
            if line.startswith("def test_") or line.startswith("async def test_")
        )
    return total


def _pytest_collected() -> int | None:
    """What pytest collected in *this* checkout, for the `--verify` report."""
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    for line in r.stdout.splitlines():
        if "tests collected" in line:
            return int(line.split()[0])
    return None


#: The prose that carries the count, with the number itself as a regex group.
#: Anchored to the *current* marker so the frozen count is never touched: the
#: frozen value is a literal in the pattern's fixed part, not something a
#: substitution fills in.
ROW_PATTERN = r"test count @ freeze \| (\d+) \(current count in `tests/`: (\d+)\)"


def rewrite(text: str, count: int) -> str:
    """Set the current count, leaving every other line byte-identical.

    Handles both forms the row can be in: the placeholder, on a doc that has
    never been synced, and a written count, on one that has. A doc that has
    neither — the row is missing entirely, or was hand-edited away from the
    pattern — is an error rather than a silent skip, because a count that
    cannot be located cannot be updated and the freeze is only as good as the
    row a reader finds.
    """
    if MARKER in text:
        return text.replace(MARKER, str(count))
    m = re.search(ROW_PATTERN, text)
    if m:
        return re.sub(ROW_PATTERN,
                      lambda _: f"test count @ freeze | {m.group(1)} "
                                f"(current count in `tests/`: {count})",
                      text)
    raise SystemExit(
        f"{DOC} has neither the {MARKER} placeholder nor a row matching "
        f"{ROW_PATTERN}. Either the provenance row was rewritten by hand — "
        "in which case restore the row — or the row moved, in which case "
        "point ROW_PATTERN at it. A count that cannot be updated silently "
        "stops being a count."
    )


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check",
        action="store_true",
        help="report whether the doc is stale without writing to it",
    )
    ap.add_argument(
        "--verify",
        action="store_true",
        help="also report what pytest itself collects, to expose any gap",
    )
    args = ap.parse_args(argv)

    count = live_test_count()
    if args.verify:
        # Show the terms the count is built from. Where the generator is
        # present the collection already includes the gated cases and the
        # correction is zero; where it is absent, the correction is the count
        # of cases the gate swallowed.
        gated = 0 if GATED_SOURCE.exists() else _gated_case_count()
        print(f"verify: collection {_pytest_collected()} "
              f"+ gated {gated} = {count}")
    text = DOC.read_text(encoding="utf-8")

    if args.check:
        if MARKER in text:
            print(f"STALE: {DOC} carries the placeholder; run sync_test_count.py")
            return 1
        expected = f"current count in `tests/`: {count}"
        if expected not in text:
            print(f"STALE: {DOC} does not say {expected!r}")
            return 1
        print(f"OK: {DOC} reports the current count, {count}")
        return 0

    # Refresh, then report what changed.
    before = text
    after = rewrite(text, count)
    if after == before:
        print(f"OK: already current at {count}")
        return 0
    DOC.write_text(after, encoding="utf-8")
    for a, b in zip(before.splitlines(), after.splitlines()):
        if a != b:
            print(f"updated: {DOC}")
            print(f"  - {a.strip()}")
            print(f"  + {b.strip()}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
