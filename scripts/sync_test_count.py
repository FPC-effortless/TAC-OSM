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


def live_test_count() -> int:
    """The number of tests pytest would collect.

    Collected rather than run, because the count is what a reader compares
    against the frozen figure — running the suite to count it would make this
    a seconds-to-minutes task rather than a subsecond one, and the count does
    not depend on the tests passing.
    """
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    for line in r.stdout.splitlines():
        # pytest's summary line: "648 tests collected in 0.32s"
        if "tests collected" in line:
            return int(line.split()[0])
    raise RuntimeError(
        f"could not read a test count from pytest; output was:\n{r.stdout}\n{r.stderr}"
    )


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
    args = ap.parse_args(argv)

    count = live_test_count()
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
