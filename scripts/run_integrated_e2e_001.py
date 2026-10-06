#!/usr/bin/env python3
"""Invalidated TACOSM-PLM-INTEGRATED-E2E-001 compatibility stub.

The predecessor benchmark was invalidated for query-entity leakage. This module
is retained solely so historical references fail closed instead of silently
executing the obsolete runner.
"""
from __future__ import annotations

PROVENANCE_ONLY = True
EXPERIMENT_ID = "TACOSM-PLM-INTEGRATED-E2E-001"

def main() -> int:
    raise RuntimeError(
        "TACOSM-PLM-INTEGRATED-E2E-001 is invalidated and provenance-only; "
        "run TACOSM-PLM-INTEGRATED-E2E-003 instead."
    )

if __name__ == "__main__":
    raise SystemExit(main())
