#!/usr/bin/env python3
"""Repository-wide research governance preflight."""
from __future__ import annotations
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    ROOT / "docs" / "RESEARCH_GOVERNANCE.md",
    ROOT / "docs" / "RESEARCH_LANE_REGISTRY.md",
    ROOT / "docs" / "RESEARCH_RUN_GATES_V2.md",
)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-sha", default="")
    ap.parse_args()
    errors = [str(p) for p in REQUIRED if not p.is_file()]
    contract_test = ROOT / 'tests' / 'test_contract.py'
    if not contract_test.is_file():
        errors.append('missing tests/test_contract.py')
    else:
        text = contract_test.read_text(encoding='utf-8')
        exempt = {
            'scripts/measure_history_scaling.py',
            'scripts/measure_retrieval_ceiling.py',
            'scripts/measure_baseline.py',
        }
        for p in sorted((ROOT / 'scripts').glob('measure_*.py')):
            rel = p.relative_to(ROOT).as_posix()
            if rel not in exempt and rel not in text:
                errors.append(f'measurement script is not registered/exempted: {rel}')
    if errors:
        for error in errors: print('[FAIL]', error)
        return 1
    print('[PASS] unified research governance preflight')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())