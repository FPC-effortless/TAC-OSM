#!/usr/bin/env python3
"""Repository-wide research governance preflight."""
from __future__ import annotations
import argparse
import ast
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
        source = contract_test.read_text(encoding='utf-8')
        try:
            tree = ast.parse(source, filename=str(contract_test))
            assigned = {}
            for node in tree.body:
                if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id in {'_WITH_CONTRACT', '_WITHOUT_CONTRACT'}
                    for t in node.targets
                ):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id in {'_WITH_CONTRACT', '_WITHOUT_CONTRACT'}:
                            assigned[target.id] = ast.literal_eval(node.value)
            contracted = {name for name, _reason in assigned.get('_WITH_CONTRACT', ())}
            exempt = {name for name, _reason in assigned.get('_WITHOUT_CONTRACT', ())}
            if not assigned.get('_WITH_CONTRACT') or not assigned.get('_WITHOUT_CONTRACT'):
                errors.append('tests/test_contract.py does not expose both M0 measurement partitions')
            if contracted & exempt:
                errors.append(
                    'measurement script appears in both contracted and exempt partitions: '
                    + ', '.join(sorted(contracted & exempt))
                )
        except (SyntaxError, ValueError, TypeError) as exc:
            contracted, exempt = set(), set()
            errors.append(f'cannot parse M0 measurement partitions from tests/test_contract.py: {exc}')
        for p in sorted((ROOT / 'scripts').glob('measure_*.py')):
            rel = p.relative_to(ROOT).as_posix()
            name = p.name
            if name not in {Path(x).name for x in contracted | exempt}:
                errors.append(f'measurement script is not registered/exempted: {rel}')
    if errors:
        for error in errors:
            print('[FAIL]', error)
        return 1
    print('[PASS] unified research governance preflight')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())