#!/usr/bin/env python3
"""Hash original repository-backed FIT/DEV replay inputs, without executing them."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from tac_osm.replay_intake import inspect, safe_file, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--inventory', help='reviewed repository-relative inventory path')
    parser.add_argument('--inventory-sha256', help='reviewed original inventory digest')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        contract = json.loads(args.contract.read_text())
        names = subprocess.check_output(['git', '-C', str(args.root), 'ls-tree', '-rz', '--name-only', 'HEAD'])
        tracked = set(names.decode().rstrip('\0').split('\0'))
        inventory = None
        if args.inventory:
            path = safe_file(args.root, args.inventory)
            if args.inventory not in tracked or not args.inventory_sha256:
                raise ValueError('reviewed repository-owned inventory and digest required')
            if sha256(path) != args.inventory_sha256:
                raise ValueError('inventory hash differs before evidence inspection')
            inventory = json.loads(path.read_text())
        elif args.inventory_sha256:
            raise ValueError('digest without inventory')
        report = inspect(contract, inventory, args.root, tracked)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        report = dict(state='FAIL', REPO_REPLAY='BLOCKED', reason=str(exc),
                      model_executions=0, training_steps=0, protected_TEST=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(report, sort_keys=True))
    return {'READY_FOR_REPLAY': 0, 'BLOCKED': 2, 'FAIL': 1}[report['state']]


if __name__ == '__main__':
    sys.exit(main())
