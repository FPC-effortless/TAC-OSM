"""Replay and verify original checkpoints in fresh processes within each arm's budget."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ARMS = ('tac_selected','nanochat','gru')


def main(runs, output):
    output.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        original = json.loads((runs/f'{arm}-receipt.json').read_text())
        if original['status'] != 'succeeded':
            raise ValueError(f'Failed training receipt for {arm}')
        remaining = 900 - original['elapsed_seconds']
        if remaining <= 0:
            raise ValueError(f'No replay budget left for {arm}')
        run_dir = runs / original['id']
        start = time.monotonic()
        result = subprocess.run([sys.executable, str(HERE/'pilot.py'), arm, '--replay', str(run_dir/'output')],
                                check=True, capture_output=True, text=True, timeout=remaining)
        replay = json.loads(result.stdout)
        replay['elapsed_s'] = time.monotonic()-start
        if replay['checkpoint_sha256'] != original['artifacts']['checkpoint.pt']:
            raise ValueError('Replay and original TrainLab checkpoint digests differ')
        (output / f'{arm}-replay.json').write_text(json.dumps(replay,indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path)
    parser.add_argument('output', type=Path)
    args=parser.parse_args()
    main(args.runs,args.output)
