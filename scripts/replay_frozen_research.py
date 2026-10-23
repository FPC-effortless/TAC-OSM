"""Run an explicitly pinned historical replay with all optimizer updates blocked."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from scripts.verify_frozen_replay import REGISTRY, git_bytes, verify_frozen_replay

SCRIPTS = {
    "me071": "me071_persistent_workflow.py",
    "me072": "me072_marked_world.py",
    "me073": "me073_autoregressive_world.py",
}
# Executed only inside the original snapshot, before importing its runner.
# It deliberately does not import current model implementations into that process.
BOOTSTRAP = """
import runpy,sys
from pathlib import Path
sys.path[:0]=[str(Path.cwd()/'src'),str(Path.cwd())]
from torch.optim.optimizer import register_optimizer_step_pre_hook
def reject_update(optimizer,args,kwargs):
    raise RuntimeError('IMMUTABLE_REPLAY: optimizer updates are forbidden')
hook=register_optimizer_step_pre_hook(reject_update)
script=sys.argv[1]
sys.argv=sys.argv[1:]
try:
    runpy.run_path(script,run_name='__main__')
finally:
    hook.remove()
"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", choices=SCRIPTS, required=True)
    p.add_argument("--cache", type=Path, required=True)
    args = p.parse_args()
    registry = json.loads(REGISTRY.read_text())
    entries = [
        registry["entries"][args.study + "-" + phase] for phase in ("prefit", "final")
    ]
    for phase, entry in zip(("prefit", "final"), entries):
        verify_frozen_replay(
            json.loads(git_bytes(entry["commit"], entry["registration_path"])),
            args.study + "-" + phase,
        )
    if len({e["commit"] for e in entries}) != 1:
        raise RuntimeError("one immutable replay snapshot required")
    with tempfile.TemporaryDirectory(prefix="tac-readonly-replay-") as temp:
        snapshot = Path(temp) / "original"
        subprocess.run(
            [
                "git",
                "worktree",
                "add",
                "--quiet",
                "--detach",
                str(snapshot),
                entries[0]["commit"],
            ],
            cwd=ROOT,
            check=True,
        )
        try:
            env = dict(os.environ)
            env.pop("PYTHONPATH", None)
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    BOOTSTRAP,
                    "scripts/" + SCRIPTS[args.study],
                    "--phase",
                    "replay",
                    "--cache",
                    str(args.cache.resolve()),
                ],
                cwd=snapshot,
                env=env,
                check=True,
            )
        finally:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(snapshot)],
                cwd=ROOT,
                check=True,
            )
    print(
        json.dumps(
            {
                "status": "PASS_PINNED_SCIENTIFIC_REPLAY",
                "study": args.study,
                "original_commit": entries[0]["commit"],
                "optimizer_updates_allowed": False,
            }
        )
    )


if __name__ == "__main__":
    main()
