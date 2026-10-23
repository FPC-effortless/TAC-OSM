"""Launch a new registered research entrypoint with an optimizer backstop."""

import argparse
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from tac_osm.parameter_budget import _ACTIVE_MANIFEST, install_unbound_optimizer_guard
from scripts.validate_research_parity import check_manifest
import json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--script", type=Path, required=True)
    args, tail = p.parse_known_args()
    manifest = args.manifest.resolve()
    check_manifest(json.loads(manifest.read_text()), "preflight")
    target = args.script.resolve()
    if not target.is_relative_to(ROOT):
        p.error("entrypoint must belong to this repository")
    hook = install_unbound_optimizer_guard()
    token = _ACTIVE_MANIFEST.set(manifest)
    previous = sys.argv
    try:
        sys.argv = [str(target)] + (tail[1:] if tail[:1] == ["--"] else tail)
        runpy.run_path(str(target), run_name="__main__")
    finally:
        sys.argv = previous
        _ACTIVE_MANIFEST.reset(token)
        hook.remove()


if __name__ == "__main__":
    main()
