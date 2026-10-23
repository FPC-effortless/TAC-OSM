"""ESWAM004: EXACT source-window sampler, loss and scorer as frozen ESWAM003.

The registered delta is only the model's kinetic gate, not a new data recipe.
This wrapper switches the globally resolved candidate of the original reference
implementation without altering ESWAM003 source, training, or checkpoint files.
No source loading happens before preflight. This study's outputs are separate.
"""
from __future__ import annotations
import argparse
from pathlib import Path

from scripts import eswam003_train_real as registered
from tac_osm.variants.kinetic_exchange_world import KineticExchangeWorldModel

registered.ROOT = Path("research/mathematical-engineering/eswam004")
registered.MANIFEST = registered.ROOT / "FAIR_COMPARISON.json"
registered.OUT = Path("research/replay-bundles/ESWAM004")
registered.STUDY = "ESWAM004_CAUSAL_KINETIC_EVIDENCE_REAL_DEVELOPMENT"
registered.PARAMETERS = 101706
registered.SelectiveExchangeWorldModel = KineticExchangeWorldModel

# Keep all unchanged methods (ids_for, supervised_next_command, rollout, physical_loss,
# extended_real_order, score, original_me076_same_case, preflight, train_one, fit,
# replay) owned by the ORIGINAL ESWAM003 real study. The original ESWAM003 module
# still reads/writes only its own targets when invoked as an entrypoint; this
# experiment's globals are process-local in this entrypoint.
preflight = registered.preflight
fit = registered.fit
replay = registered.replay
score = registered.score
rollout = registered.rollout
physical_loss = registered.physical_loss
extended_real_order = registered.extended_real_order

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["preflight", "fit", "replay"])
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    if args.mode == "preflight":
        print(f"ESWAM004_LIVE_PREFIT_PARAMETERS={preflight()}",flush=True)
    else:
        if args.cache is None:
            raise ValueError("registered real publisher source cache required")
        if args.mode == "fit":
            fit(args.cache)
        else:
            replay(args.cache)
