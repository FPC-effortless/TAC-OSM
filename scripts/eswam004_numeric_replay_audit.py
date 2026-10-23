"""ESWAM004 original whole-real-checkpoint immutable numeric replay audit.

ALL original model/optimizer/RNG/raw scores and preregistered tolerances stay
unchanged. This runs on the stored real source and NEVER optimizes weights.
"""
from __future__ import annotations
import argparse
from pathlib import Path

from scripts import eswam004_train_real as original
from scripts import eswam003_numeric_replay_audit as audit
from tac_osm.variants.kinetic_exchange_world import KineticExchangeWorldModel

audit.ROOT = original.registered.ROOT
audit.OUT = original.registered.OUT
audit.STUDY = original.registered.STUDY
audit.PARAMETERS = original.registered.PARAMETERS
audit.score = original.score
audit.preflight = original.preflight
audit.SelectiveExchangeWorldModel = KineticExchangeWorldModel

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--cache",type=Path,required=True)
    args=parser.parse_args()
    audit.audit(args.cache)
