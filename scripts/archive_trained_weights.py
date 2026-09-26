#!/usr/bin/env python
"""Archive trained router weights so a frozen evaluation can verify its state.

The model-state integrity gate
(``src/tac_osm/integrity.py``) exists because TACOSM-HS-001 measured a
``routing@1`` of ~1/40 from a router whose weights had never been loaded:
``w = [0]*n`` scored every candidate uniformly, and a uniform distribution is
smooth, bounded and plausible-looking, so nothing downstream objected.

A gate that runs at evaluation time can only refuse the bad state if it has
the good one. This script is that half: it trains a router per arm and seed
under a fixed configuration, snapshots the resulting parameters, and writes
them to ``results/trained_weights/`` as JSON alongside the manifest a reader
needs to verify the record independently:

    arm / seed / n_steps / train_candidates
    parameter_hash          — the state's identity
    untrained_hash          — what an all-zero vector of this width hashes to
    weight_signature        — the block means, i.e. what rule was learned
    parameter_norm          — how much weight training actually moved
    weights                 — the parameters themselves, so the record is
                              self-verifying rather than merely descriptive

An evaluation then loads the weights with ``load_weights``, which records the
checkpoint as copied rather than trained-in-place, and the gate compares the
loaded state against the untrained hash.

The TACOSM-BASELINE-001 and TACOSM-HS-001 numbers in ``docs/`` predate this
archive. They are unaffected — their scripts are unchanged — but they cannot
be re-verified by the gate, which is recorded in ``docs/CLAIMS.md`` as a
limitation of the frozen baseline rather than a claim about it.

**The archive is a local artifact, not a committed one.** ``results/*.json``
is gitignored, so the weights do not travel with the repository; re-running
this script regenerates them deterministically from the same seeds. The
*gate* is the committed part — it is what refuses a bad state — and the
weights it verifies are reproduced on demand rather than stored in version
control.

Usage:
    python scripts/archive_trained_weights.py --n-steps 500 --seeds 0,1,2,3,4
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tac_osm.ablation import AblationConfig, RouterSwitch  # noqa: E402
from tac_osm.builder import build_model  # noqa: E402
from tac_osm.integrity import (  # noqa: E402
    Checkpoint,
    IntegrityError,
    manifest_to_dict,
    parameter_hash,
    snapshot_router,
    untrained_hash_for,
)

#: Where the archive lives. Under ``results/`` rather than ``docs/`` because it
#: is a generated artifact, not prose; the manifest is the documentation.
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "trained_weights")

#: The arms that carry learned parameters. The others (static, random, oracle,
#: full_context) have nothing to snapshot — recorded as an explicit absence in
#: the manifest rather than as a missing file.
ARMS = ("learned",)


def _train_one(arm: str, seed: int, n_steps: int, train_candidates: int) -> Checkpoint:
    """Train one router and snapshot it, refusing to archive an untrained state."""
    cfg = AblationConfig(seed=seed, router=RouterSwitch(type=arm))
    bm = build_model(cfg, n_steps=n_steps)
    bm.model.environment.config.n_candidates = train_candidates
    bm.model.run()
    checkpoint = snapshot_router(bm.model.router)
    if checkpoint is None:
        raise IntegrityError(
            f"arm {arm!r} carries no learned parameters; nothing to archive"
        )
    if checkpoint.untrained:
        raise IntegrityError(
            f"arm {arm!r} seed={seed} trained for {n_steps} steps and still has "
            "an all-zero parameter vector: training did not move the weights, so "
            "archiving this state would publish a uniform scorer as a trained one"
        )
    return checkpoint


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-steps", type=int, default=500)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--train-candidates", type=int, default=8)
    ap.add_argument("--experiment-id", default="TACOSM-BASELINE-001")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    commit = _current_commit()
    os.makedirs(OUT_DIR, exist_ok=True)

    manifests = []
    for arm in ARMS:
        for seed in seeds:
            checkpoint = _train_one(arm, seed, args.n_steps, args.train_candidates)
            # The manifest is written with the checkpoint so the archive is
            # self-verifying: a reader recomputes the hash from the weights
            # and confirms the state, rather than trusting a hex string.
            manifest = manifest_to_dict(
                experiment_id=f"{args.experiment_id}:{arm}",
                arm=arm,
                checkpoint=checkpoint,
                commit=commit,
                seed=seed,
                n_steps=args.n_steps,
            )
            manifest["train_candidates"] = args.train_candidates
            # ``untrained_hash_for`` is re-derived here from the checkpoint's
            # own width so the comparison value travels with the record.
            manifest["model_state"]["untrained_hash"] = untrained_hash_for(
                checkpoint.weights
            )
            path = os.path.join(OUT_DIR, f"{arm}_seed{seed}.json")
            with open(path, "w") as fh:
                json.dump(manifest, fh, indent=2, sort_keys=True)
                fh.write("\n")
            manifests.append((arm, seed, path))
            print(f"archived {arm} seed={seed} -> {os.path.relpath(path)} "
                  f"(updates={checkpoint.n_updates}, "
                  f"norm={checkpoint.norm:.4f}, hash={checkpoint.hash[:12]})")

    print()
    print(f"{len(manifests)} checkpoint(s) written to {os.path.relpath(OUT_DIR)}")
    print("an evaluation loads these with load_weights; the integrity gate "
          "then compares")
    print("the loaded state against the untrained hash before it measures "
          "anything.")


def _current_commit() -> str:
    """The working-tree commit, so an archive names the code that produced it."""
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=os.path.dirname(__file__),
        ).stdout.strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
