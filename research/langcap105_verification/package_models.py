"""Package only the verified complete original cohort, with auditable bytes."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from checkpoints import PROTOCOL_SHA, TRAINING_COMMIT, digest, require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fits", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--checkpoint-audit", type=Path, required=True)
    parser.add_argument("--equation-audit", type=Path, required=True)
    parser.add_argument("--cohort-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Do not overwrite model bundles")
    require(digest(args.protocol) == PROTOCOL_SHA, "Original protocol differs")
    checked = json.loads(args.checkpoint_audit.read_text())
    equations = json.loads(args.equation_audit.read_text())
    cohort = json.loads(args.cohort_audit.read_text())
    require(checked["status"] == "ALL_18_TRAINED_CHECKPOINTS_VERIFIED" and checked["verified_fit_count"] == 18,
            "Checkpoint cohort is incomplete")
    require(equations["complete_18_fit_cohort"] is True and equations["model_count"] == 18 and
            equations["case_count"] == 36, "Trained-equation cohort is incomplete")
    require(cohort["status"] == "18_FIT_COMPLETED_NOT_SCIENTIFICALLY_QUALIFIED" and cohort["fit_count"] == 18,
            "Original final cohort audit missing")
    accepted = {(r["scale"], r["arm"], r["seed"]): r["checkpoint_sha256"] for r in checked["models"]}
    accepted_receipts = {(r["scale"], r["arm"], r["seed"]): r["receipt_sha256"] for r in checked["models"]}
    require({(r["scale"], r["arm"], r["seed"]): r["checkpoint_sha256"] for r in equations["models"]} == accepted,
            "Equation and checkpoint audits refer to different weights")
    require({(r["scale"], r["arm"], r["seed"]): r["checkpoint_sha256"] for r in cohort["fits"]} == accepted,
            "Original and supplemental audits refer to different weights")
    paths = {}
    for receipt in args.fits.rglob("FIT_RECEIPT.json"):
        v = json.loads(receipt.read_text()); key = (v["scale"], v["arm"], v["seed"])
        require(key in accepted and key not in paths, "Unexpected or duplicate original receipt")
        require(digest(receipt) == accepted_receipts[key], "Original training receipt changed")
        require(digest(receipt.parent / "TRAINED.pt") == accepted[key], "Original trained bytes changed")
        paths[key] = receipt.parent
    require(set(paths) == set(accepted), "Incomplete original artifact set")
    groups = [("LANGCAP105_1M_all_models.zip", sorted(k for k in paths if k[0] == "small"))]
    groups += [("LANGCAP105_10M_" + arm + ".zip", sorted(k for k in paths if k[:2] == ("medium", arm)))
               for arm in ("tac_associative_stack", "causal_transformer", "stacked_gru")]
    readme = f"""LANGCAP105 owner-authorized exploratory trained models

Original training repository: FPC-effortless/TAC-OSM
Original immutable training commit: {TRAINING_COMMIT}
Original Actions run: 41229365510
Protocol SHA256: {PROTOCOL_SHA}

All 18 original fits completed 2,560 AdamW updates, batch 4, 128-byte context,
with 1,310,720 FIT byte presentations each, on authenticated CPython FIT-only
data. This bundle contains the specified subset; reports cover all 18 fits.
No DEV/TEST scoring, independent scientific qualification or architecture
superiority is established. Global mathematical release remains OPEN.

Each model folder contains ORIGINAL TRAINED.pt and FIT_RECEIPT.json bytes.
TRAINED.pt retains model state_dict, AdamW moments/counters, CPU RNG, seed,
source and protocol identity. Only final checkpoints are retained; original
intermediate LATEST.pt snapshots were overwritten/deleted by the worker.

Use Python 3.12, torch==2.15.0+cpu and NumPy. Obtain the exact source commit
above. Construct the model using research/langcap103/preflight.py and the
configuration in OWNER_PROTOCOL.json. Load with:
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model'], strict=True)
The read-only verifier in the accompanying audit PR #685 shows complete
construction, loading, source-order first-loss replay and integrity checks.
Do not execute arbitrary deserializers or infer deployed capability from FIT.

BUNDLE_CONTENTS.json binds every retained file to its SHA256.
"""
    args.output.mkdir(parents=True)
    receipts = []
    for filename, keys in groups:
        require(len(keys) == (9 if "1M_all" in filename else 3), "Incorrect bundle grouping")
        target = args.output / filename
        contents = []
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
            z.writestr("README.txt", readme)
            for name, p in (("OWNER_PROTOCOL.json", args.protocol), ("CHECKPOINT_AUDIT.json", args.checkpoint_audit),
                            ("TRAINED_GRAPH_AUDIT.json", args.equation_audit), ("ORIGINAL_FULL_FIT_AUDIT.json", args.cohort_audit)):
                z.write(p, name)
            for key in keys:
                label = "-".join(map(str, key))
                for name in ("TRAINED.pt", "FIT_RECEIPT.json"):
                    source = paths[key] / name; relative = label + "/" + name
                    checksum = digest(source)
                    z.write(source, relative)
                    contents.append({"file":relative, "sha256":checksum, "bytes":source.stat().st_size})
            z.writestr("BUNDLE_CONTENTS.json", json.dumps(contents, indent=2) + "\n")
        with zipfile.ZipFile(target) as z:
            require(z.testzip() is None, "Bundle CRC failed")
            for item in contents:
                h = hashlib.sha256()
                with z.open(item["file"]) as stream:
                    for block in iter(lambda: stream.read(1048576), b""):
                        h.update(block)
                require(h.hexdigest() == item["sha256"], "Bundled trained bytes differ")
        receipts.append({"filename":filename, "sha256":digest(target), "bytes":target.stat().st_size,
                         "fit_count":len(keys), "models":[list(k) for k in keys]})
        print(json.dumps(receipts[-1]), flush=True)
    (args.output / "BUNDLE_RECEIPTS.json").write_text(json.dumps({"original_training_run":41229365510,
        "training_commit":TRAINING_COMMIT, "fit_count":18, "bundles":receipts}, indent=2) + "\n")


if __name__ == "__main__":
    main()
