"""Read-only, posthoc decomposition of original LANGACQUIRE097 ARC outputs.

Accepts only the exact preserved parent result archive. Imports no model source,
executes no neural forward, reads no LANGCONTROL098 outputs, and changes no
registered evaluator, endpoint, target, checkpoint, or prediction. NumPy reads
saved NPZ arrays with allow_pickle=False; archive members are never extracted.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
from io import BytesIO
import json
from pathlib import Path
import zipfile

import numpy as np


ARCHIVE_SHA256 = "0e8c425a8cc99483cc70f5b9cde238614d348c340762f848f5ca4c59e07a5b63"
ARCHIVE_BYTES = 135590835
PARENT_SOURCE = "d21c7e7b385ca164eac9046fafca43fc9a3762b1"
FAMILIES = ("tac", "gru", "ssm", "transformer")
OBJECTIVES = ("full", "query")
SEEDS = (1701, 1702, 1703, 1704, 1705)
ACTOR = "review-source/frozen-data/DEV_ACTOR.json"
JUDGE = "review-source/frozen-data/DEV_JUDGE.json"
DATA_MANIFEST = "review-source/frozen-data/MANIFEST.json"
REGISTER = "review-source/original-source/research/mathematical-engineering/langacquire097/REGISTER.json"
SCORED = "evaluation/SCORED_ROWS.json"
PREDICTION_SEAL = "evaluation/ALL_PREDICTIONS_SEALED.json"
CHECKPOINT_SEAL = "evaluation/ALL_CHECKPOINTS_SEALED.json"
REASONS = ("illegal_header", "missing_delimiter", "non_digit_payload",
           "payload_length", "valid_wrong_shape", "valid_wrong_cells", "exact")
METRICS = ("outputs", "delimiter_present", "header_legal", "header_matches_gold",
           "row_matches_gold", "column_matches_gold", "format_valid",
           "valid_shape_matches_gold", "exact_grid",
           "gold_dimension_and_cell_prefix_exact_ignoring_delimiter",
           "correct_first_delimiter_position")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def json_member(archive, name):
    return json.loads(archive.read(name))


def npz_member(archive, name):
    with np.load(BytesIO(archive.read(name)), allow_pickle=False) as value:
        metadata = json.loads(str(value["metadata"].item()))
        arrays = {key: value[key].copy() for key in value.files if key != "metadata"}
    return arrays, metadata


def decode(raw):
    """Independent implementation of the parent's strict first-semicolon rule."""
    raw = bytes(raw)
    delimiter = raw.find(b";")
    if delimiter < 0:
        return None
    prefix = raw[:delimiter]
    if len(prefix) < 3 or prefix[0] not in b"123456" or prefix[1] not in b"123456":
        return None
    rows, columns = prefix[0] - 48, prefix[1] - 48
    cells = prefix[2:]
    if len(cells) != rows * columns or any(value not in b"0123456789" for value in cells):
        return None
    return rows, columns, tuple(value - 48 for value in cells)


def decompose(raw, gold):
    raw, gold = bytes(raw), bytes(gold)
    expected = decode(gold)
    require(expected is not None, "original gold output does not satisfy the fixed parser")
    header = len(raw) >= 2 and raw[0] in b"123456" and raw[1] in b"123456"
    delimiter = raw.find(b";")
    prefix = raw[:delimiter] if delimiter >= 0 else raw
    predicted = decode(raw)
    shape_matches = raw[:2] == gold[:2]
    if not header:
        reason = "illegal_header"
    elif delimiter < 0:
        reason = "missing_delimiter"
    elif any(value not in b"0123456789" for value in prefix[2:]):
        reason = "non_digit_payload"
    elif len(prefix[2:]) != (raw[0] - 48) * (raw[1] - 48):
        reason = "payload_length"
    elif not shape_matches:
        reason = "valid_wrong_shape"
    elif predicted != expected:
        reason = "valid_wrong_cells"
    else:
        reason = "exact"
    metrics = {
        "outputs": 1,
        "delimiter_present": int(delimiter >= 0),
        "header_legal": int(header),
        "header_matches_gold": int(shape_matches),
        "row_matches_gold": int(raw[0] == gold[0]),
        "column_matches_gold": int(raw[1] == gold[1]),
        "format_valid": int(predicted is not None),
        "valid_shape_matches_gold": int(predicted is not None and shape_matches),
        "exact_grid": int(predicted == expected),
        # This is retrospective error classification, not a repaired prediction.
        # Gold lengths never enter a model call or replace the original endpoint.
        "gold_dimension_and_cell_prefix_exact_ignoring_delimiter":
            int(raw[:len(gold) - 1] == gold[:-1]),
        "correct_first_delimiter_position": int(delimiter == len(gold) - 1),
    }
    return {"metrics": metrics, "exclusive_first_failure": reason,
            "first_delimiter_offset": delimiter if delimiter >= 0 else None,
            "predicted_shape": list(predicted[:2]) if predicted is not None else None,
            "gold_shape": list(expected[:2]), "generated_bytes_hex": raw.hex(),
            "gold_bytes_hex": gold.hex()}


def summarize(counter):
    return {"metrics": {name: counter[name] for name in METRICS},
            "exclusive_first_failures": {reason: counter["reason:" + reason] for reason in REASONS}}


def diagnose(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "regular original result archive required")
    require(path.stat().st_size == ARCHIVE_BYTES and file_digest(path) == ARCHIVE_SHA256,
            "archive is not the exact preserved LANGACQUIRE097 parent result")
    totals = Counter()
    grouped = defaultdict(Counter)
    rows = []
    payload_hashes = {}
    with zipfile.ZipFile(path) as archive:
        require(len(archive.namelist()) == len(set(archive.namelist())), "duplicate ZIP member")
        actor, judge = json_member(archive, ACTOR), json_member(archive, JUDGE)
        register = json_member(archive, REGISTER)
        manifest = json_member(archive, DATA_MANIFEST)
        require(len(actor) == len(judge) == 72, "complete parent DEV cohort required")
        require(Counter(row["domain"] for row in actor) == {"lexical": 64, "arc": 8},
                "parent DEV domains differ")
        require(digest(archive.read(DATA_MANIFEST)) == register["data"]["manifest_sha256"],
                "parent frozen manifest binding differs")
        for name, member in (("DEV_ACTOR.json", ACTOR), ("DEV_JUDGE.json", JUDGE)):
            require(digest(archive.read(member)) == manifest["artifact_sha256"][name],
                    "parent frozen actor/judge bytes differ")
        score_rows = [row for row in json_member(archive, SCORED)
                      if row["domain"] == "arc" and row["condition"] == "carry"]
        scored = {(row["family"], row["objective"], row["seed"], row["case_id"]): row
                  for row in score_rows}
        require(len(scored) == len(score_rows) == 320, "complete saved carried ARC score matrix required")
        expected_models = {(f, o, s) for f in FAMILIES for o in OBJECTIVES for s in SEEDS}
        replay_seal = json_member(archive, PREDICTION_SEAL)
        checkpoint_seal = json_member(archive, CHECKPOINT_SEAL)
        replays = {(row["family"], row["objective"], row["seed"]): row
                   for row in replay_seal["replays"]}
        checkpoints = {(row["family"], row["objective"], row["seed"]): row
                       for row in checkpoint_seal["models"]}
        require(set(replays) == set(checkpoints) == expected_models,
                "parent forty-model prediction/checkpoint seals differ")
        for family, objective, seed in sorted(expected_models):
            identity = family, objective, seed
            folder = f"evaluation/{family}-{objective}-{seed}"
            original, replay = folder + "/original.npz", folder + "/replay.npz"
            prediction_sha = digest(archive.read(original))
            require(prediction_sha == replays[identity]["prediction_sha256"]
                    and digest(archive.read(replay)) == replays[identity]["replay_sha256"],
                    "parent sealed prediction/replay bytes differ")
            arrays, metadata = npz_member(archive, original)
            repeated, repeated_metadata = npz_member(archive, replay)
            require(set(arrays) == set(repeated)
                    and all(np.array_equal(value, repeated[key]) for key, value in arrays.items()),
                    "parent fresh-process saved replay differs")
            require(metadata["case_ids"] == repeated_metadata["case_ids"] == [row["id"] for row in actor]
                    and metadata["domains"] == [row["domain"] for row in actor],
                    "saved prediction case ordering differs")
            checkpoint = metadata["checkpoint"]
            require((checkpoint["family"], checkpoint["objective"], checkpoint["seed"]) == identity
                    and checkpoint["source_commit"] == PARENT_SOURCE
                    and checkpoint["optimizer_steps"] == 256 and metadata["optimizer_steps"] == 0
                    and metadata["checkpoint_sha256"] == checkpoints[identity]["checkpoint_sha256"],
                    "saved model/source binding differs")
            require(arrays["generated_bytes"].shape == (5, 72, 40)
                    and arrays["generated_bytes"].dtype == np.uint8
                    and arrays["log_probs"].shape == (5, 72, 256), "saved prediction geometry differs")
            carry = metadata["conditions"].index("carry")
            payload_hashes[f"{family}-{objective}-{seed}"] = prediction_sha
            group = family + "/" + objective
            for index, observed in enumerate(actor):
                if observed["domain"] != "arc":
                    continue
                require(bool(arrays["available"][carry, index])
                        and int(arrays["lengths"][index]) == 40, "parent carried prediction unavailable")
                raw = bytes(arrays["generated_bytes"][carry, index, :40])
                require(raw[0] == int(arrays["log_probs"][carry, index].argmax()),
                        "saved greedy first byte differs from stored distribution")
                gold = judge[observed["id"]]
                row = decompose(raw, gold["target_bytes"])
                saved = scored[(*identity, observed["id"])]
                require(row["metrics"]["format_valid"] == saved["format_valid"]
                        and row["metrics"]["exact_grid"] == saved["exact_match"],
                        "posthoc parser decomposition differs from the original score")
                row.update(family=family, objective=objective, seed=seed,
                           case_id=observed["id"], original_task=gold["cluster_id"],
                           prediction_member=original, prediction_sha256=prediction_sha)
                rows.append(row)
                for counter in (totals, grouped[group]):
                    counter.update(row["metrics"])
                    counter["reason:" + row["exclusive_first_failure"]] += 1
    require(len(rows) == 320, "complete carried ARC decomposition required")
    return {
        "schema": "langcontrol098.parent-arc-posthoc-diagnostic.v1",
        "status": "COMPLETED_SAVED_OUTPUT_REANALYSIS",
        "scope": "Posthoc read-only prior LANGACQUIRE097 outputs; separate from frozen LANGCONTROL098 endpoints.",
        "evidence_level": "INTERNAL_POSTHOC_ERROR_DIAGNOSIS_NOT_NEW_CAPABILITY_OR_INDEPENDENT_QUALIFICATION",
        "archive": {"artifact_id": 13019296161, "bytes": ARCHIVE_BYTES, "sha256": ARCHIVE_SHA256,
                    "source_url": "https://github.com/FPC-effortless/TAC-OSM/actions/runs/41141983424/artifacts/13019296161",
                    "parent_training_source": PARENT_SOURCE,
                    "actor_member": ACTOR, "judge_member": JUDGE,
                    "data_manifest_member": DATA_MANIFEST, "register_member": REGISTER,
                    "original_scores_member": SCORED},
        "diagnostic_script_sha256": file_digest(__file__),
        "population": {"model_families": 4, "objectives": 2, "seeds": 5,
                       "original_ARC_tasks": 8, "carried_outputs": 320,
                       "independence_note": "320 outputs repeat eight tasks across models/objectives/seeds; they are not 320 independent tasks."},
        "exclusive_category_order": list(REASONS),
        "totals": summarize(totals),
        "by_family_objective": {key: summarize(value) for key, value in sorted(grouped.items())},
        "saved_prediction_sha256": payload_hashes,
        "rows": rows,
        "verification": {"archive_hash_matches_preserved_parent": True,
                         "saved_model_replay_pairs_equal": 40,
                         "carried_ARC_rows_match_original_scorer": 320,
                         "neural_forward_calls": 0, "optimizer_steps": 0,
                         "LANGCONTROL098_outcomes_read": False,
                         "frozen_evaluator_or_endpoint_changed": False},
        "interpretation_limits": [
            "First-byte ARC NLL predicts only the row-count byte, not cell content or full syntax.",
            "A missing delimiter is not evidence that the grid content was correct.",
            "Ignoring the delimiter checks an already-generated gold-shaped prefix; it does not repair a prediction or change original exact success.",
            "Saved greedy strings and the first-token distribution cannot reconstruct constrained-decoding outcomes or later teacher-forced NLL without new model calls.",
            "These posthoc categories localize observed errors; they do not identify an internal causal mechanism or establish hidden-rule acquisition.",
        ],
    }


def markdown(report):
    totals = report["totals"]["metrics"]
    lines = ["# Parent097 ARC output diagnosis", "",
             "Posthoc read-only analysis of preserved LANGACQUIRE097 predictions, prepared separately from LANGCONTROL098 outcomes. No model calls or optimizer steps; frozen098 endpoints are unchanged.", "",
             "The original archive SHA256 is `" + ARCHIVE_SHA256 + "` (135,590,835 bytes; artifact13019296161). The script validates frozen actor/judge hashes, forty prediction/replay pairs, checkpoint identities and all320 carried ARC score labels.", "",
             "| Measurement | Outputs /320 |", "| --- | ---: |"]
    labels = (("Legal dimension header", "header_legal"), ("Correct dimensions", "header_matches_gold"),
              ("Contains semicolon", "delimiter_present"), ("Valid complete grid", "format_valid"),
              ("Exact correct grid", "exact_grid"),
              ("Correct dimension-and-cell prefix even ignoring delimiter", "gold_dimension_and_cell_prefix_exact_ignoring_delimiter"))
    lines.extend("| " + label + " | " + str(totals[key]) + " |" for label, key in labels)
    lines += ["", "Exclusive first-failure categories, in the script's declared order:", "",
              "| Category | Outputs |", "| --- | ---: |"]
    lines.extend("| " + reason + " | " + str(value) + " |"
                 for reason, value in report["totals"]["exclusive_first_failures"].items())
    lines += ["", "The one valid carried grid was Transformer/query/seed1705/task9565186b: `33300000000;` versus gold `33535335535;`. Both declare3×3; the generated cells are wrong. None of the320 strings already contains the complete correct dimension-and-cell prefix, even when the final delimiter is ignored. The observed failures therefore cannot be described as only a delimiter problem.", "",
              "First-byte NLL evaluates the row-count character. Header, stopping, length and cell errors can be decomposed from saved bytes; outcomes under constrained decoding and probabilities of later correct tokens require new model calls and are not estimated here.", "",
              "The320 outputs repeat eight original tasks across four families, two objectives and five seeds. They are not320 independent tasks. This is error diagnosis of a prior negative, not a new capability result or evidence for an architecture winner.", "",
              "Reproduce using the exact original archive:", "",
              "```bash",
              "python research/mathematical-engineering/langcontrol098/results/parent_diagnostics/diagnose_parent_arc_serialization.py --archive /path/to/LANGACQUIRE097_original_results.zip --output /new/path/PARENT_ARC_DIAGNOSIS.json --markdown /new/path/PARENT_ARC_DIAGNOSIS.md",
              "```", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    require(args.output.resolve() != args.markdown.resolve(), "distinct output paths required")
    require(not args.output.exists() and not args.markdown.exists(), "never overwrite an earlier diagnosis")
    report = diagnose(args.archive)
    for path, content in ((args.output, json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"),
                          (args.markdown, markdown(report))):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            stream.write(content)
    print(json.dumps({"status": report["status"], "totals": report["totals"],
                      "neural_forward_calls": 0, "optimizer_steps": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
