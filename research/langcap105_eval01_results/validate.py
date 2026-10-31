"""Reproduce published score arithmetic; never load weights or score models."""
import argparse
import json
import math
import os
from pathlib import Path
from pathlib import PurePosixPath
import stat
import sys
import tempfile
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "langcap105_eval01"))
from common import EVAL_PROTOCOL_SHA256, REGISTRATION_COMMIT, digest, load_plan, require
from aggregate import run as aggregate

EXPECTED_EVALUATOR = "9866092ff7e44baf7de26a3cf0c277abd7316fe3"
EXPECTED_RUN = 41238556668


def equal(actual, expected, path="root"):
    if isinstance(actual, float) and type(expected) in (int, float):
        require(math.isfinite(actual) and math.isfinite(expected) and
                math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), "Metric changed: " + path)
    elif isinstance(actual, dict) and isinstance(expected, dict):
        require(actual.keys() == expected.keys(), "Fields changed: " + path)
        for key in actual: equal(actual[key], expected[key], path + "." + key)
    elif isinstance(actual, list) and isinstance(expected, list):
        require(len(actual) == len(expected), "Array length changed: " + path)
        for i, (a, b) in enumerate(zip(actual, expected)): equal(a, b, path + "." + str(i))
    else:
        require(type(actual) is type(expected) and actual == expected, "Value changed: " + path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-remote", action="store_true")
    args = parser.parse_args()
    plan = load_plan()
    receipt = json.loads((HERE / "ARTIFACT_RECEIPT.json").read_text())
    require(receipt["evaluation_run"] == EXPECTED_RUN and receipt["evaluator_commit"] == EXPECTED_EVALUATOR,
            "Wrong original evaluation run/source")
    require(receipt["registration_commit"] == REGISTRATION_COMMIT, "Wrong preregistration")
    require(digest(HERE / "EVAL_PROTOCOL.json") == EVAL_PROTOCOL_SHA256, "Published preregistration differs")
    with tempfile.TemporaryDirectory(prefix="langcap105-eval01-recompute-") as temp:
        original = Path(temp) / "original-evidence"
        archive = HERE / "ORIGINAL_EVIDENCE.zip"
        require(digest(archive) == receipt["archive_sha256"], "Original evidence archive bytes changed")
        with zipfile.ZipFile(archive) as z:
            require(sum(info.file_size for info in z.infolist()) < 64*1024*1024, "Unexpected evidence size")
            for info in z.infolist():
                path = PurePosixPath(info.filename)
                require(not path.is_absolute() and ".." not in path.parts and "\\" not in info.filename and
                        ":" not in info.filename and not stat.S_ISLNK(info.external_attr >> 16), "Unsafe evidence member")
            require(z.testzip() is None, "Original evidence archive CRC failed")
            z.extractall(original)
        require(digest(original / "EVAL_PROTOCOL.json") == EVAL_PROTOCOL_SHA256, "Archived preregistration differs")
        for path in (original / "individual-results").rglob("RESULT.json"):
            result = json.loads(path.read_text())
            require(result["evaluator_commit"] == EXPECTED_EVALUATOR and result["registration_commit"] == REGISTRATION_COMMIT,
                    "Unregistered evaluator source")
            require(result["source_sha256"] == plan["source"]["source_sha256"], "Source receipt differs")
        out = Path(temp) / "recomputed"
        aggregate(original / "individual-results", out)
        equal(json.loads((out / "SUMMARY.json").read_text()), json.loads((HERE / "metrics/SUMMARY.json").read_text()))
        equal(json.loads((original / "metrics/SUMMARY.json").read_text()), json.loads((HERE / "metrics/SUMMARY.json").read_text()))
    if args.verify_remote:
        token = os.environ.get("GH_READ_TOKEN")
        require(bool(token), "Read-only Actions identity is required for remote provenance verification")
        base = "https://api.github.com/repos/FPC-effortless/TAC-OSM"
        def get(path):
            request = urllib.request.Request(base + path, headers={"Authorization":"Bearer " + token,
                                            "Accept":"application/vnd.github+json"})
            with urllib.request.urlopen(request, timeout=30) as response: return json.load(response)
        run = get("/actions/runs/" + str(EXPECTED_RUN))
        require(run["conclusion"] == "success" and run["status"] == "completed" and
                run["head_sha"] == EXPECTED_EVALUATOR, "Original scoring run was not a successful frozen execution")
        artifact = get("/actions/artifacts/" + str(int(receipt["artifact_id"])))
        require(artifact["name"] == "LANGCAP105-EVAL01-final-evidence" and
                artifact["digest"] == "sha256:" + receipt["archive_sha256"] and
                artifact["workflow_run"]["head_sha"] == EXPECTED_EVALUATOR,
                "Original artifact digest/source differs")
    print(json.dumps({"status":"ALL_18_PUBLISHED_RESULTS_REPRODUCED_WITHOUT_MODEL_FORWARD",
        "model_prediction_calls":0, "optimizer_updates":0,
        "remote_run_and_archive_digest":"PASS" if args.verify_remote else "NOT_REQUESTED",
        "independent_scientific_qualification":False}))


if __name__ == "__main__": main()
