"""Frozen public development battery: ARC training, GSM8K train, HumanEval.

This is a small, exposed capability diagnostic, not a general-intelligence
certificate or a blind benchmark. Judge outputs/references are written to a
separate file; only acquisition_bytes/query_bytes are model observations.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen
import zipfile

SALT = "LANGPERSIST096-GENERAL-96002"
PER_DOMAIN = 8
MAX_ASSET_BYTES = 32 * 1024 * 1024
DOMAINS = ("arc", "gsm8k", "humaneval")
ASSETS = {
    "arc.zip": {
        "repository": "fchollet/ARC-AGI", "commit": "399030444e0ab0cc8b4e199870fb20b863846f34",
        "path": "archive snapshot; only data/training/*.json consumed",
        "url": "https://codeload.github.com/fchollet/ARC-AGI/zip/399030444e0ab0cc8b4e199870fb20b863846f34",
        "sha256": "c57507f289aa398fc22547901ee05af61c8e71e4fe6795b73ee4dd5795bd8950",
        "license": "Apache-2.0", "license_asset": "arc.LICENSE"},
    "arc.LICENSE": {
        "repository": "fchollet/ARC-AGI", "commit": "399030444e0ab0cc8b4e199870fb20b863846f34",
        "path": "LICENSE", "license": "Apache-2.0",
        "url": "https://raw.githubusercontent.com/fchollet/ARC-AGI/399030444e0ab0cc8b4e199870fb20b863846f34/LICENSE",
        "sha256": "b3a161754ffc20840ccf92aec0fe464fbbcf42f14c22f78a3ec2e98fca3864ca"},
    "gsm8k.train.jsonl": {
        "repository": "openai/grade-school-math", "commit": "3101c7d5072418e28b9008a6636bde82a006892c",
        "path": "grade_school_math/data/train.jsonl", "license": "MIT", "license_asset": "gsm8k.LICENSE",
        "url": "https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/train.jsonl",
        "sha256": "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465"},
    "gsm8k.LICENSE": {
        "repository": "openai/grade-school-math", "commit": "3101c7d5072418e28b9008a6636bde82a006892c",
        "path": "LICENSE", "license": "MIT",
        "url": "https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/LICENSE",
        "sha256": "86bbb73e855821d7c401912fd4bf82e34313e6e3b6fd6f909f2b6cc9e209a53b"},
    "humaneval.jsonl.gz": {
        "repository": "openai/human-eval", "commit": "b9453e8f1865df0a0057a1383767caa400acfbcf",
        "path": "data/HumanEval.jsonl.gz", "license": "MIT", "license_asset": "humaneval.LICENSE",
        "url": "https://raw.githubusercontent.com/openai/human-eval/b9453e8f1865df0a0057a1383767caa400acfbcf/data/HumanEval.jsonl.gz",
        "sha256": "b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef"},
    "humaneval.LICENSE": {
        "repository": "openai/human-eval", "commit": "b9453e8f1865df0a0057a1383767caa400acfbcf",
        "path": "LICENSE", "license": "MIT",
        "url": "https://raw.githubusercontent.com/openai/human-eval/b9453e8f1865df0a0057a1383767caa400acfbcf/LICENSE",
        "sha256": "bcba3de214851cce46ed5af42d6698044616eeace887c3231bc7a20474ab639e"},
}


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _immutable(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or path.read_bytes() != raw:
            raise ValueError(f"existing general battery artifact differs: {path}")
        return
    fd, temporary = tempfile.mkstemp(prefix=".general-data-", dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or path.read_bytes() != raw:
                raise ValueError("concurrent general artifact differs") from None
    finally:
        temporary.unlink(missing_ok=True)


def _asset(cache, name):
    specification = ASSETS[name]
    path = cache / "source" / name
    if path.exists() or path.is_symlink():
        if path.is_symlink():
            raise ValueError("source asset symlink refused")
        raw = path.read_bytes()
    else:
        with urlopen(specification["url"], timeout=45) as response:
            raw = response.read(MAX_ASSET_BYTES + 1)
    if len(raw) > MAX_ASSET_BYTES or _hash(raw) != specification["sha256"]:
        raise ValueError(f"public source asset hash/size mismatch: {name}")
    _immutable(path, raw)
    return raw


def rank_task(domain, task_id):
    return _hash((SALT + "\0" + domain + "\0" + task_id).encode())


def _select(domain, eligible):
    if len({case[0]["task_id"] for case in eligible}) != len(eligible):
        raise ValueError("duplicate public task identity")
    if len(eligible) < PER_DOMAIN:
        raise ValueError(f"fewer than eight structurally eligible {domain} tasks")
    return sorted(eligible, key=lambda case: rank_task(domain, case[0]["task_id"]))[:PER_DOMAIN]


def _actor(domain, task_id, acquisition, query):
    return {"domain": domain, "task_id": task_id, "acquisition_bytes": list(acquisition.encode("utf-8")),
            "query_bytes": list(query.encode("utf-8")), "generation_limit": 128}


def _grid_valid(grid):
    return (isinstance(grid, list) and 1 <= len(grid) <= 6
            and isinstance(grid[0], list) and 1 <= len(grid[0]) <= 6
            and all(isinstance(row, list) and len(row) == len(grid[0])
                    and all(type(value) is int and 0 <= value <= 9 for value in row) for row in grid))


def grid_text(grid):
    if not _grid_valid(grid):
        raise ValueError("ARC grid must be rectangular 1..6 by 1..6 with integer colors0..9")
    return "\n".join("".join(str(v) for v in row) for row in grid)


def build_arc_cases(archive_bytes):
    """Inspect only ARC training-task members; final test output is judge-only."""
    eligible = []
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        for name in sorted(archive.namelist()):
            if "/data/training/" not in name or not name.endswith(".json"):
                continue
            if archive.getinfo(name).file_size > 1024 * 1024:
                raise ValueError("oversized ARC training member")
            raw = archive.read(name)
            record = json.loads(raw)
            demonstrations, tests = record.get("train", []), record.get("test", [])
            if len(demonstrations) < 2 or not tests:
                continue
            selected_test = tests[0]
            examples = demonstrations + [selected_test]
            if not all(isinstance(ex, dict) and all(_grid_valid(ex.get(k)) for k in ("input", "output"))
                       for ex in examples):
                continue
            acquisition = "Examples:\n" + "".join(
                "Input:\n" + grid_text(ex["input"]) + "\nOutput:\n" + grid_text(ex["output"]) + "\n\n"
                for ex in demonstrations)
            query = "Input:\n" + grid_text(selected_test["input"]) + "\nOutput:\n"
            if (len(acquisition.encode()) > 1024 or len(query.encode()) > 256
                    or len(grid_text(selected_test["output"]).encode()) > 128):
                continue
            source_id = Path(name).stem
            task_id = f"arc/train/{source_id}/test0"
            actor = _actor("arc", task_id, acquisition, query)
            judge = {"domain": "arc", "expected": selected_test["output"],
                     "source_path": f"data/training/{source_id}.json", "source_file_sha256": _hash(raw),
                     "official_test_case_index": 0, "demonstration_count": len(demonstrations)}
            eligible.append((actor, judge))
    return _select("arc", eligible), len(eligible)


def _json_lines(raw):
    for index, line in enumerate(raw.splitlines()):
        if line.strip():
            yield index, json.loads(line), line


def build_gsm8k_cases(raw):
    eligible = []
    for index, record, line in _json_lines(raw):
        question, solution = record.get("question"), record.get("answer")
        if not isinstance(question, str) or not isinstance(solution, str) or "####" not in solution:
            raise ValueError("malformed GSM8K train record")
        query = question + "\nAnswer:"
        answer = solution.rsplit("####", 1)[1].strip()
        if not answer:
            raise ValueError("empty GSM8K final answer")
        if len(query.encode()) > 1024:
            continue
        task_id = f"gsm8k/train/{index}"
        actor = _actor("gsm8k", task_id,
                       "Solve the word problem. Give the final numerical answer after Answer:.\n", query)
        judge = {"domain": "gsm8k", "answer": answer,
                 "source_path": ASSETS["gsm8k.train.jsonl"]["path"],
                 "source_line_index": index, "source_line_sha256_without_lf": _hash(line)}
        eligible.append((actor, judge))
    return _select("gsm8k", eligible), len(eligible)


def build_humaneval_cases(compressed):
    raw = gzip.decompress(compressed)
    if len(raw) > MAX_ASSET_BYTES:
        raise ValueError("oversized decompressed HumanEval asset")
    eligible = []
    for _, record, line in _json_lines(raw):
        keys = ("task_id", "prompt", "canonical_solution", "test", "entry_point")
        if any(not isinstance(record.get(k), str) for k in keys):
            raise ValueError("malformed public HumanEval record")
        if len(record["prompt"].encode()) > 1024:
            continue
        actor = _actor("humaneval", record["task_id"],
                       "Complete the Python function. Output only the function body continuation.\n",
                       record["prompt"])
        judge = {k: record[k] for k in ("prompt", "canonical_solution", "test", "entry_point")}
        judge.update(domain="humaneval", source_path=ASSETS["humaneval.jsonl.gz"]["path"],
                     source_line_sha256_without_lf=_hash(line))
        # Selection never depends on reference executability or model generations.
        eligible.append((actor, judge))
    return _select("humaneval", eligible), len(eligible)


def validate_actor_records(records):
    allowed = {"domain", "task_id", "acquisition_bytes", "query_bytes", "generation_limit"}
    counts = {domain: 0 for domain in DOMAINS}
    identities = set()
    for actor in records:
        if set(actor) != allowed or actor["domain"] not in DOMAINS or actor["task_id"] in identities:
            raise ValueError("actor schema/identity violation")
        identities.add(actor["task_id"])
        counts[actor["domain"]] += 1
        for key in ("acquisition_bytes", "query_bytes"):
            if not isinstance(actor[key], list) or not actor[key] or any(
                    type(v) is not int or not 0 <= v <= 255 for v in actor[key]):
                raise ValueError("model observations must be explicit nonempty byte arrays")
        if len(actor["acquisition_bytes"]) > 1024 or len(actor["query_bytes"]) > (
                256 if actor["domain"] == "arc" else 1024) or actor["generation_limit"] != 128:
            raise ValueError("actor byte/generation budget exceeded")
    if counts != {domain: PER_DOMAIN for domain in DOMAINS}:
        raise ValueError("fixed eight-task-per-domain matrix required")
    return counts


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode()


def prepare_general_data(cache_dir):
    cache = Path(cache_dir).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    source = {name: _asset(cache, name) for name in ASSETS}
    builders = {"arc": lambda: build_arc_cases(source["arc.zip"]),
                "gsm8k": lambda: build_gsm8k_cases(source["gsm8k.train.jsonl"]),
                "humaneval": lambda: build_humaneval_cases(source["humaneval.jsonl.gz"])}
    actors, judges, eligible = [], {}, {}
    for domain in DOMAINS:
        cases, eligible[domain] = builders[domain]()
        for actor, judge in cases:
            actors.append(actor)
            judges[actor["task_id"]] = judge
    counts = validate_actor_records(actors)
    actor_raw, judge_raw = _json_bytes(actors), _json_bytes(judges)
    actor_path, judge_path = cache / "GENERAL_ACTOR.json", cache / "GENERAL_JUDGE.json"
    _immutable(actor_path, actor_raw)
    _immutable(judge_path, judge_raw)
    manifest = {
        "study": "LANGPERSIST096_GENERAL", "status": "FROZEN_PUBLIC_DEVELOPMENT_DATA",
        "classification": "PUBLIC_DEVELOPMENT_EXPOSED_NOT_BLIND", "selection_salt": SALT,
        "selection": "eight smallest salted SHA256 task identities per domain after fixed structural limits",
        "selection_uses_model_metrics": False, "task_counts": counts, "eligible_counts": eligible,
        "task_ids": [actor["task_id"] for actor in actors], "source_assets": ASSETS,
        "source_asset_bytes": {name: len(raw) for name, raw in source.items()},
        "actor_path": str(actor_path), "actor_sha256": _hash(actor_raw),
        "judge_path": str(judge_path), "judge_sha256": _hash(judge_raw),
        "attribution": {
            "arc": "ARC-AGI by Francois Chollet; Apache-2.0; training-task demonstrations and public queries",
            "gsm8k": "GSM8K by OpenAI; MIT; train split only",
            "humaneval": "HumanEval by OpenAI; MIT; public code completion prompts and functional tests"},
        "prediction_boundary": "Only acquisition_bytes and query_bytes; task IDs, gold, tests and references are metadata/judge-only",
        "protected_repository_TEST": "NOT_OPENED", "new_training": False,
        "qualification": "INTERNAL_DEVELOPMENT_NOT_INDEPENDENTLY_QUALIFIED",
        "general_intelligence_claim": False,
        "limitations": ["24 exposed public tasks do not establish general intelligence",
                        "ARC transformation rules are not given, but input/output demonstrations are provided",
                        "GSM8K and HumanEval test problem solving rather than learned persistent-structure acquisition",
                        "Historical source exposure is not fully audited; no uncontaminated benchmark claim",
                        "Unsupported functional reference checks remain INCONCLUSIVE; task selection is not revised"]}
    _immutable(cache / "GENERAL_DATA.json", _json_bytes(manifest))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, type=Path)
    args = parser.parse_args()
    result = prepare_general_data(args.cache_dir)
    print(json.dumps({k: result[k] for k in ("status", "task_counts", "eligible_counts", "actor_sha256",
                                           "judge_sha256")}, indent=2))
