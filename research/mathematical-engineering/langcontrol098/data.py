"""Authenticated, pre-model LANGACQUIRE097 acquisition/query episodes.

Real publisher code is the primary task. ARC is a secondary, human-authored
benchmark diagnostic, never evidence of operational or general intelligence.
Only FIT.json contains training answers. DEV_ACTOR.json contains no answers,
source identities, strata, offsets or operator labels; DEV_JUDGE.json is owned
by the separate post-checkpoint evaluator. No neural library is imported.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CONTRACT_PATH = HERE / "SOURCE_PROVENANCE.json"
SALT = "LANGACQUIRE097-DATA-97001"
ACQUISITION, QUERY, TARGET = 64, 32, 4
GROUPS = ("positive", "negative", "uniform")
MAX_SOURCE_BYTES = 32 * 1024 * 1024
_SOURCE_IMPORT_LOCK = threading.Lock()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True,
                       allow_nan=False) + "\n").encode()


def rank(*parts):
    return digest(json.dumps([SALT, *parts], separators=(",", ":"),
                             ensure_ascii=True).encode())


def opaque_episode_id(original_identity):
    """Stable handle without filenames, source paths, task IDs or targets."""
    return "097-" + digest((SALT + "\0actor\0" + original_identity).encode())


def contract():
    value = json.loads(CONTRACT_PATH.read_text())
    if value["salt"] != SALT:
        raise ValueError("selection salt changed")
    for relative, expected in value["source_support_sha256"].items():
        path = ROOT / relative
        if path.is_symlink() or digest(path.read_bytes()) != expected:
            raise ValueError("reviewed source support changed: " + relative)
    return value


def _lexical_source():
    contract()  # Authenticate the reviewed tokenizer before executing it.
    name = "langacquire097_reviewed_lexical_source"
    with _SOURCE_IMPORT_LOCK:
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(
                name, ROOT / "research/mathematical-engineering/langpersist096/data.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.modules[name] = module
    return sys.modules[name]


def immutable(path, raw):
    """Create once, or verify byte equality; never overwrite a prior cohort."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or path.read_bytes() != raw:
            raise ValueError("sealed artifact differs: " + str(path))
        return
    fd, temporary = tempfile.mkstemp(prefix=".langacquire097-", dir=path.parent)
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
                raise ValueError("concurrent artifact differs") from None
    finally:
        temporary.unlink(missing_ok=True)


def _read_source(cache, relative, specification, bootstrap=None):
    destination = cache / "sources" / relative
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink():
            raise ValueError("source symlink refused")
        raw = destination.read_bytes()
    elif bootstrap is not None:
        bootstrap = Path(bootstrap)
        if bootstrap.is_symlink():
            raise ValueError("bootstrap source symlink refused")
        raw = bootstrap.read_bytes()
    else:
        with urllib.request.urlopen(specification["url"], timeout=45) as response:
            raw = response.read(MAX_SOURCE_BYTES + 1)
    if len(raw) > MAX_SOURCE_BYTES or digest(raw) != specification["sha256"]:
        raise ValueError("publisher source identity differs: " + str(relative))
    if "size" in specification:
        _lexical_source().authenticate_bytes(raw, specification)
    immutable(destination, raw)
    return raw


def _interval_free(start, end, occupied):
    return all(not (start < old_end and old_start < end)
               for old_start, old_end in occupied)


def lexical_candidates(raw, excluded_names=frozenset()):
    """Scorer-only eligibility over exact Python NAME spans, not substrings."""
    spans = _lexical_source().ascii_name_spans(raw)
    occurrences = {}
    for start, end, name in spans:
        occurrences.setdefault(name, []).append((start, end))
    candidates = {group: [] for group in GROUPS[:2]}
    for offset, end, name in spans:
        if (offset < ACQUISITION + QUERY or offset + TARGET > len(raw)
                or not 4 <= len(name) <= 24 or name in excluded_names):
            continue
        acquisition_start, query_start = offset - ACQUISITION - QUERY, offset - QUERY
        if name in raw[query_start:offset]:
            continue
        older = next(((lo, hi) for lo, hi in occurrences[name]
                      if lo >= acquisition_start and hi <= query_start), None)
        if older is not None:
            candidates["positive"].append((offset, name, older[0]))
        elif name not in raw[acquisition_start:query_start]:
            candidates["negative"].append((offset, name, None))
    return candidates


def _quota(total, owner_index, owners):
    return total // owners + (owner_index < total % owners)


def build_lexical(documents, source_rows, policy):
    source = _lexical_source()
    if len({row["path"] for row in source_rows}) != len(source_rows):
        raise ValueError("duplicate code source owner")
    if set(documents) != {row["path"] for row in source_rows}:
        raise ValueError("code source coverage differs")
    fit_rows = [row for row in source_rows if row["partition"] == "FIT"]
    inventory = source.fit_identifier_inventory({r["path"]: documents[r["path"]] for r in fit_rows})
    selected, diagnostics, failures = [], [], []
    for partition in ("FIT", "DEV"):
        rows = [row for row in source_rows if row["partition"] == partition]
        if not rows:
            raise ValueError("each partition requires whole source files")
        totals = policy["fit_counts" if partition == "FIT" else "dev_counts"]
        for owner_index, row in enumerate(rows):
            raw = documents[row["path"]]
            candidates = lexical_candidates(raw, inventory if partition == "DEV" else frozenset())
            occupied = []
            for group in GROUPS:
                quota = _quota(totals[group], owner_index, len(rows))
                pool = (candidates[group] if group != "uniform" else
                        [(i, None, None) for i in range(ACQUISITION + QUERY, len(raw) - TARGET + 1)])
                pool = sorted(pool, key=lambda item: (rank("lexical", partition, group,
                                                          row["path"], item[0]), item[0]))
                chosen = 0
                for offset, name, historical in (pool if quota else []):
                    start, end = offset - ACQUISITION - QUERY, offset + TARGET
                    if not _interval_free(start, end, occupied):
                        continue
                    occupied.append((start, end))
                    original_identity = f"lexical/{partition}/{row['path']}:{offset}"
                    identifier = opaque_episode_id(original_identity)
                    selected.append({"id": identifier, "domain": "lexical", "partition": partition,
                        "acquisition_bytes": list(raw[start:offset - QUERY]),
                        "query_bytes": list(raw[offset - QUERY:offset]),
                        "target_bytes": list(raw[offset:end]), "generation_limit": TARGET,
                        "metadata": {"group": group, "cluster_id": row["path"],
                            "original_episode_id": original_identity,
                            "source_owner": f"{row['repo']}@{row['commit']}:{row['path']}",
                            "source_path": row["path"], "source_file_sha256": row["sha256"],
                            "target_offset": offset, "acquisition_start": start,
                            "query_start": offset - QUERY, "target_end": end,
                            "target_name": name.decode() if name else None,
                            "historical_name_offset": historical,
                            "selection_sha256": rank("lexical", partition, group, row["path"], offset)}})
                    chosen += 1
                    if chosen == quota:
                        break
                diagnostics.append({"partition": partition, "source": row["path"], "group": group,
                                    "eligible": len(pool), "required": quota, "selected": chosen})
                if chosen != quota:
                    failures.append(diagnostics[-1])
    return selected, {"status": "INSUFFICIENT_COHORT" if failures else "COHORT_READY",
        "counts": diagnostics, "failures": failures, "fit_inventory_names": len(inventory),
        "fit_inventory_sha256": digest(b"\n".join(sorted(inventory)) + b"\n"),
        "whole_file_partitions": {p: [r["path"] for r in source_rows if r["partition"] == p]
                                  for p in ("FIT", "DEV")}}


def encode_grid(grid):
    if not (isinstance(grid, list) and 1 <= len(grid) <= 6 and isinstance(grid[0], list)
            and 1 <= len(grid[0]) <= 6 and all(isinstance(row, list) and len(row) == len(grid[0])
            and all(type(v) is int and 0 <= v <= 9 for v in row) for row in grid)):
        raise ValueError("grid must be rectangular1..6 with integer colors0..9")
    return (str(len(grid)) + str(len(grid[0])) + "".join(str(v) for row in grid for v in row)).encode()


def decode_grid(raw):
    raw = bytes(raw)
    if len(raw) < 3 or raw[0] not in b"123456" or raw[1] not in b"123456":
        raise ValueError("invalid compact grid dimensions")
    rows, columns = raw[0] - 48, raw[1] - 48
    cells = raw[2:]
    if len(cells) != rows * columns or any(v not in b"0123456789" for v in cells):
        raise ValueError("invalid compact grid payload")
    return [[v - 48 for v in cells[start:start + columns]]
            for start in range(0, len(cells), columns)]


def decode_arc_output(raw):
    """Fixed semicolon delimiter; no gold-dependent generation or stopping."""
    raw = bytes(raw)
    if b";" not in raw:
        raise ValueError("missing output delimiter")
    return decode_grid(raw.split(b";", 1)[0])


def shuffled_demonstration_outputs(acquisition):
    """Rotate only observed demonstration outputs, preserving byte count.

    This API intentionally accepts no query or future answer. Identical
    observed outputs leave the intervention inconclusive, without resampling.
    """
    raw = bytes(acquisition)
    if not raw.startswith(b"R;") or not raw.endswith(b";"):
        raise ValueError("invalid ARC acquisition envelope")
    parts = raw[2:-1].split(b";")
    pairs = []
    for part in parts:
        if part.count(b">") != 1:
            raise ValueError("invalid demonstration pair")
        before, after = part.split(b">")
        decode_grid(before); decode_grid(after)
        pairs.append((before, after))
    if len(pairs) < 2:
        raise ValueError("at least two observed demonstrations required")
    outputs = [after for _, after in pairs]
    shifted = outputs[1:] + outputs[:1]
    changed = b"R;" + b"".join(before + b">" + after + b";"
                               for (before, _), after in zip(pairs, shifted))
    if len(changed) != len(raw):
        raise ValueError("demonstration shuffle changed acquisition clock")
    return changed, changed != raw


def control_actor_records(actor):
    records = []
    for episode in actor:
        if episode["domain"] == "arc":
            acquisition, available = shuffled_demonstration_outputs(episode["acquisition_bytes"])
            reason = ("Cyclic rotation of observed outputs; query and future answer unused"
                      if available else "INCONCLUSIVE: all observed outputs identical; case retained")
        else:
            acquisition, available = bytes(episode["acquisition_bytes"]), False
            reason = "NOT_APPLICABLE: observed-demonstration shuffle is ARC-only"
        records.append({"id": episode["id"], "acquisition_bytes": list(acquisition),
                        "available": available, "reason": reason})
    return records


def donor_records(actor, judge):
    """Fixed source/clock-compatible donors; never inspect target or stratum."""
    records = []
    for episode in actor:
        candidates = [other for other in actor if other["id"] != episode["id"]
            and other["domain"] == episode["domain"]
            and len(other["acquisition_bytes"]) == len(episode["acquisition_bytes"])
            and judge[other["id"]]["source_owner"] != judge[episode["id"]]["source_owner"]]
        candidates.sort(key=lambda other: (rank("donor", episode["id"], other["id"]), other["id"]))
        available = bool(candidates)
        records.append({"id": episode["id"], "donor_id": candidates[0]["id"] if available else None,
                        "available": available,
                        "reason": ("Different source owner, same domain and observed acquisition length; no future/group filter"
                                   if available else "INCONCLUSIVE: no same-domain clock-compatible other source; case retained")})
    return records


def build_arc(archive_bytes, policy):
    eligible, excluded = [], Counter()
    geometry = {(128, 208): 0, (192, 272): 0, (256, 336): 0}
    prior = set(policy["exclude_langpersist096_tasks"])
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        for name in sorted(archive.namelist()):
            if "/data/training/" not in name or not name.endswith(".json"):
                continue
            source_id = Path(name).stem
            if source_id in prior:
                excluded["prior096task"] += 1
                continue
            if archive.getinfo(name).file_size > 1024 * 1024:
                raise ValueError("oversized ARC source member")
            raw = archive.read(name)
            record = json.loads(raw)
            demonstrations, tests = record.get("train", []), record.get("test", [])
            if len(demonstrations) < policy["minimum_demonstrations"] or not tests:
                excluded["insufficient_demonstrations_or_queries"] += 1
                continue
            try:
                pairs = [(encode_grid(e["input"]), encode_grid(e["output"])) for e in demonstrations]
                query_grid, target_grid = encode_grid(tests[0]["input"]), encode_grid(tests[0]["output"])
            except (KeyError, TypeError, ValueError):
                excluded["grid_geometry"] += 1
                continue
            if (query_grid == target_grid or query_grid in {x for x, _ in pairs}
                    or not any(x != y for x, y in pairs)):
                excluded["identity_or_repeated_query"] += 1
                continue
            acquisition = b"R;" + b"".join(x + b">" + y + b";" for x, y in pairs)
            query, target = query_grid + b">", target_grid + b";"
            for (maximum_a, maximum_total) in geometry:
                if len(acquisition) <= maximum_a and len(acquisition + query + target) <= maximum_total:
                    geometry[maximum_a, maximum_total] += 1
            if (len(acquisition) > policy["maximum_acquisition_bytes"]
                    or len(acquisition + query + target) > policy["maximum_total_bytes"]):
                excluded["byte_budget"] += 1
                continue
            original_identity = f"arc/train/{source_id}/test0"
            eligible.append({"id": opaque_episode_id(original_identity), "domain": "arc",
                "acquisition_bytes": list(acquisition), "query_bytes": list(query),
                "target_bytes": list(target), "generation_limit": policy["generation_limit"],
                "metadata": {"group": "unknown_rule", "cluster_id": source_id,
                    "original_episode_id": original_identity,
                    "source_owner": "fchollet/ARC-AGI@399030444e0ab0cc8b4e199870fb20b863846f34:" + source_id,
                    "source_path": "data/training/" + source_id + ".json",
                    "source_file_sha256": digest(raw), "demonstration_count": len(pairs),
                    "official_test_case_index": 0, "selection_sha256": rank("arc", source_id)}})
    eligible.sort(key=lambda e: (e["metadata"]["selection_sha256"], e["id"]))
    fit_count, dev_count = policy["fit_tasks"], policy["dev_tasks"]
    requested = fit_count + dev_count
    selected = []
    if len(eligible) >= requested:
        for index, episode in enumerate(eligible[:requested]):
            selected.append({**episode, "partition": "FIT" if index < fit_count else "DEV"})
    return selected, {"status": "COHORT_READY" if len(eligible) >= requested else "INSUFFICIENT_COHORT",
        "eligible_tasks": len(eligible), "required_tasks": requested, "excluded": dict(excluded),
        "task_partitions": {p: [e["metadata"]["cluster_id"] for e in selected if e["partition"] == p]
                            for p in ("FIT", "DEV")},
        "pre_model_geometry": [{"maximum_acquisition_bytes": a, "maximum_total_bytes": t,
                                "eligible_tasks": n} for (a, t), n in geometry.items()],
        "unused_eligible_tasks": [e["metadata"]["cluster_id"] for e in eligible[requested:]]}


def validate_episodes(episodes, partition):
    identities = set()
    for episode in episodes:
        if episode["partition"] != partition or episode["id"] in identities:
            raise ValueError("episode identity or partition violation")
        identities.add(episode["id"])
        for key in ("acquisition_bytes", "query_bytes", "target_bytes"):
            if not isinstance(episode[key], list) or not episode[key] or any(
                    type(value) is not int or not 0 <= value <= 255 for value in episode[key]):
                raise ValueError("episode observations must be nonempty byte arrays")
        if episode["domain"] == "lexical":
            if tuple(len(episode[k]) for k in ("acquisition_bytes", "query_bytes", "target_bytes")) != (64, 32, 4):
                raise ValueError("lexical byte geometry changed")
        elif episode["domain"] == "arc":
            if len(episode["acquisition_bytes"]) > 192 or sum(len(episode[k]) for k in
                    ("acquisition_bytes", "query_bytes", "target_bytes")) > 272:
                raise ValueError("ARC byte geometry changed")
            decode_arc_output(episode["target_bytes"])
        else:
            raise ValueError("unknown domain")
    return len(identities)


def prepare(cache, *, code_bootstrap=None, arc_bootstrap=None):
    """Authenticate originals and seal FIT/actor/judge files before any fit."""
    cache = Path(cache)
    specification = contract()
    rows = specification["cpython_files"]
    cache.mkdir(parents=True, exist_ok=True)
    def code(row):
        original = Path(code_bootstrap) / row["file"] if code_bootstrap is not None else None
        return row["path"], _read_source(cache, "cpython/" + row["path"], row, original)
    with ThreadPoolExecutor(max_workers=5) as pool:
        documents = dict(pool.map(code, rows))
    license_row = specification["cpython_license"]
    _read_source(cache, "cpython.LICENSE", license_row,
                 Path(code_bootstrap) / license_row["file"] if code_bootstrap is not None else None)
    arc_sources = {}
    for name, row in specification["arc_assets"].items():
        original = Path(arc_bootstrap) / name if arc_bootstrap is not None else None
        arc_sources[name] = _read_source(cache, name, row, original)
    lexical, lexical_selection = build_lexical(documents, rows, specification["lexical_policy"])
    arc, arc_selection = build_arc(arc_sources["arc.zip"], specification["arc_policy"])
    selection = {"lexical": lexical_selection, "arc": arc_selection}
    immutable(cache / "SELECTION.json", json_bytes(selection))
    if any(item["status"] != "COHORT_READY" for item in selection.values()):
        failure = {"status": "INSUFFICIENT_COHORT", "selection": selection}
        immutable(cache / "INSUFFICIENT_COHORT.json", json_bytes(failure))
        return failure
    episodes = lexical + arc
    fit = [e for e in episodes if e["partition"] == "FIT"]
    dev = [e for e in episodes if e["partition"] == "DEV"]
    validate_episodes(fit, "FIT"); validate_episodes(dev, "DEV")
    fit_owners = {e["metadata"]["source_owner"] for e in fit}
    dev_owners = {e["metadata"]["source_owner"] for e in dev}
    if fit_owners & dev_owners:
        raise ValueError("source/task owner crosses FIT and DEV")
    actor = [{k: e[k] for k in ("id", "domain", "acquisition_bytes", "query_bytes", "generation_limit")}
             for e in dev]
    judge = {e["id"]: {"domain": e["domain"], "target_bytes": e["target_bytes"], **e["metadata"]}
             for e in dev}
    controls = control_actor_records(actor)
    donors = donor_records(actor, judge)
    for control, donor in zip(controls, donors):
        judge[control["id"]]["binding_shuffle_available"] = control["available"]
        judge[donor["id"]]["donor_available"] = donor["available"]
    payloads = {"FIT.json": fit, "DEV_ACTOR.json": actor, "DEV_JUDGE.json": judge,
                "DEV_CONTROL_ACTOR.json": controls, "DEV_DONORS.json": donors}
    identities = {}
    for name, value in payloads.items():
        raw = json_bytes(value); immutable(cache / name, raw); identities[name] = digest(raw)
    manifest = {"schema": "langacquire097.data.v1", "status": "COHORT_READY",
        "source_provenance_sha256": digest(CONTRACT_PATH.read_bytes()),
        "artifact_sha256": identities, "selection_sha256": digest(json_bytes(selection)),
        "counts": {p: dict(Counter(e["domain"] for e in episodes if e["partition"] == p))
                   for p in ("FIT", "DEV")},
        "group_counts": {p: dict(Counter(e["metadata"]["group"] for e in episodes if e["partition"] == p))
                         for p in ("FIT", "DEV")},
        "control_counts": {"donor_available": sum(item["available"] for item in donors),
                           "binding_shuffle_available": sum(item["available"] for item in controls)},
        "fit_owners": sorted(fit_owners), "dev_owners": sorted(dev_owners),
        "source_hashes": {r["path"]: r["sha256"] for r in rows},
        "arc_archive_sha256": specification["arc_assets"]["arc.zip"]["sha256"],
        "scope": "Real code primary; ARC benchmark diagnostic secondary. Exposed development, not independent blind TEST.",
        "before_any_model_predictions_or_optimizer_steps": True}
    immutable(cache / "MANIFEST.json", json_bytes(manifest))
    return manifest


def future_poison_audit(cache):
    """Scorer-owned fixed-episode source interventions; never reselect cases.

    Code is poisoned in private raw copies at the registered future offsets.
    ARC is poisoned in a private parsed record's official query output; source
    JSON formatting may normalize, but the observed demonstrations/query must
    remain byte-identical under the registered serializer.
    """
    cache = Path(cache)
    actor = json.loads((cache / "DEV_ACTOR.json").read_text())
    judge = json.loads((cache / "DEV_JUDGE.json").read_text())
    originals, poisoned = {}, {}
    for item in actor:
        gold = judge[item["id"]]
        if item["domain"] == "lexical":
            key = "cpython/" + gold["source_path"]
            if key not in originals:
                originals[key] = (cache / "sources" / key).read_bytes()
                if digest(originals[key]) != gold["source_file_sha256"]:
                    raise ValueError("code source changed before future poison")
                poisoned[key] = bytearray(originals[key])
            start, end = gold["target_offset"], gold["target_end"]
            poisoned[key][start:end] = bytes(value ^ 1 for value in poisoned[key][start:end])
    archive_raw = (cache / "sources/arc.zip").read_bytes()
    if digest(archive_raw) != contract()["arc_assets"]["arc.zip"]["sha256"]:
        raise ValueError("ARC archive changed before future poison")
    with zipfile.ZipFile(io.BytesIO(archive_raw)) as archive:
        members = {"data/training/" + Path(name).name: name for name in archive.namelist()
                   if "/data/training/" in name and name.endswith(".json")}
        alternate = []
        for item in actor:
            gold = judge[item["id"]]
            if item["domain"] == "lexical":
                raw = poisoned["cpython/" + gold["source_path"]]
                acquisition = raw[gold["acquisition_start"]:gold["query_start"]]
                query = raw[gold["query_start"]:gold["target_offset"]]
            else:
                key = gold["source_path"]
                original = archive.read(members[key])
                if digest(original) != gold["source_file_sha256"]:
                    raise ValueError("ARC original member changed")
                record = json.loads(original)
                record["test"][0]["output"] = [
                    [(value + 1) % 10 for value in row] for row in record["test"][0]["output"]]
                originals[key], poisoned[key] = original, json_bytes(record)
                acquisition = b"R;" + b"".join(encode_grid(e["input"]) + b">" +
                    encode_grid(e["output"]) + b";" for e in record["train"])
                query = encode_grid(record["test"][0]["input"]) + b">"
            copy = {**item, "acquisition_bytes": list(acquisition), "query_bytes": list(query)}
            if copy != item:
                raise ValueError("future source poison changed registered model observations")
            alternate.append(copy)
    for key, raw in originals.items():
        if key.startswith("cpython/") and (cache / "sources" / key).read_bytes() != raw:
            raise ValueError("future poison modified original file")
    return alternate, {"status": "PASS_FIXED_SOURCE_OBSERVATION_NONINTERFERENCE",
        "episodes": len(actor), "source_originals_unchanged": True, "eligibility_recomputed": False,
        "scope": "Code raw future-byte copies and ARC private parsed future-output intervention; actor observations unchanged",
        "original_hashes": {key: digest(raw) for key, raw in originals.items()},
        "poisoned_hashes": {key: digest(bytes(raw)) for key, raw in poisoned.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--code-bootstrap", type=Path)
    parser.add_argument("--arc-bootstrap", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.cache, code_bootstrap=args.code_bootstrap,
                             arc_bootstrap=args.arc_bootstrap), sort_keys=True))


if __name__ == "__main__":
    main()
