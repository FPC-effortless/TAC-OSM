"""Authenticated original099 data for internal LANGMATH100 continuation.

No network, random split, optimizer or model code. FIT files and held-out DEV
files retain the original published-source cohort. Public DEV is exposed.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

FIT_MANIFEST_SHA = "651ffb5223990cef1d32ddccc41e94797c42a3e5a12ba378afaac1e28afc226e"
EVAL_MANIFEST_SHA = "549a2fa587de07e6d9cc30944255d5d980c9974e7fb9d366d435567098465c31"
SOURCE_MANIFEST_SHA = "f587e98c45b28c8fd9cecdf4de60a72019599839993d92c75dac609e7c1c1d4f"
FIT_ARCHIVE_SHA = "d392f2e45b2ffac2cf9453c509dcb6602629d46053deacddf04ca8044e13f39c"
EVAL_ARCHIVE_SHA = "a933af65d33a97bf0e631ca126c50d805bb0ae9b03b299e9cb6b406f9f1654da"
LICENSES = frozenset(("MIT", "BSD-3-Clause", "Apache-2.0", "PSF-2.0"))
ALLOWED_ACTOR_KEYS = frozenset(("id", "domain", "acquisition_bytes", "query_bytes", "generation_limit"))


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _require(ok, reason):
    if not ok:
        raise ValueError(reason)


def _raw(root, row):
    p = Path(root) / row["file"]
    _require(not p.is_symlink() and p.is_file() and p.resolve().is_relative_to(Path(root).resolve()), "unsafe source path")
    raw = p.read_bytes()
    _require(len(raw) == row["bytes"] and hashlib.sha256(raw).hexdigest() == row["sha256"], "original source bytes differ: " + row["file"])
    _require(row.get("license") in LICENSES, "missing reviewed original source license")
    _require(all(isinstance(row.get(k), str) and row[k] for k in ("owner", "id", "publisher", "commit", "source_path")), "incomplete original publisher identity")
    return raw


def _byte_list(value, name):
    _require(isinstance(value, list) and value and all(type(x) is int and 0 <= x <= 255 for x in value), "legal nonempty byte list required: " + name)


@dataclass(frozen=True)
class FitBundle:
    """Only training-owned bytes; no DEV directory, actor, target or corpus."""
    fit_directory: Path
    fit_sources: tuple[dict, ...]
    fit_corpus: tuple[bytes, ...]
    fit_windows: tuple[tuple[int, int, str], ...]
    fit_episodes: tuple[dict, ...]
    data_identity: dict


def load_fit(fit_dir: str | Path, *, context: int = 128) -> FitBundle:
    """Fit-worker entry point; authentic original FIT only, no DEV access."""
    fit_dir = Path(fit_dir)
    _require(context == 128, "original099 training-window contract is exactly128 bytes")
    _require(sha(fit_dir / "FIT_MANIFEST.json") == FIT_MANIFEST_SHA, "frozen FIT manifest differs")
    fm = _read(fit_dir / "FIT_MANIFEST.json")
    _require(fm["source_manifest_sha256"] == SOURCE_MANIFEST_SHA, "original source cohort changed")
    _require(sha(fit_dir / "FIT_EPISODES.json") == fm["episodes_sha256"], "FIT episodes changed")
    _require(not any(p.name.startswith("DEV") or p.name == "EVAL_MANIFEST.json" for p in fit_dir.rglob("*")), "DEV material delivered into FIT directory")
    sources = tuple(fm["sources"])
    corpus = tuple(_raw(fit_dir, r) for r in sources)
    windows = tuple((i, start, r["id"]) for i, r in enumerate(sources) for start in range(0, len(corpus[i]) - context, context))
    episodes = tuple(_read(fit_dir / "FIT_EPISODES.json"))
    _require(len({e["id"] for e in episodes}) == len(episodes), "duplicate FIT episode ID")
    for e in episodes:
        _require(e.get("partition") == "FIT", "nonFIT episode in training records")
        _require(set(e) == {"id", "domain", "acquisition_bytes", "query_bytes", "generation_limit", "metadata", "partition", "target_bytes"}, "unexpected FIT episode interface")
        for name in ("acquisition_bytes", "query_bytes", "target_bytes"): _byte_list(e[name], name)
        _require(e["domain"] in ("lexical", "arc"), "unknown FIT domain")
        _require(len(e["acquisition_bytes"] + e["query_bytes"] + e["target_bytes"]) - 1 <= 272, "FIT episode exceeds original dense geometry")
    # Verify all real lexical observations and labels against exact FIT files.
    by_hash = {r["sha256"]: raw for r, raw in zip(sources, corpus)}
    for e in episodes:
        if e["domain"] != "lexical": continue
        meta = e["metadata"]; raw = by_hash[meta["source_file_sha256"]]
        _require(raw[meta["acquisition_start"]:meta["query_start"]] == bytes(e["acquisition_bytes"]), "FIT acquisition source join failed")
        _require(raw[meta["query_start"]:meta["target_offset"]] == bytes(e["query_bytes"]), "FIT query source join failed")
        _require(raw[meta["target_offset"]:meta["target_end"]] == bytes(e["target_bytes"]), "FIT answer source join failed")
    identity = {"FIT_manifest_sha256": FIT_MANIFEST_SHA, "FIT_archive_sha256": FIT_ARCHIVE_SHA,
                "source_manifest_sha256": SOURCE_MANIFEST_SHA, "fit_examples_sha256": digest([list(w) for w in windows]),
                "fit_episode_ids_sha256": digest([e["id"] for e in episodes]), "context_bytes": context,
                "qualification": "original_authenticated_public_FIT_only", "DEV_delivered": False,
                "protected_TEST_access": False, "synthetic_primary_training": False}
    return FitBundle(fit_dir, sources, corpus, windows, episodes, identity)


@dataclass(frozen=True)
class DataBundle:
    fit_directory: Path
    eval_directory: Path
    fit_sources: tuple[dict, ...]
    fit_corpus: tuple[bytes, ...]
    fit_windows: tuple[tuple[int, int, str], ...]
    fit_episodes: tuple[dict, ...]
    dev_sources: tuple[dict, ...]
    dev_corpus: dict[str, bytes]
    dev_windows: tuple[dict, ...]
    dev_actors: tuple[dict, ...]
    dev_donors: tuple[dict, ...]
    dev_binding_controls: tuple[dict, ...]
    data_identity: dict


def load_data(fit_dir: str | Path, eval_dir: str | Path, *, context: int = 128) -> DataBundle:
    """Verify exact frozen inputs; DEV target values are not returned to trainers."""
    fit_dir, eval_dir = Path(fit_dir), Path(eval_dir)
    _require(context == 128, "original099 training-window contract is exactly128 bytes")
    _require(sha(fit_dir / "FIT_MANIFEST.json") == FIT_MANIFEST_SHA, "frozen FIT manifest differs")
    _require(sha(eval_dir / "EVAL_MANIFEST.json") == EVAL_MANIFEST_SHA, "frozen DEV manifest differs")
    fm, em = _read(fit_dir / "FIT_MANIFEST.json"), _read(eval_dir / "EVAL_MANIFEST.json")
    _require(fm["source_manifest_sha256"] == em["source_manifest_sha256"] == SOURCE_MANIFEST_SHA, "original source cohort changed")
    _require(sha(fit_dir / "FIT_EPISODES.json") == fm["episodes_sha256"], "FIT episodes changed")
    _require(not any(p.name.startswith("DEV") or p.name == "EVAL_MANIFEST.json" for p in fit_dir.rglob("*")), "DEV material delivered into FIT directory")
    for name, expected in em["files"].items():
        p = eval_dir / name
        _require(not p.is_symlink() and p.resolve().is_relative_to(eval_dir.resolve()) and sha(p) == expected, "DEV source/record changed: " + name)
    fs, ds = tuple(fm["sources"]), tuple(em["language_sources"])
    fc = tuple(_raw(fit_dir, r) for r in fs)
    dc = {r["file"]: _raw(eval_dir, r) for r in ds}
    _require(not ({r["owner"] for r in fs} & {r["owner"] for r in ds}), "whole-file FIT/DEV owner overlap")
    _require(not ({r["sha256"] for r in fs} & {r["sha256"] for r in ds}), "whole-file FIT/DEV byte overlap")
    windows = tuple((i, start, r["id"]) for i, r in enumerate(fs) for start in range(0, len(fc[i]) - context, context))
    episodes = tuple(_read(fit_dir / "FIT_EPISODES.json"))
    actors = tuple(_read(eval_dir / "DEV_ACTOR.json"))
    for e in episodes:
        _require(e.get("partition") == "FIT", "nonFIT episode in training records")
        _require(set(e) == {"id", "domain", "acquisition_bytes", "query_bytes", "generation_limit", "metadata", "partition", "target_bytes"}, "unexpected FIT episode interface")
        for name in ("acquisition_bytes", "query_bytes", "target_bytes"):
            _byte_list(e[name], name)
        _require(e["domain"] in ("lexical", "arc"), "unknown FIT domain")
        _require(len(e["acquisition_bytes"] + e["query_bytes"] + e["target_bytes"]) - 1 <= 272, "FIT episode exceeds original dense geometry")
    for e in actors:
        _require(set(e) == ALLOWED_ACTOR_KEYS, "DEV actor contains target/metadata/forbidden fields")
        for name in ("acquisition_bytes", "query_bytes"):
            _byte_list(e[name], name)
        _require(e["generation_limit"] == {"lexical": 4, "arc": 40}.get(e["domain"]), "target-independent generation budget changed")
    _require(len({e["id"] for e in episodes}) == len(episodes) and len({e["id"] for e in actors}) == len(actors), "duplicate episode ID")
    _require(not ({e["id"] for e in episodes} & {e["id"] for e in actors}), "FIT/DEV episode ID overlap")
    dev_windows = tuple(_read(eval_dir / "LANGUAGE_CASES.json"))
    for w in dev_windows:
        _require(w["file"] in dc and w["bytes"] == 257 and len(dc[w["file"]][w["offset"]:w["offset"] + 257]) == 257, "malformed complete held-out window")
    identity = {"fit_manifest_sha256": FIT_MANIFEST_SHA, "eval_manifest_sha256": EVAL_MANIFEST_SHA,
                "source_manifest_sha256": SOURCE_MANIFEST_SHA, "fit_examples_sha256": digest([list(w) for w in windows]),
                "fit_episode_ids_sha256": digest([e["id"] for e in episodes]), "DEV_language_ids_sha256": digest([w["id"] for w in dev_windows]),
                "DEV_episode_ids_sha256": digest([e["id"] for e in actors]), "context_bytes": context,
                "qualification": "original_authenticated_public_FIT_DEV_previously_exposed_not_blind",
                "protected_TEST_access": False, "synthetic_primary_training": False}
    return DataBundle(fit_dir, eval_dir, fs, fc, windows, episodes, ds, dc, dev_windows, actors,
                      tuple(_read(eval_dir / "DEV_DONORS.json")), tuple(_read(eval_dir / "DEV_CONTROL_ACTOR.json")), identity)


def schedule(bundle: DataBundle, seed: int, steps: int, *, batch_size: int = 4, mode: str = "mixed") -> list[dict]:
    """Original deterministic sample order; same returned plan for every arm."""
    _require(type(seed) is int and seed >= 0 and type(steps) is int and steps > 0, "legitimate deterministic schedule parameters required")
    _require(batch_size == 4, "original mixed recipe batch size is four")
    _require(mode in ("mixed", "language"), "mixed or language-only schedule required")
    rng = np.random.default_rng(seed)
    lex = [i for i, e in enumerate(bundle.fit_episodes) if e["domain"] == "lexical"]
    arc = [i for i, e in enumerate(bundle.fit_episodes) if e["domain"] == "arc"]
    plan = []
    for step in range(steps):
        if mode == "mixed" and step % 4 == 3:
            indices = [int(i) for i in np.concatenate((rng.choice(lex, 2, replace=False), rng.choice(arc, 2, replace=False)))]
            plan.append({"mode": "episodic", "indices": indices})
        else:
            plan.append({"mode": "language", "indices": [int(i) for i in rng.integers(0, len(bundle.fit_windows), batch_size)]})
    return plan


def language_batch(bundle: DataBundle, indices: list[int]):
    """Long[B,128] observations and next-byte targets; never concatenate owners."""
    import torch
    raw = [bundle.fit_corpus[bundle.fit_windows[i][0]][bundle.fit_windows[i][1]:bundle.fit_windows[i][1] + 129] for i in indices]
    values = torch.tensor([list(r) for r in raw], dtype=torch.long)
    _require(values.shape == (len(indices), 129), "fixed complete original windows required")
    return values[:, :-1], values[:, 1:]


def episode_batch(bundle: DataBundle, indices: list[int]):
    """Original272-slot x/y/valid/future contract plus acquisition boundaries.

    Future answers are targets; teacher forcing only makes an earlier answer
    byte available at the later next-byte step. The last target never enters x.
    A model's own causal graph must still be verified separately.
    """
    import torch
    rows = [bundle.fit_episodes[i] for i in indices]
    x = torch.zeros(len(rows), 272, dtype=torch.long); y = torch.zeros_like(x)
    valid = torch.zeros_like(x, dtype=torch.bool); future = torch.zeros_like(valid)
    domains, acquisition_lengths = [], []
    for i, e in enumerate(rows):
        a, q, t = e["acquisition_bytes"], e["query_bytes"], e["target_bytes"]
        raw = a + q + t; n = len(raw) - 1
        x[i, :n] = torch.tensor(raw[:-1]); y[i, :n] = torch.tensor(raw[1:]); valid[i, :n] = True
        future[i, len(a) + len(q) - 1:n] = True
        acquisition_lengths.append(len(a)); domains.append("code" if e["domain"] == "lexical" else "arc")
    return {"tokens": x, "targets": y, "valid": valid, "future": future, "domains": tuple(domains),
            "acquisition_lengths": tuple(acquisition_lengths), "query_lengths": tuple(len(e["query_bytes"]) for e in rows),
            "counts": {"observed": int(valid.sum()), "supervised": int(valid.sum()), "future_targets": int(future.sum()), "padding": int((~valid).sum())}}


def batch(bundle: DataBundle, item: dict):
    if item["mode"] == "language":
        x, y = language_batch(bundle, item["indices"])
        return {"tokens": x, "targets": y, "valid": None, "future": None, "domains": None, "acquisition_lengths": None,
                "counts": {"observed": x.numel(), "supervised": y.numel(), "future_targets": 0, "padding": 0}}
    _require(item["mode"] == "episodic", "unknown training sample mode")
    return episode_batch(bundle, item["indices"])


def exposure_summary(bundle, plan: list[dict]) -> dict:
    """Exact planned real-byte exposure; dense padding is counted separately."""
    counts = Counter(steps=0, examples=0, language_steps=0, episodic_steps=0,
                     observed=0, supervised=0, future_targets=0, padding=0, dense_slots=0)
    for item in plan:
        n = len(item["indices"]); counts["steps"] += 1; counts["examples"] += n
        if item["mode"] == "language":
            counts["language_steps"] += 1
            counts["observed"] += n * 128; counts["supervised"] += n * 128; counts["dense_slots"] += n * 128
        elif item["mode"] == "episodic":
            counts["episodic_steps"] += 1; counts["dense_slots"] += n * 272
            for i in item["indices"]:
                e = bundle.fit_episodes[i]
                real = len(e["acquisition_bytes"]) + len(e["query_bytes"]) + len(e["target_bytes"]) - 1
                counts["observed"] += real; counts["supervised"] += real
                counts["future_targets"] += len(e["target_bytes"]); counts["padding"] += 272 - real
        else:
            raise ValueError("unknown schedule item mode")
    _require(counts["supervised"] + counts["padding"] == counts["dense_slots"], "dense exposure accounting differs")
    return dict(counts)


def observed_episode_input(actor: dict, *, regime: str):
    """Observed-only native model inputs, without target or evaluator access.

    hidden_history means acquire support, then only query arrives next; the
    runner carries or resets state. direct_history replays the exact same
    acquisition+query bytes from zero. query_only receives only query bytes.
    These vary information access deliberately and are not automatically a
    parameter/data/information-matched architecture superiority comparison.
    """
    _require(set(actor) == ALLOWED_ACTOR_KEYS, "only legal original actor fields accepted")
    _require(regime in ("hidden_history", "direct_history", "query_only"), "unknown observed-information regime")
    a, q = list(actor["acquisition_bytes"]), list(actor["query_bytes"])
    return {"acquisition_bytes": a if regime == "hidden_history" else [],
            "query_bytes": a + q if regime == "direct_history" else q,
            "generation_limit": actor["generation_limit"]}


def dev_language_batch(bundle: DataBundle, indices: list[int]):
    import torch
    rows = [bundle.dev_windows[i] for i in indices]
    values = torch.tensor([list(bundle.dev_corpus[r["file"]][r["offset"]:r["offset"] + 257]) for r in rows], dtype=torch.long)
    return values[:, :-1], values[:, 1:], tuple(r["id"] for r in rows)


def load_judge(bundle: DataBundle):
    """Scorer-only interface. Training/data batch functions never call this."""
    return _read(bundle.eval_directory / "DEV_JUDGE.json")


def data_audit(bundle: DataBundle) -> dict:
    """Source/target join and finite-cohort audit, not independent certification."""
    judge = load_judge(bundle)
    fit_raw = {r["sha256"]: raw for r, raw in zip(bundle.fit_sources, bundle.fit_corpus)}
    dev_raw = {r["sha256"]: bundle.dev_corpus[r["file"]] for r in bundle.dev_sources}
    joins = {"FIT": 0, "DEV": 0}
    for partition, rows, sources in (("FIT", bundle.fit_episodes, fit_raw), ("DEV", bundle.dev_actors, dev_raw)):
        for row in rows:
            if row["domain"] != "lexical": continue
            meta = row["metadata"] if partition == "FIT" else judge[row["id"]]
            target = row["target_bytes"] if partition == "FIT" else meta["target_bytes"]
            raw = sources[meta["source_file_sha256"]]
            _require(raw[meta["acquisition_start"]:meta["query_start"]] == bytes(row["acquisition_bytes"]), "lexical original acquisition source join failed")
            _require(raw[meta["query_start"]:meta["target_offset"]] == bytes(row["query_bytes"]), "lexical query source join failed")
            _require(raw[meta["target_offset"]:meta["target_end"]] == bytes(target), "lexical target source join failed")
            _require(meta["acquisition_start"] < meta["query_start"] < meta["target_offset"] < meta["target_end"], "causal lexical source offsets failed")
            joins[partition] += 1
    fit_tasks = {e["metadata"]["cluster_id"] for e in bundle.fit_episodes if e["domain"] == "arc"}
    dev_tasks = {judge[e["id"]]["cluster_id"] for e in bundle.dev_actors if e["domain"] == "arc"}
    _require(not fit_tasks & dev_tasks, "authored ARC FIT/DEV task overlap")
    query_audit = {}
    for domain in ("lexical", "arc"):
        rows = [e for e in bundle.dev_actors if e["domain"] == domain and (domain != "lexical" or judge[e["id"]]["group"] == "positive")]
        groups = defaultdict(list)
        for e in rows: groups[tuple(e["query_bytes"])].append(e)
        conflicting = [g for g in groups.values() if len({tuple(judge[e["id"]]["target_bytes"]) for e in g}) > 1]
        query_audit[domain] = {"cases": len(rows), "distinct_queries": len(groups), "conflicting_identical_query_groups": len(conflicting),
                               "target_occurs_in_acquisition": sum(bytes(judge[e["id"]]["target_bytes"]) in bytes(e["acquisition_bytes"]) for e in rows),
                               "target_occurs_in_query": sum(bytes(judge[e["id"]]["target_bytes"]) in bytes(e["query_bytes"]) for e in rows)}
    return {"status": "PASS_ORIGINAL_BYTE_AND_COHORT_CHECKS", "identity": bundle.data_identity,
            "FIT_files": len(bundle.fit_sources), "FIT_bytes": sum(map(len, bundle.fit_corpus)), "FIT_128_target_windows": len(bundle.fit_windows),
            "FIT_publishers": dict(Counter(r["publisher"] for r in bundle.fit_sources)), "FIT_licenses": dict(Counter(r["license"] for r in bundle.fit_sources)),
            "FIT_episodes": dict(Counter(e["domain"] for e in bundle.fit_episodes)), "DEV_files": len(bundle.dev_sources),
            "DEV_language_windows": len(bundle.dev_windows), "DEV_language_targets": len(bundle.dev_windows) * 256,
            "DEV_language_modalities": dict(Counter(e["modality"] for e in bundle.dev_windows)), "DEV_episodes": dict(Counter(e["domain"] for e in bundle.dev_actors)),
            "verified_real_lexical_joins": joins, "ARC_FIT_tasks": len(fit_tasks), "ARC_DEV_tasks": len(dev_tasks), "query_audit": query_audit,
            "not_established": ["semantic deduplication", "novel vocabulary labels under broader FIT", "information-theoretic hidden-history necessity", "independent assessor ACL", "new held-out rule families or multi-step composition"],
            "classification": "Primary language/lexical bytes are authentic original source. Authored ARC is a secondary benchmark. DEV was previously exposed. No new synthetic primary training examples or TEST data."}


def manifest_proposal(bundle: DataBundle, seeds=(1701, 1702, 1703), steps=256, mode="mixed") -> dict:
    """Data/schedule facts only; root must add actual live model/compute gates."""
    plans = [{"seed": s, "steps": steps, "schedule_sha256": digest(schedule(bundle, s, steps, mode=mode)),
              "batch_size": 4, "language_context": 128, "episodic_slots": 272, "mode": mode} for s in seeds]
    for row in plans:
        row["exposure"] = exposure_summary(bundle, schedule(bundle, row["seed"], steps, mode=mode))
    return {"schema": "langmath101.original_data_proposal.v1", "registered_before_new_fits": True,
            "execution_level": "internal_research", "data_identity": bundle.data_identity, "plans": plans,
            "source_artifacts": {"run_id": 41153068368, "FIT_artifact_id": 13023429505, "FIT_archive_sha256": FIT_ARCHIVE_SHA,
                                 "DEV_artifact_id": 13024430156, "DEV_archive_sha256": EVAL_ARCHIVE_SHA},
            "parameter_and_compute_preflight": "NOT_PROVIDED_BY_DATA_MODULE; exact live pathway counts, actual operator/compute accounting and final training receipts remain compulsory",
            "regimes": {"present_information": "direct_history observed acquisition+query replay from zero", "hidden_history": "acquisition then query; support available only through state or explicitly separately costed external history", "query_only": "query only, different information access; not a full-info Transformer surrogate"},
            "causal_control": "Source-only input allowlist; no evaluator fields supplied to models; target poisoning on the last future byte must not alter current observed input. Verify actual model causality separately.",
            "independence": "Public exposed DEV only; no protected evaluation or independent replication authorization."}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(); p.add_argument("--fit", type=Path, required=True); p.add_argument("--eval", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); a = p.parse_args()
    if a.output.exists(): raise FileExistsError("retain original data audit")
    bundle = load_data(a.fit, a.eval)
    a.output.mkdir(parents=True)
    (a.output / "DATA_AUDIT.json").write_text(json.dumps(data_audit(bundle), indent=2) + "\n")
    (a.output / "DATA_MANIFEST_PROPOSAL.json").write_text(json.dumps(manifest_proposal(bundle), indent=2) + "\n")
    print(json.dumps({"output": str(a.output), "FIT_windows": len(bundle.fit_windows), "DEV_windows": len(bundle.dev_windows)}))
