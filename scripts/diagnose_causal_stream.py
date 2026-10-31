#!/usr/bin/env python3
"""SENSE-TOK-002 engineering-only replay. No optimizer and no neural fits.

Locally installed CPython 3.13.5 files are NOT assumed publisher authentic.
Compare original local Git blob SHAs against registered publisher SHAs.
Tokenize original FULL files for lexer proxy, but evaluate the same bounded
prefix bytes for all tokenizer arms. Never train or unseal a TEST set.
"""
from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
import sysconfig
import tokenize as py_lexer
from hashlib import sha1, sha256
from pathlib import Path
from time import perf_counter_ns

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tac_osm.causal_patch_stream import STREAM_VARIANTS, CausalPatchStream

OFFICIAL_BLOBS = {
    "argparse.py": "bd088ea0e66a0484f8e9390d421e769956eb3519",
    "ast.py": "37b20206b8af4fda67c631fd010eda0008985511",
    "inspect.py": "de2fe7e1ae0f6d345fdf1a53f4f0073079f50802",
    "csv.py": "cd202659873811a2ea47c7374cc268d26a9ac5f3",
    "difflib.py": "33e7e6c165aa0c12c779331927c3a40f3b09d9e2",
    "textwrap.py": "7ca393d1c371aad54a2d1d70fae15efa889175bb",
    "email/parser.py": "06d99b17f2f9c4992301c432f498b0a32339171b",
    "logging/__init__.py": "357d127c09048226780c7ac60db1e1ab792efd0f",
}
SOURCE_COMMIT = "6cb20a219a860eaf687b2d968b41c480c7461909"
SOURCE_TAG = "v3.13.5"


def git_blob_sha1(data: bytes) -> str:
    return sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def ref_boundaries(source: bytes) -> set[int]:
    lines = source.decode("utf-8").splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line.encode("utf-8")))
    positions = set()
    for token in py_lexer.tokenize(io.BytesIO(source).readline):
        if token.type in (py_lexer.ENCODING, py_lexer.ENDMARKER):
            continue
        for row, col in (token.start, token.end):
            if 1 <= row <= len(lines):
                pos = offsets[row-1] + len(lines[row-1][:col].encode("utf-8"))
                if 0 < pos < len(source):
                    positions.add(pos)
    return positions


def f1(ends: set[int], ref: set[int]) -> float:
    tp = len(ends & ref)
    p = tp/len(ends) if ends else 0
    r = tp/len(ref) if ref else 0
    return 2*p*r/(p+r) if p+r else 0


def evaluate(max_bytes: int = 8192, repeats: int = 3) -> dict:
    root = Path(sysconfig.get_path("stdlib"))
    rows = []
    for name, upstream_sha in OFFICIAL_BLOBS.items():
        path = root/name
        source = path.read_bytes()
        local_sha = git_blob_sha1(source)
        length = min(len(source), max_bytes)
        if length < len(source):
            newline = source.rfind(b"\n", 0, length)
            if newline < 0:
                raise ValueError("no complete source line in bound")
            length = newline+1
        raw = source[:length]
        ref = {offset for offset in ref_boundaries(source) if offset < length}
        arms = []
        for variant in STREAM_VARIANTS:
            def run():
                stream = CausalPatchStream(variant=variant)
                stream.observe(raw)
                stream.finish()
                view = stream.snapshot()
                if view.complete_byte_payload != raw:
                    raise ValueError("not lossless")
                return stream, view
            run()
            durations = []
            for _ in range(repeats):
                start = perf_counter_ns()
                stream, view = run()
                durations.append(perf_counter_ns()-start)
            ends = {e.patch.end for e in view.committed if e.patch.end != len(raw)}
            arms.append({
                "variant": variant, "patches": len(view.committed),
                "delayed_commits": stream.delayed_commits,
                "lexer_boundary_f1_not_capability": f1(ends, ref),
                "median_python_wall_ms_uncontrolled": statistics.median(durations)/1e6,
                "source_exact_roundtrip": True,
            })
        rows.append({
            "source_name": name, "bytes": len(raw),
            "original_sha256": sha256(source).hexdigest(),
            "excerpt_sha256": sha256(raw).hexdigest(),
            "local_git_blob_sha1": local_sha,
            "publisher_git_blob_sha1": upstream_sha,
            "matches_publisher": local_sha == upstream_sha,
            "arms": arms,
        })
    def summary(eligible):
        return {v: {
            "files": len(eligible), "bytes": sum(row["bytes"] for row in eligible),
            "patches": sum(next(a["patches"] for a in row["arms"] if a["variant"] == v) for row in eligible),
            "mean_unweighted_lexer_f1_not_capability": (
                statistics.mean(next(a["lexer_boundary_f1_not_capability"] for a in row["arms"]
                                     if a["variant"] == v) for row in eligible) if eligible else None),
        } for v in STREAM_VARIANTS}
    result = {
        "study": "SENSE-TOK-002", "status": "NO_FIT_ENGINEERING_ONLY",
        "publisher": "python/cpython", "tag": SOURCE_TAG, "commit": SOURCE_COMMIT,
        "publisher_tag_signature": "GITHUB_VERIFIED_SEPARATELY_NOT_BY_THIS_SCRIPT",
        "publisher_source": "https://github.com/python/cpython/tree/" + SOURCE_COMMIT,
        "models_fitted": 0, "optimizer_steps": 0,
        "learned_model_capability": "NOT_TESTED", "independent_evaluator": "MISSING",
        "all_sources": summary(rows),
        "exact_publisher_matches_only": summary([row for row in rows if row["matches_publisher"]]),
        "sources": rows,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-bytes", type=int, default=8192)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not (128 <= args.max_bytes <= 100000) or not (1 <= args.repeats <= 10):
        parser.error("invalid bounds")
    output = json.dumps(evaluate(args.max_bytes, args.repeats), sort_keys=True, indent=2)+"\n"
    if args.output:
        args.output.write_text(output, encoding="utf8")
    else:
        print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
