"""Small, vendor-neutral research run ledger for TAC-OSM.

The canonical object is a locally written, immutable JSON receipt referencing
existing source files by hash. It NEVER infers genuine learning/success from a
JSON metric, uploads proprietary data, or changes training/evaluation gates.
No dependencies beyond Python's standard library; all variants are names, not
architecture-specific imports.
"""
from __future__ import annotations

import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
from typing import Any

SCHEMA = "tac-osm.lab-run.v1"
_ALLOWED_STAGE = {"fixture", "source_audit", "training", "evaluation", "analysis", "literature"}
_ALLOWED_EVIDENCE = {"constructed_fixture", "historical_unverified", "source_only", "runner_report"}
_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,99}$")
_SECRET = re.compile(r"(?i)(access[_-]?token|auth[_-]?token|bearer|secret|password|api[_-]?key|credential|private[_-]?key|authorization)")


class LabError(ValueError):
    pass


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2,
                       allow_nan=False, ensure_ascii=True) + "\n").encode()


def _name(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise LabError(f"{label} must be a simple identifier (letters, digits, . _ -)")
    if _SECRET.search(value):
        raise LabError(f"{label} appears to contain credential metadata")
    return value


def _digest(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise LabError("source/checkpoint/dataset must be an existing regular file, not a symlink")
    hash_ = hashlib.sha256()
    count = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            count += len(chunk)
            hash_.update(chunk)
    return {"sha256": hash_.hexdigest(), "bytes": count,
            "basename": path.name}


def _git_revision(root: Path) -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "--verify", "HEAD"],
                                cwd=root, check=True, capture_output=True,
                                text=True, timeout=5)
        value = result.stdout.strip()
        return value if re.fullmatch(r"[0-9a-f]{40}", value) else None
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def _scalars(values: dict[str, Any] | None, label: str) -> dict[str, int | float | bool | str]:
    if not isinstance(values, dict):
        raise LabError(label + " requires a plain dictionary")
    if len(values) > 64:
        raise LabError(label + " has too many fields")
    data = {}
    for name, value in values.items():
        key = _name(name, label + " key")
        if type(value) not in (int, float, bool, str):
            raise LabError(label + " values must be scalar; no private rows or tensors")
        if isinstance(value, float) and not math.isfinite(value):
            raise LabError(label + " contains nonfinite value")
        if isinstance(value, str) and (len(value) > 160 or _SECRET.search(value)):
            raise LabError(label + " contains inappropriate long or sensitive value")
        if label == "metrics" and type(value) not in (int, float):
            raise LabError("metric values must be finite numeric scalars")
        data[key] = value
    return dict(sorted(data.items()))


def create_receipt(*, run_id: str, experiment: str, variant: str, stage: str,
                   evidence: str, source: Path, repo_root: Path,
                   metrics: dict | None = None, parameters: dict | None = None,
                   seed: int | None = None, dataset: Path | None = None,
                   checkpoint: Path | None = None,
                   external_run_url: str | None = None) -> dict[str, Any]:
    """A receipt records pointers and checksums, NOT scientific qualification.

    source is an existing diagnostic/report file; never parse arbitrary raw
    input observations, future labels, secrets or entire trained states.
    """
    for label, val in (("run_id", run_id), ("experiment", experiment)):
        _name(val, label)
    if not isinstance(variant, str) or not variant.strip() or len(variant) > 100:
        raise LabError("variant must be an explicit architecture/version label")
    if _SECRET.search(variant):
        raise LabError("variant label contains sensitive metadata")
    if stage not in _ALLOWED_STAGE or evidence not in _ALLOWED_EVIDENCE:
        raise LabError("stage or evidence category unsupported")
    if type(seed) is not int and seed is not None:
        raise LabError("seed must be an integer or unspecified")
    if external_run_url is not None and (not isinstance(external_run_url, str) or
                                        not external_run_url.startswith("https://") or
                                        len(external_run_url) > 300):
        raise LabError("optional external run link must be HTTPS")
    report = _digest(Path(source))
    # Hashes are content identities; a basename is only display information,
    # NEVER proof of trusted original source or fair comparator conditions.
    receipt = {
        "schema": SCHEMA, "run_id": run_id, "experiment": experiment,
        "variant": variant, "stage": stage, "evidence_kind": evidence,
        "seed": seed, "git_commit": _git_revision(Path(repo_root)),
        "source_report": report,
        "dataset_file": _digest(Path(dataset)) if dataset else None,
        "checkpoint_file": _digest(Path(checkpoint)) if checkpoint else None,
        "parameters": _scalars(parameters or {}, "parameters"),
        "metrics": _scalars(metrics or {}, "metrics"),
        "external_run_url": external_run_url,
        "scientific_qualification": "NOT_EVALUATED_BY_LAB",
        "capability_claim_authorized": False,
        "comparison_parity_verified": False,
        "reward_integrity_verified": False,
        "note": "Run index only; apply real-world, parity, and reward gates separately.",
    }
    return receipt


def write_receipt(receipt: dict, output_dir: Path) -> Path:
    if receipt.get("schema") != SCHEMA:
        raise LabError("cannot persist unrecognized run receipt")
    run_id = _name(receipt.get("run_id"), "run_id")
    destination = Path(output_dir) / run_id / "receipt.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    blob = _json_bytes(receipt)
    if destination.exists():
        if destination.read_bytes() != blob:
            raise LabError("run ID already recorded with different evidence; history is immutable")
        return destination
    # Exclusive create: concurrency must not rewrite an existing run.
    try:
        with destination.open("xb") as f:
            f.write(blob)
    except FileExistsError:
        if destination.read_bytes() != blob:
            raise LabError("concurrent conflicting run ID")
    return destination


def read_receipts(root: Path) -> list[dict[str, Any]]:
    root = Path(root)
    if not root.exists():
        return []
    results = []
    for path in sorted(root.glob("*/receipt.json")):
        if path.is_symlink() or path.parent.is_symlink():
            raise LabError("symlink in research ledger")
        r = json.loads(path.read_text())
        if not isinstance(r, dict) or r.get("schema") != SCHEMA:
            raise LabError("invalid research ledger schema: " + str(path))
        if _name(r.get("run_id"), "run_id") != path.parent.name:
            raise LabError("run path and receipt mismatch")
        if r.get("capability_claim_authorized") is not False:
            raise LabError("untrusted ledger tried to assert capability qualification")
        if r.get("scientific_qualification") != "NOT_EVALUATED_BY_LAB":
            raise LabError("untrusted ledger tried to assert scientific qualification")
        results.append(r)
    return results


def summary(root: Path) -> dict[str, Any]:
    rows = read_receipts(root)
    variants = sorted({x["variant"] for x in rows})
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["experiment"], []).append(row)
    return {"schema": "tac-osm.lab-summary.v1",
            "total_runs": len(rows), "experiments": len(groups),
            "variants": variants,
            "evidence_categories": {
                name: sum(x["evidence_kind"] == name for x in rows)
                for name in sorted(_ALLOWED_EVIDENCE)},
            "scientifically_qualified_by_ledger": 0,
            "runs": rows}


def render_dashboard(root: Path, output: Path) -> Path:
    """Single, offline HTML file. No external scripts, trackers or analytics."""
    report = summary(root)
    rows = report["runs"]
    esc = lambda x: html.escape(str(x), quote=True)
    headers = ("Run ID", "Experiment", "Variant", "Stage", "Evidence", "Metrics")
    items = []
    for item in rows:
        values = [item["run_id"], item["experiment"], item["variant"],
                  item["stage"], item["evidence_kind"],
                  ", ".join(f"{k}: {v}" for k,v in item["metrics"].items()) or "—"]
        items.append("<tr>"+"".join(f"<td>{esc(x)}</td>" for x in values)+"</tr>")
    # Metrics trend by run index, never a comparison-superiority graph.
    numeric: dict[str, list[float]] = {}
    for entry in rows:
        for key, value in entry["metrics"].items():
            numeric.setdefault(key, []).append(float(value))
    sparklines = []
    for metric, series in sorted(numeric.items())[:12]:
        if len(series) < 2:
            continue
        lo, hi = min(series), max(series)
        ys = [45 - (v - lo) / (hi - lo) * 35 if hi > lo else 27
              for v in series]
        xs = [10 + i * 200 / (len(series) - 1) for i in range(len(series))]
        points = " ".join(f"{x:.1f},{y:.1f}" for x,y in zip(xs, ys))
        sparklines.append(
            f"<section><strong>{esc(metric)}</strong><svg width='220' height='54' "
            f"viewBox='0 0 220 54' role='img' aria-label='{esc(metric)} by recorded run index'>"
            f"<polyline fill='none' stroke='currentColor' stroke-width='2' "
            f"points='{points}' /></svg><small>{len(series)} values; not comparable by default</small></section>")
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src data:">
<title>TAC-OSM Research Lab</title>
<style>
body{{font:15px system-ui,sans-serif;margin:2rem auto;max-width:1120px;padding:0 1rem;color:#243143;background:#f8fafc}}
h1{{font-size:1.65rem}}table{{border-collapse:collapse;width:100%;background:white}}
td,th{{padding:.65rem;text-align:left;border-bottom:1px solid #dbe3ec;vertical-align:top}}
th{{background:#eff4fa}}table{{display:block;overflow-x:auto}}
aside{{background:#fff7e7;padding:1rem;border-left:4px solid #967026;margin:1rem 0}}
section{{background:white;padding:1rem;display:inline-flex;flex-direction:column;margin:.3rem;border:1px solid #ddd;border-radius:10px}}
small{{color:#536477}}.meta{{color:#536477}}
</style></head><body>
<h1>TAC-OSM Research Lab</h1><p class="meta">{report['total_runs']} receipts ·
{report['experiments']} experiments · {len(report['variants'])} variants</p>
<aside><strong>Engineering-only evidence index.</strong> All displayed metrics are
unqualified: no real-world capability, architecture superiority, safety,
reward integrity or scientific parity is inferred from a logged number.</aside>
<h2>Run evidence</h2><table><thead><tr>{''.join(f'<th>{esc(x)}</th>' for x in headers)}
</tr></thead><tbody>{''.join(items)}</tbody></table>
<h2>Recorded metric trends</h2>{''.join(sparklines) if sparklines else '<p>At least two values per metric are needed to draw trends.</p>'}
<p><small>Generated offline. No accounts, tracking scripts or network requests.</small></p>
</body></html>"""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    return output
