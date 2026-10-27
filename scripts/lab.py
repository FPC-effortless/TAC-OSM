#!/usr/bin/env python3
"""TAC-OSM lightweight lab: use after existing scripts, no changed training.

Examples:
  python scripts/lab.py record --run-id ci-123 --experiment S1OBS --variant "System 1" \
    --stage fixture --evidence constructed_fixture --source ci-artifacts/observatory-fixture.json
  python scripts/lab.py report --output runs/lab/dashboard.html
  python scripts/lab.py status

This tool only indexes independent existing files. It NEVER launches training
or classifies experiment success/real-world capability.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tac_osm.research_lab import (  # noqa: E402
    LabError, create_receipt, write_receipt, summary, render_dashboard,
)

DEFAULT_RUNS = ROOT / "runs" / "lab"


def _key_value(items, label):
    values = {}
    for item in items:
        if "=" not in item:
            raise LabError(label + ": specify NAME=VALUE")
        name, value = item.split("=", 1)
        if name in values:
            raise LabError(label + ": duplicate name")
        if label == "metric":
            try:
                number = float(value)
            except ValueError as exc:
                raise LabError(label + ": requires numeric values") from exc
            values[name] = number
        elif label == "param":
            # Store short, human-readable scalar declarations only, NEVER full
            # structured input rows, secrets, checkpoint bytes, private labels.
            values[name] = value
    return values


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    rec = sub.add_parser("record", help="Record hashes and scalar metrics of an EXISTING report")
    rec.add_argument("--run-id", required=True)
    rec.add_argument("--experiment", required=True)
    rec.add_argument("--variant", required=True)
    rec.add_argument("--stage", required=True,
                     choices=("fixture", "source_audit", "training",
                              "evaluation", "analysis", "literature"))
    rec.add_argument("--evidence", required=True,
                     choices=("constructed_fixture", "historical_unverified",
                              "source_only", "runner_report"))
    rec.add_argument("--source", type=Path, required=True)
    rec.add_argument("--dataset", type=Path)
    rec.add_argument("--checkpoint", type=Path)
    rec.add_argument("--seed", type=int)
    rec.add_argument("--metric", action="append", default=[])
    rec.add_argument("--param", action="append", default=[])
    rec.add_argument("--external-run-url")
    rec.add_argument("--out", type=Path, default=DEFAULT_RUNS)
    status = sub.add_parser("status", help="Print existing run inventory")
    status.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    report = sub.add_parser("report", help="Build an offline HTML evidence overview")
    report.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    report.add_argument("--output", type=Path,
                        default=DEFAULT_RUNS / "dashboard.html")
    plans = sub.add_parser("plan", help="Read-only recipe for optional MLOps, cluster, dataset or collaboration providers")
    plans.add_argument("--provider", required=True)
    plans.add_argument("--receipt", type=Path)
    plans.add_argument("--minutes", type=int, default=30)
    plans.add_argument("--cpus", type=int, default=1)
    plans.add_argument("--gpus", type=int, default=0)
    plans.add_argument("--output", type=Path)
    options = p.parse_args(argv)
    try:
        if options.action == "record":
            receipt = create_receipt(
                run_id=options.run_id, experiment=options.experiment,
                variant=options.variant, stage=options.stage,
                evidence=options.evidence, source=options.source,
                repo_root=ROOT, metrics=_key_value(options.metric, "metric"),
                parameters=_key_value(options.param, "param"), seed=options.seed,
                dataset=options.dataset, checkpoint=options.checkpoint,
                external_run_url=options.external_run_url,
            )
            dest = write_receipt(receipt, options.out)
            print(json.dumps({"status": "RECORDED_NOT_QUALIFIED",
                              "path": str(dest),
                              "source_sha256": receipt["source_report"]["sha256"]},
                             sort_keys=True))
        elif options.action == "plan":
            from tac_osm.research_lab_integrations import plan, read_one_receipt
            recipe = plan(
                options.provider,
                receipt=read_one_receipt(options.receipt) if options.receipt else None,
                limit_minutes=options.minutes, cpus=options.cpus, gpus=options.gpus,
            )
            text = json.dumps(recipe, indent=2, sort_keys=True) + "\n"
            if options.output:
                options.output.parent.mkdir(parents=True, exist_ok=True)
                options.output.write_text(text, encoding="utf-8")
            print(text, end="")
        elif options.action == "status":
            index = summary(options.runs)
            print(json.dumps({"total_runs": index["total_runs"],
                              "experiments": index["experiments"],
                              "variants": index["variants"],
                              "scientifically_qualified_by_ledger": 0}, sort_keys=True))
        else:
            output = render_dashboard(options.runs, options.output)
            print(json.dumps({"status": "OFFLINE_REPORT",
                              "output": str(output)}, sort_keys=True))
        return 0
    except (LabError, OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "BLOCKED",
                          "reason": str(exc)[:200]}, sort_keys=True),
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
