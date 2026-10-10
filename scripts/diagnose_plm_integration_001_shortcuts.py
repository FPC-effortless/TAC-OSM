#!/usr/bin/env python3
"""Development-only evidence that exact-key lookup bypasses learned addressing.

Never emits a confirmatory TACOSM-PLM-INTEGRATION-001 measurement. Uses a
separate development seed namespace, checks opposite-history/reset controls
and counts index construction/updates rather than hiding them as free.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tac_osm.integration_001_diagnostic import diagnostic_rows  # noqa: E402

ID = "TACOSM-PLM-INTEGRATION-001-DEV-EXACT-LOOKUP-NEGATIVE-CONTROL"
OUT = ROOT / "artifacts" / (ID + ".json")


def main() -> None:
    rows = diagnostic_rows()
    if not rows:
        raise RuntimeError("no development measurements")
    exact = sorted(set(x["exact_key_lookup_accuracy"] for x in rows))
    without_writes = sorted(set(x["no_intervening_write_accuracy"] for x in rows))
    reset = sorted(set(x["reset_accuracy"] for x in rows))
    if exact != [1.0] or without_writes != [1.0] or reset != [0.5]:
        raise AssertionError("diagnostic invariant failed; preserve failure not PASS")
    contract = ROOT / "contracts/TACOSM-PLM-INTEGRATION-001.json"
    generator = ROOT / "src/tac_osm/integration_001_benchmark.py"
    diagnostic = ROOT / "src/tac_osm/integration_001_diagnostic.py"
    record = {
        "experiment_id": ID,
        "status": "measured-development-only",
        "confirmatory_status": "NOT_RUN",
        "model_trained": False,
        "claim": (
            "With exact target-key queries, a conventional dictionary retrieves "
            "the target perfectly despite same-format one-bit-neighbor distractor "
            "writes. The no-intervening-write control is also perfect, so this "
            "instrument alone does not demonstrate learned addressing or competing "
            "memory retention. A dense top-K scorer inspects H+D candidates and "
            "cannot claim sublinear address cost."
        ),
        "provenance": {
            "contract_sha256": hashlib.sha256(contract.read_bytes()).hexdigest(),
            "generator_sha256": hashlib.sha256(generator.read_bytes()).hexdigest(),
            "diagnostic_sha256": hashlib.sha256(diagnostic.read_bytes()).hexdigest(),
            "seed_namespace": 97_000_000,
            "registered_confirmatory_eval_namespace_used": False,
        },
        "summary": {
            "cell_rows": len(rows),
            "seed_count": len(set(x["seed"] for x in rows)),
            "exact_key_lookup_accuracy_min_max": [min(exact), max(exact)],
            "no_write_lookup_accuracy_min_max": [min(without_writes), max(without_writes)],
            "reset_accuracy_min_max": [min(reset), max(reset)],
            "dense_scans_scale_with_full_history": True,
            "amortized_index_build_is_not_free": True,
        },
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record["summary"], indent=2, sort_keys=True))
    print("RESULT: DEVELOPMENT NEGATIVE CONTROL ONLY; INTEGRATION-001 NOT RUN")
    print("Artifact:", OUT)


if __name__ == "__main__":
    main()
