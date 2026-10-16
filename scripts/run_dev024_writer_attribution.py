"""DEV-024 from-scratch matched training and exact checkpoint/action replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import torch  # noqa: E402
from tac_osm.contract import load_contract  # noqa: E402
from tac_osm.development.writer_attribution import ARMS, WriterAttributionControl  # noqa: E402
from tac_osm.development.noisy_memory import parameter_hash  # noqa: E402
from run_memory_controls_dev import train, evaluate, digest, component_hash, SOURCES as PRIOR_SOURCES  # noqa: E402
from audit_memory_controls_dev import metrics  # noqa: E402

SOURCES = tuple(PRIOR_SOURCES) + (
    "scripts/run_dev024_writer_attribution.py", "scripts/audit_dev024_writer_attribution.py",
    "src/tac_osm/development/retention_controls.py", "src/tac_osm/development/entity_aging.py", "src/tac_osm/development/writer_attribution.py", "contracts/DEV-024.json",
    "docs/DEV-024-WRITER-ATTRIBUTION-PROTOCOL.md")


def specification():
    prior = json.loads((ROOT / "contracts/DEV-018.json").read_text())
    spec = json.loads((ROOT / "contracts/DEV-024.json").read_text())
    prior.update({k: spec[k] for k in ("steps", "eval_steps", "h_levels", "seeds", "regimes")})
    contract = load_contract("DEV-024")
    contract.require_arms(ARMS)
    contract.require_steps(prior["steps"])
    contract.require_levels(prior["h_levels"])
    contract.require_eval_steps(prior["eval_steps"])
    return prior


def run(seed, regime, arm, output, smoke=False, replay=False):
    spec = specification()
    if seed not in spec["seeds"] or regime not in spec["regimes"] or arm not in ARMS:
        raise ValueError("unregistered cell")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    sources = {p: digest(ROOT / p) for p in SOURCES}
    output.mkdir(parents=True, exist_ok=True)
    path = output / "report.json"
    model = WriterAttributionControl(arm, seed + 24000)
    checkpoint = output / "model.pt"
    if replay:
        report = json.loads(path.read_text())
        assert (report["source_sha256"], report["seed"], report["regime"], report["arm"], report["smoke"]) == (
            sources, seed, regime, arm, smoke)
        assert digest(checkpoint) == report["model"]["sha256"]
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        assert parameter_hash(model) == report["training"]["final_hash"]
    else:
        # Preserve DEV-018 recipe exactly, using fresh numerical seed namespaces.
        common = {p: component_hash(model, p) for p in ("event_encoder.", "query_encoder.", "head.")}
        training = train(model, seed + 6000, regime, spec, smoke)
        training["initial_common_components"] = common
        torch.save(model.state_dict(), checkpoint)
        report = {"study": "DEV-024", "status": "ADAPTIVE_TRAINED_INTERVENTION_NOT_CONFIRMATORY",
                  "seed": seed, "regime": regime, "arm": arm, "smoke": smoke,
                  "torch": torch.__version__, "source_sha256": sources, "training": training,
                  "model": {"sha256": digest(checkpoint), "state_scalars": model.state_scalars,
                            "parameters": sum(p.numel() for p in model.parameters()),
                            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}}
    rows, timing = evaluate(model, seed + 24000, spec, smoke)
    assert not set(r["fingerprint"] for r in rows) & set(report["training"]["fingerprints"])
    summary = metrics(rows)
    if replay:
        assert rows == report["rows"] and summary == report["summary"], "checkpoint/action/cost replay mismatch"
        (output / "replay.json").write_text(json.dumps({"verified": True, "checkpoint_sha256": digest(checkpoint),
            "report_sha256": digest(path), "source_sha256": sources}) + "\n")
    else:
        report.update(rows=rows, summary=summary, evaluation_timing=timing)
        path.write_text(json.dumps(report) + "\n")
    print("DEV024_" + ("REPLAY " if replay else "TRAINED ") + json.dumps({
        "seed": seed, "regime": regime, "arm": arm, "summary": summary}), flush=True)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--regime", choices=("short", "long"), required=True)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--replay", action="store_true")
    a = p.parse_args()
    run(a.seed, a.regime, a.arm, a.output, a.smoke, a.replay)
