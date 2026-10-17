"""Execute and deterministically replay newly authored public Veritas worlds."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tac_osm.development.veritas_public_challenge import POLICIES, make_episode, run_episode

PROTOCOL = ROOT / "research/dev050/study-protocol.json"


def run(output, veritas_root):
    protocol = json.loads(PROTOCOL.read_text())
    veritas_root = Path(veritas_root).resolve()
    import investigation_world
    assert Path(investigation_world.__file__).resolve().is_relative_to(veritas_root)
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=veritas_root,
                                   text=True).strip() == protocol["veritas_source"]
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=veritas_root,
                                       text=True).strip(), "Veritas source must be clean"
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT,
                                       text=True).strip(), "Freeze TAC source first"
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    seed_results = []
    replayed = 0
    for seed in protocol["seeds"]:
        rows = []
        for index in range(protocol["episodes_per_seed"]):
            episode = make_episode(seed, index)
            for policy in POLICIES:
                first = run_episode(episode, policy)
                assert first == run_episode(make_episode(seed, index), policy)
                replayed += 1
                rows.append(dict(seed=seed, index=index, **first))
        path = output / f"{seed}.json"
        path.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
        summary = {}
        for policy in POLICIES:
            selected = [r for r in rows if r["policy"] == policy]
            summary[policy] = dict(cost_per_entity=statistics.mean(r["total_cost"]/2 for r in selected),
                failures=sum(r["failures"] for r in selected),
                complete_episodes=sum(r["verification"]["state"] == 1 for r in selected),
                blocked_actions=sum(e["blocked"] for r in selected for e in r["trace"]),
                mean_tool_calls=statistics.mean(r["verification"]["tool_calls"] for r in selected))
        seed_results.append(dict(seed=seed, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                 policies=summary))
        print(json.dumps(seed_results[-1]), flush=True)
    summary = {p: dict(mean_cost=statistics.mean(r["policies"][p]["cost_per_entity"] for r in seed_results),
        seed_sd=statistics.stdev(r["policies"][p]["cost_per_entity"] for r in seed_results),
        failures=sum(r["policies"][p]["failures"] for r in seed_results),
        complete_episodes=sum(r["policies"][p]["complete_episodes"] for r in seed_results),
        blocked_actions=sum(r["policies"][p]["blocked_actions"] for r in seed_results)) for p in POLICIES}
    report = dict(status="REPLAYED_PUBLIC_ENGINEERING_NOT_FRONTIER_QUALIFIED",
        tac_source=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        veritas_source=protocol["veritas_source"],
        protocol_sha256=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        replayed_policy_episodes=replayed, seeds=seed_results, summary=summary,
        limitations=protocol["limitations"])
    (output / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--veritas-root", required=True)
    args = parser.parse_args()
    run(args.output, args.veritas_root)
