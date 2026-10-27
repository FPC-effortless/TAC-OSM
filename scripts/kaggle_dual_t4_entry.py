"""Kaggle Notebook entrypoint: HARD dual NVIDIA T4 requirement; no fallback.

Staged with a reviewed workload.py and config.json. This runs on Kaggle,
never on GitHub CPU. Each GPU handles its own frozen seed shard. No training
data or privileged reward/TEST scoring secrets are included in this script.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

CFG = Path(__file__).resolve().parent / "config.json"
HERE = Path(__file__).resolve().parent


def real_t4_pair() -> dict:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("CUDA-enabled PyTorch missing from Kaggle environment") from exc
    count = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(count)]
    if count != 2 or any("T4" not in name.upper() for name in names):
        raise RuntimeError(f"exact dual NVIDIA T4 allocation required, found {count}: {names}")
    for i in range(2):
        torch.cuda.set_device(i)
        sample = torch.ones(1, device=f"cuda:{i}")
        torch.cuda.synchronize(i)
        if sample.item() != 1:
            raise RuntimeError(f"T4 device {i} unavailable")
    return {"count": count, "names": names, "accelerator": "NvidiaTeslaT4"}


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run() -> dict:
    config = json.loads(CFG.read_text())
    workload = HERE / "workload.py"
    if not workload.is_file() or not CFG.is_file():
        raise RuntimeError("missing reviewed workload/config bundle")
    if _file_sha(workload) != config["workload_sha256"]:
        raise RuntimeError("Kaggle workload differs from reviewed source hash")
    if config.get("accelerator") != "NvidiaTeslaT4":
        raise RuntimeError("accelerator substitution denied")
    gpu = real_t4_pair()
    strategy = config["gpu_strategy"]
    seeds = config["seeds"]
    if strategy not in ("seed_shard", "ddp") or not seeds or len(set(seeds)) != len(seeds):
        raise RuntimeError("invalid dual GPU/frozen seed plan")
    outputs = Path("/kaggle/working") if Path("/kaggle/working").exists() else HERE / "outputs"
    outputs.mkdir(exist_ok=True, parents=True)
    report = {"schema": "tac-osm.kaggle-t4x2-runtime.v1", "run_id": config["run_id"],
              "requested_accelerator": "NvidiaTeslaT4", "observed": gpu,
              "workload_sha256": config["workload_sha256"],
              "strategy": strategy, "seeds": seeds,
              "status": "RUNNING_NOT_SCIENTIFICALLY_QUALIFIED",
              "results": [], "real_world_claim_authorized": False}
    path = outputs / "phone-research-runtime.json"
    path.write_text(json.dumps(report, indent=2))
    commands = []
    envs = []
    if strategy == "ddp":
        if len(seeds) != 1:
            raise RuntimeError("DDP pilot expects exactly one frozen training seed")
        commands = [[sys.executable, "-m", "torch.distributed.run",
                     "--standalone", "--nproc_per_node=2", str(workload),
                     "--seed", str(seeds[0])]]
        envs = [dict(os.environ, CUDA_VISIBLE_DEVICES="0,1")]
    else:
        if len(seeds) < 2:
            raise RuntimeError("dual T4 seed shards require >= 2 independent seeds")
        for i, seed in enumerate(seeds):
            gpu_id = i % 2
            commands.append([sys.executable, str(workload),
                             "--seed", str(seed), "--device", "cuda:0",
                             "--output", str(outputs / f"seed-{seed}.json")])
            envs.append(dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu_id),
                             TAC_OSM_PHYSICAL_T4_SLOT=str(gpu_id)))
    # Bounded two-way concurrency: no third simultaneous GPU model.
    pool = []
    started = time.time()
    failed = False
    for i, cmd in enumerate(commands):
        log = (outputs / f"worker-{i}.log").open("w")
        child = subprocess.Popen(cmd, env=envs[i], cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
        pool.append((i, child, log))
        if len(pool) == 2 or i == len(commands)-1:
            for slot, process, stream in pool:
                exit_code = process.wait()
                stream.close()
                report["results"].append({"worker_index": slot, "exit_code": exit_code})
                if exit_code:
                    failed = True
            pool.clear()
            if failed:
                break
    report["elapsed_seconds"] = int(time.time() - started)
    report["status"] = "FAILED" if failed else "COMPLETED_UNQUALIFIED_REQUIRES_INDEPENDENT_EVALUATION"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if failed:
        raise RuntimeError("dual-T4 model workload failed; inspect worker logs")
    return report


if __name__ == "__main__":
    run()
