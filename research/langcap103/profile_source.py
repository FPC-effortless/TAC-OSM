"""Original-FIT forward/backward resource probe; zero optimizer updates.

Torch-profiler arithmetic is explicitly PARTIAL. No training loss or DEV scores
are released, and no FLOP-parity authorization is inferred from this probe.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import resource
import time
from preflight import build, comparator_classes, counts, load_native, parameter_digest
from source import canonical, digest, next_byte_batch, verify_package
from attention_accounting import from_profiler, FORWARD, BACKWARD


def profile(receipt_path, corpus_path, out):
    import torch
    torch.set_num_threads(2)
    receipt, data = verify_package(receipt_path, corpus_path)
    x, y, ids = next_byte_batch(receipt, data, length=256, batch=2)
    inputs = torch.tensor(x, dtype=torch.long); targets = torch.tensor(y, dtype=torch.long)
    prior = json.loads((Path(__file__).parent.parent / "langcap103_results/PROFILE_RUN_41219856043.json").read_text())
    native = load_native(); classes = comparator_classes(torch)
    rows = []
    for cohort in prior["cohorts"]:
        for old in cohort["arms"]:
            torch.manual_seed(10301)
            config = old["configuration"]
            model = build(config, old["target"], native, classes).float().eval()
            if counts(model)["trainable_parameters"] != old["trainable_parameters"]:
                raise ValueError("Instantiated model differs from preserved candidate")
            before = parameter_digest(model)
            started = time.perf_counter()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                    record_shapes=True, profile_memory=True, with_flops=True) as prof:
                output = model(inputs)
                logits = output[0] if isinstance(output, tuple) else output
                torch.nn.functional.cross_entropy(logits.reshape(-1, 256), targets.reshape(-1)).backward()
            elapsed = time.perf_counter() - started
            if parameter_digest(model) != before:
                raise ValueError("Zero-update resource profile mutated model weights")
            if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                raise ValueError("Actual FIT graph disconnected or nonfinite")
            events = prof.key_averages()
            measured_flops = sum(e.flops or 0 for e in events)
            attention = from_profiler(torch, prof)
            if config["arm"] == "causal_transformer":
                operators=[r["operator"] for r in attention["events"]]
                if operators.count(FORWARD)!=config["depth"] or operators.count(BACKWARD)!=config["depth"]:
                    raise ValueError("Fused attention runtime does not match registered layer count; no extrapolated charge")
            rows.append({"configuration": config, "target": old["target"], **counts(model),
                "original_fit_batch_sha256": digest(canonical({"source": receipt["source_sha256"], "rows": ids})),
                "profiled_batch_size": 2, "profiled_sequence_length": 256,
                "instrumented_cpu_forward_backward_seconds": elapsed,
                "process_peak_rss_bytes_after_profile": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                "partial_profiler_flops": measured_flops,
                "fused_attention_accounting":attention,
                "partial_profile_plus_attention_matrix_units":measured_flops+attention["supplemental_dense_attention_units"],
                "flops_accounting_complete": False,
                "counted_operations": [{"name": e.key, "calls": e.count, "flops": e.flops}
                    for e in events if e.flops],
                "zero_or_unreported_flop_operations": sorted(e.key for e in events if not e.flops),
                "optimizer_updates": 0, "finite_connected_fit_backward": True})
            del model, output, logits, prof, events
    report = {"schema": "langcap103.original-fit-resource-probe.v1", "torch_version": torch.__version__,
        "source_sha256": receipt["source_sha256"], "batch_source_rows": ids, "profiles": rows,
        "status": "ORIGINAL_FIT_ZERO_UPDATE_PROFILE_NOT_TRAINING_AUTHORIZATION",
        "optimizer_updates": 0, "dev_scoring": False, "protected_test_access": False,
        "limitations": ["instrumented CPU forward/backward excludes optimizer updates and runtime IO",
            "FLOPs are partial; unreported operations are retained, never counted as zero-cost proof",
            "shape/schema-verified fused attention matrix products are supplemented without double-counting; other costs remain open",
            "RSS is cumulative process high-water mark, not isolated per-model peak",
            "no T4 throughput, training adequacy, trained decode speed or capability result"]}
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "models": len(rows), "optimizer_updates": 0,
                     "flops_accounting_complete": False}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--receipt", required=True, type=Path)
    p.add_argument("--corpus", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args(); profile(args.receipt, args.corpus, args.out)
