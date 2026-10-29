#!/usr/bin/env python3
"""LANGPROFILE059: ATen callsite attribution of frozen TAC _advance.

No changes to model, graph, checkpoint or next-token semantics. PyTorch
with_stack callsite attribution is best-effort and can be missing depending
on build. A missing stack is UNATTRIBUTED, never guessed from operator name.
The profiled times are not production decoder latency.
"""
from __future__ import annotations
import argparse
from collections import defaultdict, Counter
import hashlib
import inspect
import json
import math
from pathlib import Path
import re
import sys

import torch
from torch.profiler import profile, ProfilerActivity

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/"src"),
                str(ROOT/"research/mathematical-engineering/langstream043")]
from incremental import IncrementalTAC043, SHAPES
from scripts.langdecode057_validation import BatchedValidationTAC057
from scripts.langdecode057_benchmark import parity, exact_state
from scripts.langprofile058_ops import operator_group
from scripts.langrecomp052_replay import load_original
from scripts.langlayer035_sampling import fit_sources

CALLS = 12
FOCUS_OPS = ("aten::_to_copy", "aten::copy_", "aten::as_strided",
             "aten::unsqueeze", "aten::mul", "aten::addmm",
             "aten::matmul", "aten::empty_strided")
FRAME_RE = re.compile(r"(?:^|[/\\])incremental\.py(?::|\()(\d+)\)?(?:$|[:\s])")


def original_line_range() -> tuple[int, int]:
    lines, start = inspect.getsourcelines(IncrementalTAC043._advance)
    return start, start + len(lines) - 1


def resolve_advance_callsite(stack, allowed: tuple[int, int]) -> int | None:
    """Find actual source file/line in profiler frame, reject false inferences."""
    if not isinstance(stack, (list, tuple)):
        return None
    for frame in stack:
        m = FRAME_RE.search(str(frame))
        if m:
            line = int(m.group(1))
            if allowed[0] <= line <= allowed[1]:
                return line
    return None


def collapse(events: list, allowed: tuple[int, int], top_n: int = 18) -> dict:
    if top_n < 1:
        raise ValueError("top_n must be positive")
    by_line = defaultdict(lambda: {"self_cpu_us": 0.0, "calls": 0,
                                   "operators": Counter()})
    unmatched = Counter()
    total = 0.0
    covered = 0.0
    count = 0
    focused = Counter()
    for event in events:
        name = str(getattr(event, "name", ""))
        if not name.startswith("aten::"):
            continue
        cost = float(getattr(event, "self_cpu_time_total", float("nan")))
        if not math.isfinite(cost) or cost < 0:
            raise ValueError("invalid profiled self CPU")
        calls = int(getattr(event, "count", 1))
        if calls < 1:
            raise ValueError("invalid profiler event calls")
        total += cost
        count += 1
        if name in FOCUS_OPS:
            focused[name] += cost
        location = resolve_advance_callsite(getattr(event, "stack", ()), allowed)
        if location is None:
            unmatched[name] += cost
            continue
        row = by_line[location]
        row["self_cpu_us"] += cost
        row["calls"] += calls
        row["operators"][name] += cost
        covered += cost
    if total <= 0 or count == 0:
        raise ValueError("no profiled ATen operations")
    breakdown = [
        {"source_line": line, "self_cpu_us": row["self_cpu_us"],
         "share_of_all_aten_self_cpu": row["self_cpu_us"]/total,
         "calls": row["calls"],
         "dominant_operators": [
             {"name": name, "self_cpu_us": cost}
             for name, cost in row["operators"].most_common(5)]}
        for line,row in by_line.items()]
    breakdown.sort(key=lambda row: (-row["self_cpu_us"], row["source_line"]))
    return {
        "total_aten_self_cpu_us": total,
        "attributed_self_cpu_us": covered,
        "attributed_fraction": covered/total,
        "unattributed_self_cpu_us": total-covered,
        "unattributed_operator_totals": dict(unmatched.most_common(15)),
        "focus_operator_self_cpu_us": dict(focused.most_common()),
        "top_advance_lines": breakdown[:top_n],
        "attribution": "PYTORCH_STACK_FRAMES_ONLY_NO_GUESSED_SOURCE",
        "status": ("SOURCE_CALLSITES_RECORDED" if breakdown else
                   "NO_ADVANCE_PYTHON_STACKS_AVAILABLE"),
    }


def profile_raw(engine, token, state, mask) -> dict:
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True,
                 profile_memory=False, with_stack=True) as trace:
        for _ in range(CALLS):
            engine._advance(token, state, mask)
    lohi = original_line_range()
    summary = collapse(trace.events(), lohi)
    summary["calls"] = CALLS
    summary["advance_source_line_range"] = list(lohi)
    summary["instrumentation"] = "torch.profiler CPU with_stack=True; event self CPU only"
    return summary


def run(folder: Path) -> dict:
    if torch.cuda.is_initialized():
        raise ValueError("CPU-only study")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    fit = fit_sources()
    model, receipt = load_original(folder, fit)
    engines = {"original": IncrementalTAC043(model, "cap_all"),
               "batched": BatchedValidationTAC057(model, "cap_all")}
    original_hash = engines["original"]._weight_hash()
    owner = next(k for k in sorted(fit) if len(fit[k]) >= 544)
    rows = []
    with torch.inference_mode():
        for length in (128, 512):
            raw = fit[owner][:length]
            tokens = torch.tensor([list(raw)],dtype=torch.long)
            generated, checks = parity(engines,tokens,(owner,))
            logits, initial_state = engines["batched"].prefill(tokens, owners=(owner,))
            token = generated[0].clone()
            mask = torch.ones_like(token,dtype=torch.bool)
            y, state = engines["batched"].step(token,initial_state,owners=(owner,))
            z, unguarded = engines["batched"]._advance(token,initial_state,mask)
            if not torch.equal(y,z):
                raise ValueError("step-vs-transition logits differ")
            exact_state(state,unguarded)
            before={name:value.clone() for name,value in initial_state.tensors.items()}
            summary=profile_raw(engines["batched"],token,initial_state,mask)
            if any(not torch.equal(initial_state.tensors[name],before[name]) for name in SHAPES):
                raise ValueError("profiling mutated persistent original state")
            if engines["original"]._weight_hash()!=original_hash:
                raise ValueError("profiling mutated checkpoint")
            rows.append({"prompt_bytes":length,
                         "source_sha256":hashlib.sha256(raw).hexdigest(),
                         "greedy_token_sha256":hashlib.sha256(bytes([int(token[0])])).hexdigest(),
                         "bitwise_reference_comparison":checks,
                         "step_to_transition_bitwise":True,
                         "persistent_state_bytes":initial_state.tensor_bytes(),
                         "source_line_profile":summary})
    return {"study":"LANGPROFILE059",
            "status":"SOURCE_CALLSITE_DIAGNOSTIC_NO_FIT_NOT_INDEPENDENTLY_QUALIFIED",
            "checkpoint":receipt,"source_owner":owner,"source_split":"FIT_ONLY_INFERENCE",
            "trained_parameters":499623,"new_training_updates":0,"gpu_minutes":0,
            "protected_TEST":"NOT_ACCESSED","results":rows,
            "limits":["PyTorch profiler Python stacks can be absent or incomplete",
                      "source frame accounting is instrumented ATen self CPU, not real wall time",
                      "single hosted CPU and original exposed FIT owner",
                      "no new model, no speedup, no DEV/TEST, mobile, CUDA or quality claims",
                      "LANGDECODE057 p95 tail failure remains unresolved"],
            "independently_qualified":False,"checkpoint_promotion":False}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--tac-dir",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError("immutable source callsite results")
    data=run(args.tac_dir)
    args.output.parent.mkdir(exist_ok=True,parents=True)
    with args.output.open("x") as fp:
        fp.write(json.dumps(data,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"study":data["study"],"results":[
        {"prompt":r["prompt_bytes"],"coverage":r["source_line_profile"]["attributed_fraction"],
         "status":r["source_line_profile"]["status"],
         "top_lines":r["source_line_profile"]["top_advance_lines"][:8]}
        for r in data["results"]]},sort_keys=True),flush=True)


if __name__=="__main__":
    main()
