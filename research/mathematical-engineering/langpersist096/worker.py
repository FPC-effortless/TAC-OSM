"""Separated acquisition/query worker. Never receives targets or source records."""
import time
PROCESS_START = time.perf_counter()
import argparse
import json
from pathlib import Path
import resource

import numpy as np
import torch

from adapters import (CONDITIONS, FrozenAdapter096, intervene, join_states,
                      load_state, make_model, save_state, select_state, sha256)


def load_bytes(path, width):
    if Path(path).stat().st_size > 8*4096*width+4096:
        raise ValueError("oversized observed-data array")
    x = np.load(path, allow_pickle=False)
    if not isinstance(x, np.ndarray) or x.dtype not in (np.dtype("uint8"), np.dtype("int64")) or x.ndim != 2 or not 2 <= len(x) <= 4096 or x.shape[1] != width or x.min() < 0 or x.max() > 255:
        raise ValueError("bounded byte matrix with exact registered width required")
    return x


def peak_rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024


def run(args):
    torch.set_num_threads(1)
    model = make_model(args.arm, args.checkpoint, args.sha256)
    adapter = FrozenAdapter096(model, args.arm)
    model_bytes = sum(x.numel()*x.element_size() for x in model.state_dict().values())
    if args.mode == "acquire":
        if args.acquisition is None or args.queries is not None or args.donors is not None:
            raise ValueError("acquisition process receives acquisition array only")
        observed = load_bytes(args.acquisition, 128)
        setup = time.perf_counter()-PROCESS_START
        started = time.perf_counter(); states = []
        for lo in range(0, len(observed), args.batch):
            hi = min(lo+args.batch, len(observed))
            owners = tuple(f"record-{i}" for i in range(lo, hi))
            states.append(adapter.observe(adapter.start(owners), torch.from_numpy(observed[lo:hi].copy()))[1])
        state = join_states(states)
        costs = dict(setup_seconds=setup, acquisition_seconds=time.perf_counter()-started,
                     query_seconds=0., generation_seconds=0., model_bytes=model_bytes,
                     state_payload_bytes=state.tensor_bytes(), serialized_metadata_bytes=0,
                     peak_rss_bytes=peak_rss())
        metadata = save_state(args.state, state, costs, sha256(args.acquisition))
        report = {**metadata, "state_file_sha256": sha256(args.state), "optimizer_steps": 0}
        with Path(args.output).open("x") as handle:
            json.dump(report, handle, sort_keys=True, allow_nan=False)
        return
    if args.queries is None or args.donors is None or args.acquisition is not None:
        raise ValueError("query process receives queries, saved state and donor indices only")
    queries = load_bytes(args.queries, 64)
    donors = np.load(args.donors, allow_pickle=False)
    if not isinstance(donors, np.ndarray) or donors.dtype != np.int64 or donors.shape != (len(queries),):
        raise ValueError("registered donor index vector required")
    state, saved = load_state(args.state, adapter)
    if len(state.owners) != len(queries):
        raise ValueError("query/state row mismatch")
    costs = dict(saved["cost"])
    costs["setup_seconds"] += time.perf_counter()-PROCESS_START
    costs["query_seconds"] = 0.; costs["generation_seconds"] = 0.
    log_probs = np.empty((4, len(queries), 256), np.float32)
    greedy = np.empty((4, len(queries), 4), np.uint8)
    per_condition = {}
    for ci, condition in enumerate(CONDITIONS):
        started = time.perf_counter()
        intervened = intervene(state, condition, donors)
        elapsed = time.perf_counter()-started
        costs["query_seconds"] += elapsed
        current_cost = dict(intervention_seconds=elapsed, query_seconds=0., generation_seconds=0.)
        for lo in range(0, len(queries), args.batch):
            hi = min(lo+args.batch, len(queries))
            started = time.perf_counter()
            logits, current = adapter.observe(select_state(intervened, range(lo, hi)), torch.from_numpy(queries[lo:hi].copy()))
            log_probs[ci, lo:hi] = logits.log_softmax(-1).numpy()
            elapsed = time.perf_counter()-started
            costs["query_seconds"] += elapsed
            current_cost["query_seconds"] += elapsed
            started = time.perf_counter()
            token = logits.argmax(-1)
            greedy[ci, lo:hi, 0] = token.numpy()
            for j in range(1, 4):
                logits, current = adapter.step(current, token)
                token = logits.argmax(-1)
                greedy[ci, lo:hi, j] = token.numpy()
            elapsed = time.perf_counter()-started
            costs["generation_seconds"] += elapsed
            current_cost["generation_seconds"] += elapsed
        per_condition[condition] = current_cost
    costs["peak_rss_bytes"] = max(costs["peak_rss_bytes"], peak_rss())
    metadata = dict(schema="langpersist096.predictions.v1", arm=args.arm, conditions=CONDITIONS,
                    checkpoint_sha256=args.sha256, implementation_sha256=adapter.implementation,
                    acquisition_sha256=saved["acquisition_sha256"], query_sha256=sha256(args.queries),
                    donor_sha256=sha256(args.donors), cost=costs, optimizer_steps=0,
                    per_condition_costs=per_condition, query_condition_count=4,
                    original_checkpoint_metadata=getattr(model, "_source_metadata_096", "CONSTRUCTED_TEST_FIXTURE_ONLY"),
                    state_payload_bytes_per_record=state.tensor_bytes()//len(state.owners),
                    worker_information_boundary="query bytes plus complete saved state; no acquisition bytes or targets",
                    evidence="internal_frozen_checkpoint_intervention_not_architecture_superiority")
    with Path(args.output).open("xb") as handle:
        np.savez_compressed(handle, log_probs=log_probs, greedy=greedy,
                            argmax=log_probs.argmax(-1).astype(np.uint8),
                            metadata=np.array(json.dumps(metadata, sort_keys=True, allow_nan=False)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("acquire", "query"), required=True)
    p.add_argument("--arm", choices=("tac", "gru", "ssm"), required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--sha256", required=True)
    p.add_argument("--acquisition", type=Path)
    p.add_argument("--queries", type=Path)
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--donors", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch", type=int, choices=(16, 32), default=32)
    args = p.parse_args()
    if args.output.exists() or (args.mode == "acquire" and args.state.exists()):
        raise FileExistsError("do not overwrite existing research artifacts")
    run(args)  # observe() owns inference mode; model parameters retain version guards


if __name__ == "__main__":
    main()
