"""Single-host inference measurements for the predeclared first seed."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from data_belief_dev063 import BeliefWorld
from intervene_belief_dev063 import restore
from run_learned_stack_dev055 import write


@torch.no_grad()
def measure(directory, output):
    root = Path(directory)
    assert json.loads((root / 'summary.json').read_text())['status'] == 'COMPLETE'
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rows = []
    for path in sorted((root / 'group-93401').glob('*/report.json')):
        report, model = restore(path.parent)
        for batch in (1, 16):
            for history in (32, 128, 512, 1024):
                world = BeliefWorld(9963301, 0, history)
                keys = torch.tensor(world.identities, dtype=torch.float64)[None].expand(batch, -1, -1)
                candidates = torch.tensor(world.candidates()[0], dtype=torch.float64)[None].expand(batch, -1, -1)
                prices = torch.tensor([[world.error, world.inspection]], dtype=torch.float64).expand(batch, -1)
                observations = [torch.tensor(e, dtype=torch.float64)[None].expand(batch, -1) for e in world.events]
                observation_cpu = []
                for _ in range(3):
                    state = model.initial(keys)
                    start = time.process_time_ns()
                    for event in observations:
                        state = model.observe(state, event)
                    observation_cpu.append((time.process_time_ns() - start) / 1e6)
                for _ in range(3):
                    model.decide(state, candidates, prices)
                cpu, wall = [], []
                for _ in range(20):
                    start_cpu, start_wall = time.process_time_ns(), time.perf_counter_ns()
                    model.decide(state, candidates, prices)
                    wall.append((time.perf_counter_ns() - start_wall) / 1e6)
                    cpu.append((time.process_time_ns() - start_cpu) / 1e6)
                rows.append({
                    'arm': report['arm'], 'seed': report['seed'], 'batch': batch,
                    'history': history, 'dtype': 'float64',
                    'observation_cpu_ms_median': float(np.median(observation_cpu)),
                    'decision_cpu_ms_median': float(np.median(cpu)),
                    'decision_wall_ms_median': float(np.median(wall)),
                    'parameter_bytes': sum(p.numel() * p.element_size() for p in model.parameters()),
                    'retained_tensor_bytes': sum(v.numel() * v.element_size() for v in state.values()),
                    'retained_unique_storage_bytes': sum({v.untyped_storage().data_ptr(): v.untyped_storage().nbytes() for v in state.values()}.values()),
                    'admitted_records': int(state['present'].sum()) // batch,
                    'raw_records': len(observations),
                })
    assert len(rows) == 56
    write(output, {
        'scope': 'Single-host process CPU and wall time. Preconstructed public tensors; observation timing includes admission/buffering, decision timing includes belief reconstruction and policy. Batch times are per batch. Tensor bytes count logical elements; unique storage bytes deduplicate shared/view backing allocations, excluding allocator overhead and temporary workspaces. Concurrent CPU work can affect wall time. No production SLA, physical DRAM traffic, FLOPs or energy claim.',
        'host': {'python': sys.version, 'torch': str(torch.__version__)},
        'measurements': rows,
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    measure(args.directory, args.output)
