"""Registered, first-seed runtime comparison; no weight or output changes."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from tac_osm.development.semantic_runtime import observe_many
from data_semantic_dev065 import BeliefWorld
from intervene_semantic_dev065 import restore
from run_learned_stack_dev055 import write, digest


@torch.no_grad()
def measure(directory, output, smoke=False):
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    root = Path(directory); summary = json.loads((root/'summary.json').read_text())
    assert summary['status'] == ('SMOKE_ONLY' if smoke else 'COMPLETE')
    seed = 10096501 if smoke else 96501
    start = time.process_time(); rows = []
    for path in sorted((root/f'group-{seed}').glob('*/report.json')):
        report, model = restore(path.parent)
        for batch in ((1,) if smoke else (1, 16)):
            for history in ((32,) if smoke else (32, 128, 512, 1024)):
                w = BeliefWorld(9966599, 0, history)
                keys = torch.tensor(w.identities, dtype=torch.float64)[None].expand(batch, -1, -1)
                events = torch.tensor(np.stack(w.events), dtype=torch.float64)[None].expand(batch, -1, -1)
                observations = list(events.unbind(1))
                candidates = torch.tensor(w.candidates()[0], dtype=torch.float64)[None].expand(batch, -1, -1)
                prices = torch.tensor([[w.error, w.inspection]], dtype=torch.float64).expand(batch, -1)
                def sequential():
                    state = model.initial(keys)
                    for event in observations:
                        state = model.observe(state, event)
                    return state
                def batched():
                    return observe_many(model, model.initial(keys), events)
                for _ in range(3):
                    a, b = sequential(), batched()
                for key in a:
                    torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
                left, right = model.decide(a, candidates, prices), model.decide(b, candidates, prices)
                for key in left:
                    torch.testing.assert_close(left[key], right[key], atol=0, rtol=0)
                times = {mode: dict(cpu=[], wall=[]) for mode in ('sequential', 'batched')}
                for _ in range(2 if smoke else 20):
                    for mode, function in (('sequential', sequential), ('batched', batched)):
                        cpu, wall = time.process_time_ns(), time.perf_counter_ns()
                        function()
                        times[mode]['cpu'].append((time.process_time_ns()-cpu)/1e6)
                        times[mode]['wall'].append((time.perf_counter_ns()-wall)/1e6)
                old = float(np.median(times['sequential']['cpu'])); new = float(np.median(times['batched']['cpu']))
                rows.append(dict(arm=report['arm'], seed=seed, batch=batch, history=history,
                    sequential_cpu_ms=old, batched_cpu_ms=new, cpu_speedup=old/new,
                    sequential_wall_ms=float(np.median(times['sequential']['wall'])),
                    batched_wall_ms=float(np.median(times['batched']['wall'])),
                    raw_event_logical_bytes=events.numel()*events.element_size(),
                    raw_event_unique_storage_bytes=events.untyped_storage().nbytes(),
                    retained_tensor_bytes=sum(v.numel()*v.element_size() for v in b.values()),
                    retained_unique_storage_bytes=sum({v.untyped_storage().data_ptr():v.untyped_storage().nbytes() for v in b.values()}.values()),
                    admitted_records=int(b['present'].sum())//batch, parity='EXACT',
                    raw_repetitions=times))
                if time.process_time()-start > 120:
                    write(output, dict(status='CAPPED', measurements=rows, cpu_seconds=time.process_time()-start))
                    raise TimeoutError('Registered runtime profile CPU cap')
    assert len(rows) == (6 if smoke else 48)
    median = float(np.median([r['cpu_speedup'] for r in rows]))
    write(output, dict(status='SMOKE_ONLY' if smoke else 'COMPLETE', cpu_seconds=time.process_time()-start,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256={name:digest(name) for name in (
            'src/tac_osm/development/semantic_runtime.py',
            'scripts/runtime_semantic_batched_dev065.py',
            'scripts/benchmark_semantic_runtime_dev065.py',
            'research/dev065-semantic-stability/RUNTIME-PROTOCOL.md')},
        median_cpu_speedup=median, engineering_timing_gate=median >= 1.2 and all(r['batched_cpu_ms'] <= 1.10*r['sequential_cpu_ms'] for r in rows),
        measurements=rows,
        host=dict(torch=str(torch.__version__),
            cpu_models=sorted({line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')})),
        scope='One host, float64/thread1, fixed first seed, preconstructed same raw tensors. CPU ingestion includes validation/gate/FIFO; JSON parsing and tensor creation excluded. All raw inputs scanned. Retained/unique storage exclude allocator and temporary workspaces. No physical DRAM, FLOPs, energy, C5 or learned-quality improvement claim. Output parity also requires the separate all-checkpoint and portable audit.'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--directory', required=True); p.add_argument('--output', required=True); p.add_argument('--smoke', action='store_true')
    a = p.parse_args(); measure(a.directory, a.output, a.smoke)
