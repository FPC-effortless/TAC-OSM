"""Descriptive one-thread encoder timings; no power/FLOP efficiency claim."""
import argparse
import json
from pathlib import Path
import statistics
import time
import torch
from run_joint_repair_dev054 import setup
from run_learned_belief_dev052b import write
from tac_osm.development import joint_repair as j
from tac_osm.development.adaptive_repair import baseline_forecasts


def profile(directory, output, veritas_root):
    setup(veritas_root)
    out=Path(directory);common=json.loads((out/'common.json').read_text())
    streams=json.loads((out/'streams.json').read_text())['validation'][:8]
    hazard=json.loads((out/'baseline-calibration.json').read_text())['hmm']['setting']['hazard']
    rows=[]
    for arm in j.NEURAL:
        model=j.JointModel(7401,arm)
        model.load_state_dict(torch.load(out/f'seed7401/{arm}.pt',weights_only=True));model.eval()
        plain=j.data(streams[:1],common['parameters'],hazard,arm=='gru_hmm')
        model.sequence(plain[0]).square().mean().backward()
        unused={k:p.numel() for k,p in model.named_parameters() if p.grad is None}
        timings=[];filter_timings=[]
        with torch.no_grad():
            for repeat in range(40):
                stream=streams[repeat%len(streams)]
                x=j.data([stream],common['parameters'],hazard,arm=='gru_hmm')[0]
                start=time.perf_counter()
                if arm=='gru_hmm': baseline_forecasts([stream],'hmm_plugin',common['parameters'],{'hazard':hazard})
                filter_time=time.perf_counter()-start
                start=time.perf_counter();model.sequence(x); elapsed=time.perf_counter()-start
                if repeat>=8:timings.append(elapsed+filter_time);filter_timings.append(filter_time)
        rows.append({'arm':arm,'stored_parameters':sum(p.numel() for p in model.parameters()),'unused_parameter_tensors':unused,
                     'parameters_with_gradient_path':sum(p.numel() for p in model.parameters())-sum(unused.values()),
                     'median_32_event_encoder_seconds':statistics.median(timings),'median_hybrid_filter_seconds':statistics.median(filter_timings),
                     'p90_32_event_encoder_seconds':sorted(timings)[int(.9*len(timings))],
                     'ideal_retained_neural_state_bytes':model.state_scalars*4,
                     'hybrid_numpy_filter_posterior_bytes':16 if arm=='gru_hmm' else 0})
    result={'study':'DEV-054','scope':'Single CPU thread, first model seed, eight validation streams,8 warmup/32 retained timings; includes hybrid filtering but excludes token conversion, planner, native runtime, logging and fixture generation. Timing during other work may be contended. No FLOP, power or production latency inference.',
            'note':'MTSK inherited write_proj is unused by the step interface but remains in stored parameter count. Matching is stored parameters, not equal active parameters. The fit metadata hybrid8B is theoretical FP32 posterior; actual NumPy float64 posterior is16B, reported here.', 'rows':rows}
    write(Path(output),result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',required=True);p.add_argument('--output',required=True);p.add_argument('--veritas-root',required=True)
    a=p.parse_args();profile(a.directory,a.output,a.veritas_root)
