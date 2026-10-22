#!/usr/bin/env python3
"""ME056 authentic robot concept-CED vs equally-resourced GRU.

Reuse the ORIGINAL ME054 publisher-data loader, source audit, FIT-only scaling,
real logged sample windows, dual physical/takeover loss, real verification,
checkpoint freezing, DEV/TEST scorer and results archiver. Never simulate data.

Before first fit, audit true source bytes and pre-profile *the exact training
update* on an authentic original FIT batch. Unequal parameters/data/computational
work BLOCK any architecture-comparison fits. CPU speed is recorded, NOT matched.
This script's real-world results, if any, are developmental: original robot
TEST was already opened by ME052/ME054.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import torch
from torch.profiler import ProfilerActivity,profile

from scripts import me054_train_real_stable_robot as base
from tac_osm.real_robot_me056_concept import (
    RealRobotConceptCED,MatchedStableRobotGRU,count_active,
    nearest_real_gru_width)

torch.set_num_threads(1)
REGISTRATION=Path("research/mathematical-engineering/me056/FAIR_COMPARISON.json")
EXPECTED=REGISTRATION
_original_source_preflight=base.verify_real_source_before_any_optimizer
ARMS=("native_tac","gru_world")
SEEDS=(56001,56002,56003,56004,56005)


def network(name,delta_scale):
    if name=="native_tac":
        return RealRobotConceptCED(delta_scale)
    if name=="gru_world":
        return MatchedStableRobotGRU(delta_scale,hidden_size=44)
    raise ValueError("unrecognized real robot prediction arm")


def objective(model,batch):
    states,actions,human=batch
    state=model.initial(len(states))
    # Real original 256 observed historical frames: no simulated burn-in.
    with torch.no_grad():
        for t in range(256):
            state,_,_,_,_=base.base.single_step(model,state,states,actions,human,t)
    loss=None
    for t in range(256,272):
        observed_now=states[:,t]
        state,forecast,measured_future,onset,eligible=base.base.single_step(
            model,state,states,actions,human,t)
        local,_=base.actual_real_physical_and_takeover_loss(
            model,forecast,observed_now,measured_future,onset,eligible)
        loss=local if loss is None else loss+local
    return loss/16


def profile_real_matched_step(actual_fit,stats,scale,folder):
    norm=base.base.normalize(actual_fit,stats)
    real_schedule,_=base.base.sample_schedule(norm,SEEDS[0])
    states,actions,human=base.base.get_batch(norm,real_schedule[0])
    batch=(states,actions,human)
    measurements=[]
    for name in ARMS:
        torch.manual_seed(SEEDS[0])
        model=network(name,scale)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        start=time.process_time()
        with profile(activities=[ProfilerActivity.CPU],with_flops=True) as prof:
            optimizer.zero_grad(set_to_none=True)
            loss=objective(model,batch)
            if not torch.isfinite(loss):
                raise RuntimeError("original FIT preflight loss nonfinite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
        values=prof.key_averages()
        flops=sum(int(event.flops) for event in values)
        if flops<=0:
            raise RuntimeError("authentic FIT profiler recorded no operator FLOPs")
        measurements.append({"arm":name,"parameters":count_active(model),
                             "one_genuine_optimizer_update_profiled_operator_flops":flops,
                             "cpu_seconds_efficiency_only":time.process_time()-start,
                             "true_FIT_loss":float(loss.detach())})
    gap=lambda field:(max(v[field] for v in measurements)-
                      min(v[field] for v in measurements))/max(v[field] for v in measurements)
    receipt={"study":"ME056_ACTUAL_ORIGINAL_ROBOT_PREFIT_NO_CLAIM",
             "source": "BodenAI physically logged original 14D robot actions and state",
             "real_source_FIT_episode_count":len(actual_fit),
             "genuine_FIT_sample_schedule_sha256":
                 base.base.sha_bytes(json.dumps(real_schedule,separators=(",",":")).encode()),
             "seeds":list(SEEDS),"same_sample_per_arm":True,
             "same_600_updates_batch2_burnin256_real_BPTT16":True,
             "cpu_time_must_match":False,"arms":measurements,
             "parameter_gap":gap("parameters"),
             "profiled_FLOP_gap":gap("one_genuine_optimizer_update_profiled_operator_flops"),
             "profiler_limitations":"PyTorch with_flops counts reported CPU ops; not full exact training FLOPs",
             "no_model_training_begun":True}
    # This is an actual real-FIT prefit, but a profiler is not proof of all
    # unreported nonlinear kernels. Treat prefit equality conservatively.
    receipt["parameter_gate_pass"]=receipt["parameter_gap"]<=.01
    receipt["profiled_operator_gate_pass"]=receipt["profiled_FLOP_gap"]<=.01
    Path(folder).mkdir(parents=True,exist_ok=True)
    base.base.write_json(Path(folder)/"ME056_REAL_PREFIT.json",receipt)
    print(json.dumps({"event":"ME056_REAL_ORIGINAL_FIT_PREFIT",
                      "parameters":{v["arm"]:v["parameters"] for v in measurements},
                      "parameter_gap":receipt["parameter_gap"],
                      "partial_operator_FLOP_gap":receipt["profiled_FLOP_gap"],
                      "CPU_TIMING_IS_NOT_A_PARITY_GATE":True}),flush=True)
    if not (receipt["parameter_gate_pass"] and receipt["profiled_operator_gate_pass"]):
        raise RuntimeError("ME056_BLOCKED_BEFORE_FIT: actual genuine original FIT preflight "
                           "violates mandatory parameter/compute parity")
    return receipt


def run(output):
    m=base
    m.network=network
    m.REGISTRATION=REGISTRATION
    m.SEEDS=SEEDS
    m.ARMS=ARMS
    m.base.UPDATES=600
    m.base.BATCH=2
    m.base.HISTORY=256
    m.base.BACKPROP=16
    m.base.NEXT_FRAMES=272

    expected=json.loads(REGISTRATION.read_text())
    nominal=nearest_real_gru_width([.03]*14)
    if nominal["gru_width"]!=44 or nominal["relative_gap"]>.01:
        raise RuntimeError("capacity survey no longer matches pre-registered full GRU")
    actual={name:count_active(network(name,[.03]*14)) for name in ARMS}
    if actual!=expected["parameter_counts"]:
        raise RuntimeError(f"architecture weights changed against frozen prefit: {actual}")
    def verified_prefit(real_fit,groups,plan):
        audited=_original_source_preflight(real_fit,groups,plan)
        norm=m.base.fit_scaling(real_fit)
        scale,_=m.FIT_only_original_delta_scales(real_fit,norm)
        profile_real_matched_step(real_fit,norm,scale,Path(output))
        return audited
    m.verify_real_source_before_any_optimizer=verified_prefit
    print(json.dumps({"event":"ME056_ORIGINAL_PHYSICAL_CONCEPT_WORLD_MODEL_FIT_REQUEST",
        "study":"ME056","source":"authentic real FIT episodes only",
        "model_parameters":actual,"no_CPU_time_matching":True,
        "test_set_previously_exposed":True}),flush=True)
    # Inherited fit checks actual source hashes, archives trained real model
    # checkpoints before scoring, then writes original developmental outcomes.
    return m.run(output)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",default="results/me056")
    args=p.parse_args()
    run(args.output)
