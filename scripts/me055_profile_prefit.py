#!/usr/bin/env python3
"""Profile one authentic ME055 update before any comparative model fitting.

Uses a pinned FIT episode selected without outcome inspection. Reports, but
does not alter, genuine TAC/GRU FLOPs and CPU. A mismatch blocks training; this
script never pads time, inserts dummy operators or changes model capacity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile
from torch._higher_order_ops import scan

from tac_osm.real_robot_me055_rollout import (
    CompactBelief, ServoRolloutGRU, ServoRolloutTAC, factual_multistep_loss)

torch.set_num_threads(1)
HISTORY, HORIZON = 256, 16


def sha_file(path):
    h = hashlib.sha256()
    with open(path,"rb") as stream:
        for block in iter(lambda: stream.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def load_authentic(path):
    import pyarrow.parquet as pq
    table = pq.read_table(path)
    state = np.asarray(table["observations.state.qpos"].to_pylist(),dtype=np.float32)
    action = np.asarray(table["action.qpos"].to_pylist(),dtype=np.float32)
    human = np.asarray(table["intervention"].to_pylist(),dtype=np.bool_)
    if state.shape != action.shape or state.ndim != 2 or state.shape[1] != 14:
        raise ValueError("authentic 14D state/action episode required")
    if len(state) < HISTORY+HORIZON+2:
        raise ValueError("authentic episode too short for registered profile")
    return state,action,human


def physical_identification(state,action):
    state_mean=state.mean(0,dtype=np.float64)
    state_scale=np.maximum(state.std(0,dtype=np.float64),.05)
    action_mean=action.mean(0,dtype=np.float64)
    action_scale=np.maximum(action.std(0,dtype=np.float64),.05)
    x=(state-state_mean)/state_scale
    a=(action-action_mean)/action_scale
    action_in_state=(a*action_scale+action_mean-state_mean)/state_scale
    delta=x[1:]-x[:-1]
    error=action_in_state[:-1]-x[:-1]
    gain=np.clip((error*delta).sum(0)/(np.square(error).sum(0)+1e-8),0,.25)
    delta_scale=np.maximum(np.abs(delta).mean(0),.0005)
    step_limit=np.maximum(np.quantile(np.abs(delta),.99,axis=0),delta_scale)
    return {"delta_scale":delta_scale.astype(np.float32),
            "state_mean":state_mean.astype(np.float32),
            "state_scale":state_scale.astype(np.float32),
            "action_mean":action_mean.astype(np.float32),
            "action_scale":action_scale.astype(np.float32),
            "servo_gain":gain.astype(np.float32),
            "servo_step_limit":step_limit.astype(np.float32)}


def authentic_batch(state,action,human,identified,batch_size):
    # Selection uses only current authority and required length, never future
    # prediction errors. Two fixed windows make the profiler source-shaped.
    eligible=[start for start in range(len(state)-(HISTORY+HORIZON+1))
              if not human[start+HISTORY]]
    if len(eligible)<batch_size:
        raise ValueError("too few authentic robot-controlled rollout origins")
    positions=np.linspace(0,len(eligible)-1,num=batch_size,dtype=np.int64)
    starts=tuple(eligible[int(i)] for i in positions)
    sm,ss=identified["state_mean"],identified["state_scale"]
    am,ast=identified["action_mean"],identified["action_scale"]
    states=torch.from_numpy(np.stack([(state[s:s+HISTORY+HORIZON+1]-sm)/ss
                                      for s in starts]).astype(np.float32))
    actions=torch.from_numpy(np.stack([(action[s:s+HISTORY+HORIZON]-am)/ast
                                       for s in starts]).astype(np.float32))
    authority=torch.from_numpy(np.stack([human[s:s+HISTORY+HORIZON+1]
                                         for s in starts])).unsqueeze(-1)
    return states,actions,authority,starts


def build(architecture,identified):
    cls=ServoRolloutTAC if architecture=="native_tac" else ServoRolloutGRU
    return cls(**{name:value.tolist() for name,value in identified.items()})


def scanned_burnin(model,belief,states,actions,human):
    """Tensor-only recurrent scan; avoids one Python dispatch per history step."""
    xs=(states[:,:HISTORY].transpose(0,1),
        actions[:,:HISTORY].transpose(0,1),
        human[:,:HISTORY].transpose(0,1),
        states[:,1:HISTORY+1].transpose(0,1),
        human[:,1:HISTORY+1].transpose(0,1))
    if isinstance(belief,CompactBelief):
        initial=tuple(value.clone() for value in belief)
        def body(carry,item):
            current,action,authority,next_state,next_authority=item
            result=model._burnin_training(
                CompactBelief(*carry),current,action,authority,
                next_state,next_authority)
            next_carry=tuple(value.clone() for value in result)
            return next_carry,result.fast.sum().clone()
        final,_=scan(body,initial,xs)
        return CompactBelief(*final)
    def body(carry,item):
        current,action,authority,next_state,next_authority=item
        result=model._burnin_training(
            carry,current,action,authority,next_state,next_authority)
        return result.clone(),result.sum().clone()
    final,_=scan(body,belief.clone(),xs)
    return final


def objective(model,batch,kernels=None,use_scan=False):
    states,actions,human,_=batch
    burnin,observe,predict=(
        (model._burnin_training,model._observe_training,model._predict_training)
        if kernels is None else kernels)
    belief=(model.initial_from_observation(states[:,0])
            if hasattr(model,"initial_from_observation") else model.initial(len(states)))
    with torch.no_grad():
        if use_scan:
            belief=scanned_burnin(model,belief,states,actions,human)
        else:
            for step in range(HISTORY):
                belief=burnin(belief,states[:,step],actions[:,step],
                               human[:,step],states[:,step+1],human[:,step+1])
    origin=states[:,HISTORY]
    control=human[:,HISTORY]
    if control.any():
        raise ValueError("profiler rollout origin must be robot controlled")
    future_human=human[:,HISTORY:HISTORY+HORIZON+1]
    current_human=future_human[:,:-1]
    alive=(~current_human).to(torch.int64).cumprod(1).bool()
    onset=(~current_human & future_human[:,1:]).to(origin.dtype)
    rollout_states=[];rollout_logits=[];current=origin
    for step in range(HORIZON):
        belief=observe(belief,current)
        one=predict(belief,current,
                    actions[:,HISTORY+step:HISTORY+step+1],control)
        current=one.next_joint_state[:,0]
        rollout_states.append(current);rollout_logits.append(one.takeover_logits[:,0])
    from tac_osm.real_robot_me055_rollout import TrajectoryForecast
    forecast=TrajectoryForecast(torch.stack(rollout_logits,1),
                                torch.stack(rollout_states,1))
    result,_=factual_multistep_loss(
        model,forecast,states[:,HISTORY+1:HISTORY+HORIZON+1],
        alive,onset,alive)
    return result


def update(model,batch,optimizer,loss_function=None,use_scan=False):
    optimizer.zero_grad(set_to_none=True)
    result=objective(model,batch,use_scan=use_scan) if loss_function is None else loss_function()
    result.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    optimizer.step()
    return float(result.detach())


def one_profile(architecture,identified,batch,repeats,compile_forward,script_kernels,
                compile_mode,use_scan):
    torch.manual_seed(55001)
    model=build(architecture,identified)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    compile_cpu_seconds=0.
    loss_function=None
    if compile_forward and script_kernels:
        raise ValueError("choose one optimized execution path")
    if script_kernels:
        start=time.process_time()
        scripted=torch.jit.script(model)
        if next(scripted.parameters()).data_ptr()!=next(model.parameters()).data_ptr():
            raise RuntimeError("TorchScript training parameters do not share storage")
        kernels=(scripted._burnin_training,scripted._observe_training,
                 scripted._predict_training)
        loss_function=lambda: objective(model,batch,kernels,use_scan)
        update(model,batch,optimizer,loss_function)
        compile_cpu_seconds=time.process_time()-start
    elif compile_forward:
        kernels=tuple(torch.compile(method,dynamic=False,fullgraph=True,
                                    mode=compile_mode)
                      for method in (model._burnin_training,
                                     model._observe_training,
                                     model._predict_training))
        loss_function=lambda: objective(model,batch,kernels,use_scan)
        start=time.process_time()
        update(model,batch,optimizer,loss_function)
        compile_cpu_seconds=time.process_time()-start
    else:
        update(model,batch,optimizer,use_scan=use_scan)
    durations=[]
    for _ in range(repeats):
        start=time.process_time()
        update(model,batch,optimizer,loss_function,use_scan)
        durations.append(time.process_time()-start)
    # FLOPs are recorded from an eager, structurally identical update because
    # compiler fusion removes profiler-visible operator accounting.
    torch.manual_seed(55001)
    eager=build(architecture,identified)
    eager_optimizer=torch.optim.AdamW(eager.parameters(),lr=.001,weight_decay=.0001)
    with profile(activities=[ProfilerActivity.CPU],with_flops=True) as recording:
        loss=update(eager,batch,eager_optimizer,use_scan=False)
    flops=sum(event.flops for event in recording.key_averages())
    if flops<=0:
        raise RuntimeError("PyTorch profiler emitted no operator FLOPs")
    return {"architecture":architecture,
            "trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),
            "deployed_parameters":sum(p.numel() for p in model.parameters()),
            "profiled_one_update_flops":int(flops),
            "median_one_update_cpu_seconds":statistics.median(durations),
            "cpu_seconds_samples":durations,"last_profile_objective":loss,
            "compiled_forward":compile_forward,
            "scanned_burnin":use_scan,
            "torchscript_kernels":script_kernels,
            "first_compiled_update_cpu_seconds":compile_cpu_seconds,
            "profiler":"torch.profiler CPU with_flops=True; one full authentic burn-in+rollout+backward+AdamW update"}


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--fit-parquet",required=True)
    cli.add_argument("--expected-sha256",required=True)
    cli.add_argument("--output",required=True)
    cli.add_argument("--repeats",type=int,default=7)
    cli.add_argument("--batch-size",type=int,default=2)
    cli.add_argument("--compile-forward",action="store_true")
    cli.add_argument("--script-kernels",action="store_true")
    cli.add_argument("--scan-burnin",action="store_true")
    cli.add_argument("--compile-mode",default="reduce-overhead",
                     choices=("default","reduce-overhead","max-autotune"))
    args=cli.parse_args()
    if sha_file(args.fit_parquet)!=args.expected_sha256:
        raise SystemExit("authentic FIT profiler episode SHA mismatch")
    state,action,human=load_authentic(args.fit_parquet)
    identified=physical_identification(state,action)
    if args.batch_size < 2:
        raise SystemExit("batch size must be at least two")
    batch=authentic_batch(state,action,human,identified,args.batch_size)
    arms=[one_profile(name,identified,batch,args.repeats,args.compile_forward,
                      args.script_kernels,args.compile_mode,args.scan_burnin)
          for name in ("native_tac","gru_world")]
    def gap(key):
        values=[row[key] for row in arms]
        return (max(values)-min(values))/max(values)
    receipt={"study":"ME055_PREFIT_AUTHENTIC_RESOURCE_PROFILE",
        "source_sha256":args.expected_sha256,
        "source_episode":12,"source_split":"FIT",
        "batch_start_frames":list(batch[3]),"history_frames":HISTORY,
        "rollout_frames":HORIZON,"batch_size":args.batch_size,"arms":arms,
        "parameter_relative_gap":gap("trainable_parameters"),
        "profiled_FLOP_relative_gap":gap("profiled_one_update_flops"),
        "measured_CPU_relative_gap":gap("median_one_update_cpu_seconds"),
        "strict_parameter_gate_pass":gap("trainable_parameters")<=.01,
        "strict_FLOP_gate_pass":gap("profiled_one_update_flops")<=.01,
        "strict_CPU_gate_pass":gap("median_one_update_cpu_seconds")<=.05,
        "model_fits_started":0}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n")
    print(json.dumps(receipt,sort_keys=True))
    return 0 if all(receipt[k] for k in ("strict_parameter_gate_pass",
                 "strict_FLOP_gate_pass","strict_CPU_gate_pass")) else 2


if __name__=="__main__":
    raise SystemExit(main())
