#!/usr/bin/env python3
"""Diagnostic ONLY: deploy frozen ME099 ALOHA physical predictor in MuJoCo proxies.

Two-degree-of-freedom arm and a simplified yaw/thrust planar-vehicle proxy.
THIS IS NOT a trained action policy, valid car adapter, tire simulation, or
evidence of real-world robotic/automotive control. No training or protected TEST.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tac_osm.variants.regime_trajectory_world import RegimeTrajectoryWorld, marginal_intervals

ARM_XML = """
<mujoco model="diagnostic_two_joint_arm">
  <compiler angle="radian"/>
  <option timestep=".02" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="link1" pos="0 0 .25">
      <joint name="joint1" type="hinge" axis="0 1 0" damping=".4" limited="true" range="-1.3 1.3"/>
      <geom type="capsule" fromto="0 0 0 0 0 .45" size=".04" mass="1"/>
      <body name="link2" pos="0 0 .45">
        <joint name="joint2" type="hinge" axis="0 1 0" damping=".3" limited="true" range="-1.4 1.4"/>
        <geom type="capsule" fromto="0 0 0 0 0 .35" size=".035" mass=".6"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="motor1" joint="joint1" gear="1" ctrllimited="true" ctrlrange="-1 1"/>
    <motor name="motor2" joint="joint2" gear="1" ctrllimited="true" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""

# Force-actuated planar yaw/thrust proxy. NOT Ackermann steering, tires,
# wheel-ground contact or real vehicle dynamics.
CAR_XML = """
<mujoco model="diagnostic_yaw_thrust_vehicle_proxy">
  <option timestep=".02" gravity="0 0 0"/>
  <worldbody>
    <body name="vehicle" pos="0 0 0">
      <joint name="x" type="slide" axis="1 0 0" damping=".35"/>
      <joint name="y" type="slide" axis="0 1 0" damping=".35"/>
      <joint name="yaw" type="hinge" axis="0 0 1" damping=".5"/>
      <geom type="box" size=".22 .11 .08" mass="3"/>
    </body>
  </worldbody>
</mujoco>
"""

def verify_checkpoint(seed: int):
    f = ROOT / f"research/replay-bundles/ME099/seed-{seed}/final.pt"
    index = json.loads((ROOT / "research/replay-bundles/ME099/FIT_INDEX.json").read_text())
    receipt = next(x for x in index["models"] if x["seed"] == seed)
    digest = hashlib.sha256(f.read_bytes()).hexdigest()
    if digest != receipt["checkpoint_sha256"]:
        raise RuntimeError(f"FROZEN_CHECKPOINT_SHA_MISMATCH {seed}")
    # Trusted first-party original committed model bundle. Never unpickle untrusted files.
    ck = torch.load(f, map_location="cpu", weights_only=False)
    model = RegimeTrajectoryWorld()
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    p = ck["preprocessing"]
    def vec(k): return np.asarray(p[k], dtype=np.float32).copy()
    prep = {k: vec(k) for k in ("obs_mean","obs_scale","action_mean","action_scale")}
    assert all(z.shape == (14,) for z in prep.values())
    count = sum(q.numel() for q in model.parameters())
    assert count == receipt["trainable_parameters"] == 103380
    return model, prep, receipt, digest

def wrap(x):
    return (x + math.pi) % (2 * math.pi) - math.pi

def create(task):
    m = mujoco.MjModel.from_xml_string(ARM_XML if task == "arm" else CAR_XML)
    d = mujoco.MjData(m)
    if task == "arm":
        d.qpos[:2] = [-.25, .4]
    mujoco.mj_forward(m, d)
    return m,d

def phys(task, d):
    if task=="arm":
        return np.array([d.qpos[0],d.qpos[1],d.qvel[0],d.qvel[1]],dtype=np.float64)
    return np.array([d.qpos[0],d.qpos[1],d.qpos[2], math.hypot(d.qvel[0],d.qvel[1])],dtype=np.float64)

def observation(prep, task, p):
    raw=prep["obs_mean"].copy()
    raw[:4]=p.astype(np.float32)
    return raw

def past_command(prep, command):
    a=prep["action_mean"].copy()
    a[:len(command)]=np.asarray(command,dtype=np.float32)
    return a

def controller(task, sensed, goal):
    if task=="arm":
        pos,vel=sensed[:2],sensed[2:4]
        return np.clip(7.5*(goal-pos)-1.9*vel,-1,1)
    x,y,yaw,speed=sensed
    dx,dy=goal[0]-x,goal[1]-y
    theta=math.atan2(dy,dx)
    desired_speed=min(.65,math.hypot(dx,dy)*.6)
    return np.array([np.clip(2.2*(desired_speed-speed),-1,1),
                     np.clip(2.5*wrap(theta-yaw),-1,1)],dtype=np.float64)

def physics_step(task,m,d,ctrl):
    if task=="arm":
        d.ctrl[:]=ctrl
    else:
        d.xfrc_applied[:]=0
        yaw=float(d.qpos[2])
        bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"vehicle")
        d.xfrc_applied[bid,:]=[float(ctrl[0])*math.cos(yaw)*3,
                               float(ctrl[0])*math.sin(yaw)*3,0,0,0,float(ctrl[1])*.6]
    mujoco.mj_step(m,d)

@torch.inference_mode()
def episode(model,prep,task,controller_kind,sensor_gap,steps):
    m,d=create(task)
    target=np.array([.65,-.3] if task=="arm" else [2.,.4],dtype=np.float64)
    observed=phys(task,d)
    prev_action=past_command(prep,np.zeros(2))
    state=None
    errors=[];covered=[];widths=[];tracking=[];used=0;fallback=0;nan_predictions=0
    t0=time.perf_counter()
    gap_start=60;gap_end=110  # exactly 50 consecutive physics steps
    for t in range(steps):
        true=phys(task,d)
        drop=sensor_gap and gap_start<=t<gap_end
        if not drop:
            observed=true.copy()
        raw=observation(prep,task,observed)
        obs=(raw-prep["obs_mean"])/prep["obs_scale"]
        action=(prev_action-prep["action_mean"])/prep["action_scale"]
        ot=torch.from_numpy(obs).reshape(1,1,14)
        at=torch.from_numpy(action).reshape(1,1,14)
        # Source-compliant: no new observation is committed during blackout;
        # invalid input explicitly cannot update the original associative state.
        valid=torch.tensor([[not drop]],dtype=torch.bool)
        pred,state=model(ot,at,horizons=(1,),valid=valid,state=state)
        lo,hi=marginal_intervals(pred)
        pred_world=pred.mean[0,0,0].cpu().numpy()*prep["obs_scale"]+prep["obs_mean"]
        lower=lo[0,0,0].cpu().numpy()*prep["obs_scale"]+prep["obs_mean"]
        upper=hi[0,0,0].cpu().numpy()*prep["obs_scale"]+prep["obs_mean"]
        if not (np.isfinite(pred_world).all() and np.isfinite(lower).all() and np.isfinite(upper).all()):
            nan_predictions+=1
            raise RuntimeError("nonfinite pretrained forecast; no silent replacement")
        forecast=pred_world[:4].astype(np.float64)
        # Do not turn out-of-distribution forecasts into unrestricted actuation.
        # This small bounded advisory channel is intentionally diagnostic.
        control_obs=observed.copy()
        if controller_kind=="assisted" and not drop:
            deviation=np.abs(forecast-observed)
            if np.all(deviation[:2] < .30) and np.all(deviation[2:] < .60):
                control_obs=.75*observed+.25*forecast
                used+=1
            else:
                fallback+=1
        action2=controller(task,control_obs,target)
        physics_step(task,m,d,action2)
        nxt=phys(task,d)
        prev_action=past_command(prep,action2)
        forecast_error=np.abs(forecast[:2]-nxt[:2]).mean()
        persistence_error=np.abs(observed[:2]-nxt[:2]).mean()
        errors.append([float(forecast_error),float(persistence_error),int(drop)])
        covered.append(float(np.mean((lower[:2]<=nxt[:2])&(nxt[:2]<=upper[:2]))))
        widths.append(float(np.mean(upper[:2]-lower[:2])))
        tracking.append(float(np.linalg.norm((nxt[:2]-target) if task=="arm" else (nxt[:2]-target))))
    elapsed=time.perf_counter()-t0
    z=np.asarray(errors,dtype=np.float64)
    masked=z[:,2]==1;ok=~masked
    def avg(a):return float(np.mean(a)) if len(a) else None
    return dict(task=task,controller=controller_kind,blackout=bool(sensor_gap),
        physics_steps=steps,blackout_steps=int(masked.sum()),params=103380,
        native_primary_float_state_bytes=608,
        diagnostics=dict(mae_1step_model_all=avg(z[:,0]),mae_1step_persistence_all=avg(z[:,1]),
          mae_model_gap=avg(z[masked,0]),mae_persistence_gap=avg(z[masked,1]),
          interval_coverage_gap=avg(np.asarray(covered)[masked]),
          interval_coverage_visible=avg(np.asarray(covered)[ok]),
          mean_interval_width_gap=avg(np.asarray(widths)[masked]),
          mean_interval_width_visible=avg(np.asarray(widths)[ok])),
        control=dict(final_target_distance=tracking[-1],mean_target_distance=avg(tracking),
          neural_assisted_steps=used,trust_fallback_steps=fallback),
        measured=dict(total_cpu_wall_seconds=elapsed,forward_plus_physics_seconds_per_step=elapsed/steps))
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,default=99003)
    p.add_argument("--steps",type=int,default=160)
    p.add_argument("--output",default="research/diagnostics/me099_mujoco_results.json")
    a=p.parse_args()
    assert a.steps >= 111
    torch.set_num_threads(1)
    model,prep,receipt,digest=verify_checkpoint(a.seed)
    rows=[]
    for task in ("arm","car"):
        for gap in (False,True):
            for control in ("baseline","assisted"):
                row=episode(model,prep,task,control,gap,a.steps)
                rows.append(row)
                print("DIAGNOSTIC",json.dumps(row,sort_keys=True),flush=True)
    out={"status":"SIMULATION_DIAGNOSTIC_ONLY_NOT_REAL_QUALIFICATION",
         "checkpoint_family":"ME099","seed":a.seed,"sha256":digest,
         "checkpoint_source":"ALOHA mobile cabinet 14D joint state + lagged action",
         "warning":"Never trained to drive; current action candidates cannot be scored by this model. Car is a planar force/yaw proxy without wheels, tire force, pedestrian sensing or safety proof. The 50-step blackout is scripted. Simulator results cannot qualify real-world intelligence.",
         "blackout_range":[60,110],"rows":rows}
    path=ROOT/a.output
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(out,indent=2)+"\n")
    assert len(rows)==8
    assert all(r["blackout_steps"]==(50 if r["blackout"] else 0) for r in rows)
    print("CHECKPOINT_SHA256="+digest)
    print("SIMULATOR_DIAGNOSTIC_COMPLETED=PASS")
    print("REAL_ROBOT_OR_CAR_CONTROL_QUALIFIED=NO")
if __name__=="__main__":main()
