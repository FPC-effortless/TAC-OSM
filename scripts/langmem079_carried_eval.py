#!/usr/bin/env python3
"""LANGMEM079: original genuine publisher dual-native carried memory test.

Fitted 078 model checked by original immutable 3-seed FIT source receipt.
Every observed source byte advances original true 10-field persistent state.
Only original already-exposed publisher DEV is scored; protected TEST sealed.
"""
from __future__ import annotations
import argparse,hashlib,json,math,sys,time
from dataclasses import replace
from pathlib import Path
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
for n in ("langmem079","langmem078","langscale075","langscale070",
          "context026","langmemory042","langbase024","langstream076"):
    p=str(ROOT/"research/mathematical-engineering"/n)
    if p not in sys.path:sys.path.insert(0,p)
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from dual_native import DualNativeTimescaleTAC078
from dual_streaming import DualNativeStreamingTAC079
from byte_next_reference import original_source
from scripts.langlayer035_sampling import fit_sources
from scripts.langmem078_cpu_fit import model_source_preflight
from scripts.langstream076_carried_source_eval import score

SEEDS=(1701,1702,1703)
SCORED=tuple(range(128,320,3))
WIDTH=8
STATE_BYTES=4160

def restore_original(seed,ckpt_dir:Path,source_verified:Path):
    if seed not in SEEDS:raise ValueError("unregistered trained seed")
    summary=json.loads(source_verified.read_text())
    if (summary.get("study")!="LANGMEM078"
        or summary.get("new_original_checkpoint_count")!=3
        or summary.get("source_paired") is not True
        or summary.get("protected_TEST")!="NOT_ACCESSED"):
        raise ValueError("original 078 trained source verification not accepted")
    rows=summary["metrics"]["dual_fast_slow"]["original_seed_records"]
    r=next((x for x in rows if x["seed"]==seed),None)
    if r is None:raise ValueError("trained model original seed absent")
    paths=list(ckpt_dir.rglob("*.pt"))
    if len(paths)!=1:raise ValueError("exactly one original trained dual checkpoint needed")
    cp=paths[0]
    if hashlib.sha256(cp.read_bytes()).hexdigest()!=r["checkpoint_sha256"]:
        raise ValueError("original 078 trained checkpoint SHA differs from source verified result")
    original=torch.load(cp,map_location="cpu",weights_only=True)
    if (original.get("study"),original.get("arm"),original.get("seed"),
        original.get("updates"),original.get("source_commit"))!=(
        "LANGMEM078","dual",seed,160,"60403a5409ff2c3f3b07dd2ca91a7a3e096839c7"):
        raise ValueError("original fitted publisher checkpoint identity mismatch")
    if original.get("sample_order_sha256")!=r["source_sample_order_sha256"]:
        raise ValueError("original publisher source exposure SHA differs")
    if original.get("parameters")!={"trainable":499627,"deployed":499627}:
        raise ValueError("original learned parameter count not source qualified")
    if not all(key in original for key in ("model","optimizer","rng_torch","rng_numpy","rng_python")):
        raise ValueError("original optimizer/RNG checkpoint receipt missing")
    model_source_preflight()
    m=DualNativeTimescaleTAC078().eval()
    m.load_state_dict(original["model"],strict=True)
    return m,r


@torch.no_grad()
def evaluate(seed,ckpt_dir,dual_verified,original_fast_stream_summary):
    torch.set_num_threads(2)
    fit=fit_sources()
    _,dev,_=original_source(ROOT)
    if len(fit)!=14 or len(dev)!=4 or set(fit)&set(dev):
        raise ValueError("original PSF 14 FIT/four DEV source owners not preserved")
    baseline=json.loads(original_fast_stream_summary.read_text())
    if baseline.get("study")!="LANGSTREAM076" or baseline.get("model_seed_results")!=9:
        raise ValueError("unverified original m8 true streaming source reference")
    model,r=restore_original(seed,ckpt_dir,dual_verified)
    engine=DualNativeStreamingTAC079(model,"cap_all")
    rho=float(torch.exp(-F.softplus(model.slow_log_decay)).item())
    beta=float((.05*torch.sigmoid(model.slow_erase_logit)).item())
    write=float((.01*torch.sigmoid(model.slow_write_logit)).item())
    mix=float(torch.sigmoid(model.slow_read_logit).item())
    if not (0<rho<1 and 0<beta<.05 and 0<write<.01 and 0<mix<1):
        raise ValueError("trained slow rank-one gates violate source-bounded contract")
    target_hash=hashlib.sha256()
    started=time.process_time()
    rows=[]
    for owner in sorted(dev):
        raw=dev[owner]
        if len(raw)<321:raise ValueError("registered genuine original source byte stream too short")
        prefix=torch.tensor(list(raw[:128]),dtype=torch.long).view(1,128)
        native=model(prefix)[:,-1]
        logits,main=engine.prefill(prefix,owners=(owner,))
        if (logits-native).abs().max().item()>3e-5:
            raise ValueError("new dual streaming prefill fails original full graph")
        if main.clock!=128 or main.tensor_bytes()!=STATE_BYTES:
            raise ValueError("new 10-field native source state incorrect")
        origin_slow_norm=float(torch.linalg.matrix_norm(main.tensors["slow_native"].double()).item())
        if origin_slow_norm<=1e-11 or not math.isfinite(origin_slow_norm):
            raise ValueError("original trained slow source matrix empty at t128")
        slow_cleared=replace(main,tensors={**main.tensors,
             "slow_native":torch.zeros_like(main.tensors["slow_native"])})
        fast_cleared=replace(main,tensors={**main.tensors,
             "native":torch.zeros_like(main.tensors["native"])})
        for t in range(128,320):
            if not (main.clock==slow_cleared.clock==fast_cleared.clock==t):
                raise ValueError("stream clock mismatch/source owner cross")
            observed=torch.tensor([raw[t]],dtype=torch.long)
            full,full_next=engine.step(observed,main,owners=(owner,))
            slow,slow_next=engine.step(observed,slow_cleared,owners=(owner,))
            fast,fast_next=engine.step(observed,fast_cleared,owners=(owner,))
            slow_delta=float(torch.linalg.matrix_norm((
                full_next.tensors["slow_native"]-slow_next.tensors["slow_native"]).double()).item())
            ratio=slow_delta/origin_slow_norm
            upper=rho**(t-128+1)
            if ratio>upper+1e-4:
                raise ValueError(f"true original slow retained-matrix perturbation violates exponential rho bound at t={t}")
            if t in SCORED:
                at_slow=replace(main,tensors={**main.tensors,
                    "slow_native":torch.zeros_like(main.tensors["slow_native"])})
                at_fast=replace(main,tensors={**main.tensors,
                    "native":torch.zeros_like(main.tensors["native"])})
                one_slow,_=engine.step(observed,at_slow,owners=(owner,))
                one_fast,_=engine.step(observed,at_fast,owners=(owner,))
                y=int(raw[t+1]) # strictly AFTER same-source original byte t was processed
                target_hash.update(owner.encode()+b":"+str(t).encode()+b":"+bytes([y])+b";")
                rows.append({
                    "owner":owner,"clock":t,"next_byte_real_source":y,
                    "age_after_128":t-128+1,
                    "slow_initial_128_norm":origin_slow_norm,
                    "slow_old_state_delta_ratio":ratio,
                    "slow_rho_product_upper_bound":upper,
                    "native_slow_read_mix":mix,
                    "full":score(full,y),
                    "one_step_slow_zero":score(one_slow,y),
                    "one_step_fast_zero":score(one_fast,y),
                    "slow_wipe_t128_carried":score(slow,y),
                    "fast_wipe_t128_carried":score(fast,y),
                    "one_step_slow_logit_max_abs":float((one_slow-full).abs().max()),
                    "one_step_fast_logit_max_abs":float((one_fast-full).abs().max())
                })
            main,slow_cleared,fast_cleared=full_next,slow_next,fast_next
        if not (main.clock==slow_cleared.clock==fast_cleared.clock==320):
            raise ValueError("real source output stream incomplete")
    if len(rows)!=256:raise ValueError("source label count not 256")
    sha=target_hash.hexdigest()
    if sha!=baseline.get("original_targets_sha256"):
        raise ValueError("original real publisher owner/clock/target bytes not same as original 076 baseline")
    arms=("full","one_step_slow_zero","one_step_fast_zero",
          "slow_wipe_t128_carried","fast_wipe_t128_carried")
    metrics={arm:{field:sum(x[arm][field] for x in rows)/len(rows)
                  for field in ("nll","brier","top1")} for arm in arms}
    base=baseline["metrics"]["m8"]["by_seed"][str(seed)]
    if not math.isfinite(base["full_nll"]):
        raise ValueError("not original source m8 baseline NLL")
    return {"study":"LANGMEM079","seed":seed,
        "status":"REAL_PUBLISHER_CARRIED_DUAL_FAST_SLOW_CAUSAL_READ_RETENTION_DIAGNOSTIC",
        "original_dual_checkpoint_sha256":r["checkpoint_sha256"],
        "original_fitted_sample_order_sha256":r["source_sample_order_sha256"],
        "same_original_fast_source_baseline_full_nll":base["full_nll"],
        "original_baseline_run_id":40883211351,
        "trainable_parameters":499627,"fp32_state_bytes_per_owner":STATE_BYTES,
        "real_source_owner_count":4,"real_source_scored_targets":256,
        "original_source_target_sha256":sha,
        "actual_scored_clocks":SCORED,
        "carried_last_clock":320,
        "trained_context_length":128,
        "source_position_mode":"cap_all_POST_EXPOSED_DEV",
        "learned_gates":{"rho_s":rho,"erase_beta_s":beta,"write_epsilon_s":write,
                         "read_mix_s":mix,"source_bound_half_life_bytes":math.log(.5)/math.log(rho)},
        "metrics":metrics,
        "slow_old_state_delta_ratio_16":sum(x["slow_old_state_delta_ratio"] for x in rows if x["age_after_128"]==16)/4,
        "slow_old_state_retention_upper_bound_16":rho**16,
        "source_rows":rows,
        "measured_cpu_process_seconds":time.process_time()-started,
        "no_new_fits":True,"gpu_minutes":0,"protected_TEST":"NOT_ACCESSED",
        "independently_qualified":False,
        "limits":["original genuine licensed source DEV previously exposed (not independent)",
            "cap_all position policy chosen after observed 073 DEV failure",
            "new slow lane was trained on reset128 windows, this true stream is nofit extrapolation",
            "native one-step lesion leaves current source write active, not exact fact-recall",
            "slow clear only at clock128 may propagate through other upper/concept memories",
            "training FLOPs not matched to original 070, no architecture superiority",
            "source retention math does not prove useful old-source content recovery"]}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,choices=SEEDS,required=True)
    p.add_argument("--original-checkpoint-dir",type=Path,required=True)
    p.add_argument("--dual-verified",type=Path,required=True)
    p.add_argument("--original-stream-baseline",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError("original evidence overwrite denied")
    out=evaluate(a.seed,a.original_checkpoint_dir,a.dual_verified,a.original_stream_baseline)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print("LANGMEM079_REAL_STREAM "+json.dumps({
       "seed":a.seed,"gates":out["learned_gates"],"metrics":out["metrics"],
       "original_fast_only_source_nll":out["same_original_fast_source_baseline_full_nll"],
       "slow_old_state_delta_ratio_16":out["slow_old_state_delta_ratio_16"],
       "state_bytes":out["fp32_state_bytes_per_owner"]},sort_keys=True))
if __name__=="__main__":main()
