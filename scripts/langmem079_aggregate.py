#!/usr/bin/env python3
"""LANGMEM079 genuine observed source 3-seed dual slow vs original 076 baseline."""
import argparse,json,math,statistics
from pathlib import Path
SEEDS=(1701,1702,1703)
ARMS=("full","one_step_slow_zero","one_step_fast_zero",
      "slow_wipe_t128_carried","fast_wipe_t128_carried")
CLOCKS=tuple(range(128,320,3))

def aggregate(folder:Path):
    files=list(folder.rglob("STREAM_RESULTS.json"))
    if len(files)!=3:raise ValueError("requires exactly three original fitted dual source reports")
    runs={}
    for p in files:
        r=json.loads(p.read_text())
        seed=r.get("seed")
        if seed not in SEEDS or seed in runs:
            raise ValueError("unexpected or repeated originally trained seed")
        if (r.get("study")!="LANGMEM079" or r.get("no_new_fits") is not True
            or r.get("gpu_minutes")!=0 or r.get("protected_TEST")!="NOT_ACCESSED"
            or r.get("trainable_parameters")!=499627
            or r.get("fp32_state_bytes_per_owner")!=4160
            or r.get("real_source_owner_count")!=4 or r.get("real_source_scored_targets")!=256
            or tuple(r.get("actual_scored_clocks",[]))!=CLOCKS
            or r.get("carried_last_clock")!=320 or r.get("trained_context_length")!=128
            or r.get("source_position_mode")!="cap_all_POST_EXPOSED_DEV"
            or r.get("original_baseline_run_id")!=40883211351):
            raise ValueError("new dual true-source state/clock/no-fit receipt mismatch")
        if set(r.get("metrics",{}))!=set(ARMS):
            raise ValueError("a source-native lesion arm is missing")
        if len(r.get("source_rows",[]))!=256:
            raise ValueError("source true next-byte rows missing")
        g=r["learned_gates"]
        if not (0<g["rho_s"]<1 and 0<g["erase_beta_s"]<.05
                and 0<g["write_epsilon_s"]<.01 and 0<g["read_mix_s"]<1
                and math.isfinite(g["source_bound_half_life_bytes"])):
            raise ValueError("trained actual slow gate unregistered or unstable")
        if not 0<=r["slow_old_state_delta_ratio_16"]<=r["slow_old_state_retention_upper_bound_16"]+1e-4:
            raise ValueError("slow long-state source retention bound broken")
        if not math.isfinite(r["same_original_fast_source_baseline_full_nll"]):
            raise ValueError("source reference baseline missing")
        for arm in ARMS:
            for field in ("nll","brier","top1"):
                if not math.isfinite(r["metrics"][arm][field]):
                    raise ValueError("nonfinite original source next-byte score")
        if not all(type(r.get(k)) is str and len(r[k])==64 for k in
                   ("original_dual_checkpoint_sha256","original_fitted_sample_order_sha256",
                    "original_source_target_sha256")):
            raise ValueError("source or actual original trained checkpoint SHA absent")
        runs[seed]=r
    if set(runs)!=set(SEEDS):
        raise ValueError("missing source fitted model seed")
    if len({r["original_source_target_sha256"] for r in runs.values()})!=1:
        raise ValueError("different genuine source owner/clock/next-byte observations")
    owners=None
    for r in runs.values():
        x=tuple((q["owner"],q["clock"],q["next_byte_real_source"]) for q in r["source_rows"])
        if owners is None:owners=x
        if x!=owners or len(set((a,b) for a,b,c in x))!=256:
            raise ValueError("missing or reordered original source true label")
    mean={arm:{field:statistics.mean(r["metrics"][arm][field] for r in runs.values())
               for field in ("nll","brier","top1")} for arm in ARMS}
    baseline={str(seed):runs[seed]["same_original_fast_source_baseline_full_nll"] for seed in SEEDS}
    paired={str(seed):runs[seed]["metrics"]["full"]["nll"]-baseline[str(seed)] for seed in SEEDS}
    return {"study":"LANGMEM079","status":"THREE_TRAINED_DUAL_TRUE_STREAM_ORIGINAL_SOURCE_NOFIT_LESION_VERIFIED",
       "model_seed_count":3,"no_new_fits":True,
       "exact_original_source_target_count":256,
       "original_source_target_sha256":next(iter({r["original_source_target_sha256"] for r in runs.values()})),
       "original_fast_source_baseline_run":40883211351,
       "source_paired":True,"parameters":499627,
       "additional_persistent_fp32_bytes":256,
       "full_native_persistent_fp32_bytes":4160,
       "gates":{field:statistics.mean(r["learned_gates"][field] for r in runs.values())
                for field in ("rho_s","erase_beta_s","write_epsilon_s","read_mix_s",
                              "source_bound_half_life_bytes")},
       "original_seed":{str(seed):{"original_baseline_nll":baseline[str(seed)],
                 "dual_full_nll":runs[seed]["metrics"]["full"]["nll"],
                 "paired_dual_minus_original_nll":paired[str(seed)],
                 "slow_early_erase_minus_full":runs[seed]["metrics"]["slow_wipe_t128_carried"]["nll"]-runs[seed]["metrics"]["full"]["nll"],
                 "slow_one_step_zero_minus_full":runs[seed]["metrics"]["one_step_slow_zero"]["nll"]-runs[seed]["metrics"]["full"]["nll"],
                 "slow_rho_s":runs[seed]["learned_gates"]["rho_s"],
                 "slow_old_state_delta_ratio_16":runs[seed]["slow_old_state_delta_ratio_16"],
                 "checkpoint_sha256":runs[seed]["original_dual_checkpoint_sha256"]}
                for seed in SEEDS},
       "mean_original_fast_stream_nll":statistics.mean(baseline.values()),
       "mean_paired_dual_minus_original_nll":statistics.mean(paired.values()),
       "dual_lower_NLL_seeds":sum(x<0 for x in paired.values()),
       "metrics":mean,
       "mean_slow_state_original_128_delta_ratio_16":statistics.mean(r["slow_old_state_delta_ratio_16"] for r in runs.values()),
       "mean_cpu_process_seconds":statistics.mean(r["measured_cpu_process_seconds"] for r in runs.values()),
       "independently_qualified":False,"gpu_minutes":0,
       "protected_TEST":"NOT_ACCESSED",
       "limits":["original real publisher DEV previously exposed, cap_all policy post-exposed DEV",
                 "short original 128-byte training only, continuous stream is nofit extrapolation",
                 "trained original baseline and dual have different training FLOPs",
                 "slow survival of one initial state is not source-grounded fact content retrieval",
                 "full original PyTorch source IEEE AD Lean correspondence and held-out source validation OPEN"]}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--results",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError("original evidence cannot be overwritten")
    r=aggregate(a.results)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(r,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print("LANGMEM079_CARRIED_SOURCE_AGGREGATE "+json.dumps({
      "original_mean_nll":r["mean_original_fast_stream_nll"],
      "dual_mean_nll":r["metrics"]["full"]["nll"],
      "paired_delta":r["mean_paired_dual_minus_original_nll"],
      "slow_zero_nll":r["metrics"]["one_step_slow_zero"]["nll"],
      "slow_wipe_at_128_nll":r["metrics"]["slow_wipe_t128_carried"]["nll"],
      "fast_zero_nll":r["metrics"]["one_step_fast_zero"]["nll"],
      "rho":r["gates"]["rho_s"],
      "source_half_life":r["gates"]["source_bound_half_life_bytes"],
      "slow_delta_16":r["mean_slow_state_original_128_delta_ratio_16"],
      "win_seeds":r["dual_lower_NLL_seeds"]},sort_keys=True))
if __name__=="__main__":main()
