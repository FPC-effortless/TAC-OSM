#!/usr/bin/env python3
"""LANGLAYER040 independent source/seed/hash-complete factorial aggregation."""
import argparse,hashlib,json,math,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.langlayer035_sampling import fit_sources,schedule_receipt
SEEDS=(1701,1702,1703,1704,1705)
NEW={"state48_no_graph":499623,"state59_full_graph":499848}
FACTORS={
 "h48_graph":"original035",
 "h48_plain":"state48_no_graph",
 "h59_graph":"state59_full_graph",
 "h59_plain":"original039",
}
def aggregate(root,old035,old039):
    a=json.loads(Path(old035).read_text());b=json.loads(Path(old039).read_text())
    if (a.get("study"),a.get("cohort"),a.get("status"))!=(
       "LANGLAYER035",15,"COMPLETE_INTERNAL_EXPOSED_DEV_EXPLORATORY_NOT_FLOP_MATCHED"):
        raise ValueError("original graph48 reference no longer verified")
    if (b.get("study"),b.get("status"),b.get("parameter_count"))!=(
       "LANGLAYER039","COMPLETE_5_ORIGINAL_CPU_FITS_INTERNAL_EXPOSED_DEV_ONLY",499901):
        raise ValueError("original no-graph59 reference no longer verified")
    sch={s:schedule_receipt(fit_sources(),s,steps=320) for s in SEEDS}
    rows={}
    for p in Path(root).rglob("FIT_AND_EXPOSED_DEV_RECEIPT.json"):
        r=json.loads(p.read_text());key=(r.get("arm"),r.get("seed"))
        if key[0] not in NEW or key[1] not in SEEDS or key in rows:
            raise ValueError("unregistered duplicated original artifact")
        if r["study"]!="LANGLAYER040" or r["updates"]!=320 or r["target_bytes_seen"]!=327680 or r["examples_seen"]!=2560:
            raise ValueError("wrong training/source exposure")
        if r["parameters"]!={"trainable":NEW[key[0]],"deployed":NEW[key[0]]}:
            raise ValueError("invalid active/deployed learned parameter count")
        if (r["source_sample_order_sha256"],r["source_bytes_sha256"])!=(
            sch[key[1]]["sample_order_sha256"],sch[key[1]]["sample_bytes_sha256"]):
            raise ValueError("original source-owner source digest mismatch")
        if r["source_commit"]!="60403a5409ff2c3f3b07dd2ca91a7a3e096839c7":
            raise ValueError("source commit drifted")
        if not r["checkpoint_reloaded_identical_scored_DEV"]:
            raise ValueError("original trained checkpoint replay mismatch")
        cp=p.parent/r["checkpoint_filename"]
        if not cp.is_file() or hashlib.sha256(cp.read_bytes()).hexdigest()!=r["checkpoint_sha256"]:
            raise ValueError("missing or corrupted real trained checkpoint")
        if r["protected_TEST"]!="NOT_ACCESSED" or r["GPU_minutes"]!=0 or r["matched_training_FLOPs"]!="NOT_VERIFIED":
            raise ValueError("scientific or permission boundary violated")
        sc=r["DEV_EXPOSED_4096_ONLY"]
        if sc["n"]!=4096 or any(not math.isfinite(float(sc[k])) for k in ("nll","top1","top5","brier")):
            raise ValueError("source DEV altered")
        rows[key]=r
    expected={(name,s) for name in NEW for s in SEEDS}
    if set(rows)!=expected:raise ValueError("INCOMPLETE 10 CHECKPOINT FACTORIAL")
    grid={}
    for s in SEEDS:
        grid[str(s)]={
         "h48_graph":float(a["per_seed"]["tac_layer2"][str(s)]["nll"]),
         "h59_plain":float(b["per_seed"][str(s)]["nll"]),
         "h48_plain":float(rows["state48_no_graph",s]["DEV_EXPOSED_4096_ONLY"]["nll"]),
         "h59_graph":float(rows["state59_full_graph",s]["DEV_EXPOSED_4096_ONLY"]["nll"])
        }
    effects={}
    expr={
      "remove_graph_h48": lambda r:r["h48_plain"]-r["h48_graph"],
      "remove_graph_h59": lambda r:r["h59_plain"]-r["h59_graph"],
      "widen_recurrence_graph_on":lambda r:r["h59_graph"]-r["h48_graph"],
      "widen_recurrence_graph_off":lambda r:r["h59_plain"]-r["h48_plain"],
      "interaction_widen_effect_off_minus_on":lambda r:(r["h59_plain"]-r["h48_plain"])-(r["h59_graph"]-r["h48_graph"])
    }
    for name,formula in expr.items():
        vals=[formula(grid[str(s)]) for s in SEEDS]
        mu=statistics.mean(vals);sd=statistics.stdev(vals)
        margin=2.7764451051977987*sd/math.sqrt(5)
        effects[name]={"mean_delta_nll":mu,"sd":sd,"five_seed_pairing":dict(zip(map(str,SEEDS),vals)),
          "five_seed_positive":sum(x>0 for x in vals),
          "descriptive_95pct_t_interval":[mu-margin,mu+margin]}
    report={
      "study":"LANGLAYER040",
      "status":"COMPLETE_ORIGINAL_TEN_CHECKPOINTS_2X2_BUDGETED_ALLOCATION",
      "real_source":"Publisher PSF CPython3.13.0 genuine FIT and previously EXPOSED DEV",
      "new_training_jobs":10,"original_reference_runs":[40786118546,40795790808],
      "full_composed_Lean_IEEE_AD":"OPEN",
      "protected_TEST":"NOT_ACCESSED","GPU_minutes":0,
      "same_FFN_width_across_corners":False,
      "training_FLOPs_matched":False,
      "independent_architectural_superiority":False,
      "new_models":{},"factorial_mean_nll":{},"factor_effects":effects,
      "grid_per_seed":grid,"all_original_checkpoint_sha256":{}}
    report["factorial_mean_nll"]={k:statistics.mean(row[k] for row in grid.values()) for k in FACTORS}
    for arm in NEW:
        vals=[rows[arm,s]["DEV_EXPOSED_4096_ONLY"] for s in SEEDS]
        res={}
        for k in ("nll","top1","top5","brier"):
            res[k+"_mean"]=statistics.mean(v[k] for v in vals)
            res[k+"_sd"]=statistics.stdev(v[k] for v in vals)
        res["perplexity_mean"]=statistics.mean(math.exp(v["nll"]) for v in vals)
        res["cpu_process_seconds_mean"]=statistics.mean(rows[arm,s]["measured_cpu_process_seconds"] for s in SEEDS)
        res["params"]=NEW[arm]
        res["per_seed"]={str(s):rows[arm,s]["DEV_EXPOSED_4096_ONLY"] for s in SEEDS}
        report["new_models"][arm]=res
        report["all_original_checkpoint_sha256"][arm]={str(s):rows[arm,s]["checkpoint_sha256"] for s in SEEDS}
    return report

def main():
    p=argparse.ArgumentParser()
    for flag in ("results","old035","old039","output"):
        p.add_argument("--"+flag,type=Path,required=True)
    a=p.parse_args()
    r=aggregate(a.results,a.old035,a.old039)
    if a.output.exists():raise FileExistsError("never overwrite evidence")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(r,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"models":r["new_models"],"factor_means":r["factorial_mean_nll"],
                      "factor_effects":r["factor_effects"]},sort_keys=True))
if __name__=="__main__":main()
