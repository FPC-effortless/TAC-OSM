"""ME077A: no-fit genuine robot movement-discrimination scoring.

Uses only immutable original ME077 source-observed REPORT prediction receipts.
No model loads required, no optimizer updates, no source TEST, no tuning.
Cross-case uncertainty uses ORIGINAL 40 episodes, NOT 5x960 independent cases.
"""
from __future__ import annotations
import hashlib, json, math, subprocess
from pathlib import Path
import numpy as np
from scripts.validate_research_parity import check_manifest

ROOT=Path("research/mathematical-engineering/me077a")
OUT=Path("research/replay-bundles/ME077A")
SRC=Path("research/replay-bundles/ME077")
REG=ROOT/"FAIR_COMPARISON.json"
SEEDS=(77001,77002,77003,77004,77005)
THRESHOLDS=(.05,.10,.25,.50)
BINS=(0.,.05,.10,.20,.40,.60,.80,1.)
NBOOT=2000
BOOT_SEED=77078
EXPECTED_ORIGINAL_SUMMARY="095816cf45de5ff7e3e1e3a6fd9f84f0165302c72fc725f7abb0f69bf438c959"

def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()

def original(path):
    path=Path(path)
    binary=path.read_bytes()
    owned=subprocess.check_output(["git","show","HEAD:"+path.as_posix()])
    if sha_bytes(owned)!=sha_bytes(binary):
        raise RuntimeError(f"original Git-owned real source prediction mutated: {path}")
    return binary

def auc(y, p):
    y=np.asarray(y,dtype=bool);p=np.asarray(p,dtype=float)
    pos=int(y.sum());neg=len(y)-pos
    if pos==0 or neg==0:return None
    order=np.argsort(p,kind="mergesort")
    vals=p[order]; yy=y[order]
    end=np.r_[vals[1:]!=vals[:-1],True]
    last=np.flatnonzero(end)
    start=np.r_[0,last[:-1]+1]
    average_rank=(start+1+last+1)*.5
    ranks=np.repeat(average_rank,last-start+1)
    positive_rank=float(ranks[yy].sum())
    return (positive_rank-pos*(pos+1)/2)/(pos*neg)

def ap(y,p):
    y=np.asarray(y,dtype=bool);p=np.asarray(p,dtype=float)
    pos=int(y.sum())
    if pos==0:return None
    order=np.argsort(-p,kind="mergesort")
    vals=p[order];labels=y[order]
    last=np.flatnonzero(np.r_[vals[1:]!=vals[:-1],True])
    tp=np.cumsum(labels)[last]
    precision=tp/(last+1)
    recall=tp/pos
    return float(np.diff(np.r_[0.,recall])@precision)

def brier(y,p):
    return float(np.mean((np.asarray(y,dtype=float)-np.asarray(p,dtype=float))**2))

def ranking(y,p):
    y=np.asarray(y,dtype=bool);p=np.asarray(p,dtype=float)
    n=len(y)
    k=max(1,int(math.ceil(.1*n)))
    top=np.argsort(-p,kind="mergesort")[:k]
    return {"ROC_AUC":auc(y,p),"average_precision":ap(y,p),
            "top10_count":k,"top10_true_motion":int(y[top].sum()),
            "top10_precision":float(y[top].mean()),
            "top10_movement_recall":float(y[top].sum()/y.sum()) if y.any() else None,
            "mean_probability_moving":float(p[y].mean()) if y.any() else None,
            "mean_probability_stationary":float(p[~y].mean()) if (~y).any() else None,
            "Brier":brier(y,p)}

def thresholds(y,p):
    y=np.asarray(y,dtype=bool);p=np.asarray(p,dtype=float);res={}
    for th in THRESHOLDS:
        alarm=p>=th
        tp=int((alarm&y).sum());fp=int((alarm&~y).sum())
        fn=int((~alarm&y).sum());tn=int((~alarm&~y).sum())
        res[str(th)]={"TP":tp,"FP":fp,"FN":fn,"TN":tn,
                      "precision":tp/(tp+fp) if tp+fp else None,
                      "recall":tp/(tp+fn) if tp+fn else None,
                      "false_positive_rate":fp/(fp+tn) if fp+tn else None}
    return res

def bins(y,p):
    y=np.asarray(y,dtype=bool);p=np.asarray(p,dtype=float);out=[]
    for lo,hi in zip(BINS[:-1],BINS[1:]):
        select=(p>=lo)&(p<(hi if hi<1 else 1.00000001))
        out.append({"lower":lo,"upper":hi,"n":int(select.sum()),
                    "mean_probability":float(p[select].mean()) if select.any() else None,
                    "observed_motion_frequency":float(y[select].mean()) if select.any() else None})
    return out

def bootstrap(y,p,episode):
    groups=sorted(set(episode))
    positions=[np.asarray([i for i,v in enumerate(episode) if v==group],dtype=int) for group in groups]
    if len(groups)!=40:
        raise ValueError(f"expected 40 ORIGINAL robot episode clusters; got {len(groups)}")
    rng=np.random.default_rng(BOOT_SEED)
    prior=float(y.mean())   # HINDSIGHT REPORT oracle; not available to an online controller
    vals={"ROC_AUC":[],"average_precision":[],"Brier_minus_hindsight_prevalence":[]}
    for _ in range(NBOOT):
        draws=rng.integers(0,len(groups),size=len(groups))
        idx=np.concatenate([positions[j] for j in draws])
        yy=y[idx];pp=p[idx]
        a=auc(yy,pp);average=ap(yy,pp)
        if a is not None and average is not None:
            vals["ROC_AUC"].append(a)
            vals["average_precision"].append(average)
        vals["Brier_minus_hindsight_prevalence"].append(
            brier(yy,pp)-brier(yy,np.full(len(yy),prior)))
    return {name:{"95pct_interval":np.quantile(z,[.025,.975]).tolist(),
                  "valid_bootstraps":len(z)}
            for name,z in vals.items()}

def load_original():
    reg=json.loads(REG.read_text())
    status=check_manifest(reg,"preflight")
    if status["status"]!="PASS_DIAGNOSTIC_ONLY" or not reg["no_new_model_fits"]:
        raise RuntimeError("this study is strictly no-fit")
    summary=original(SRC/"ORIGINAL_SUMMARY.json")
    if sha_bytes(summary)!=EXPECTED_ORIGINAL_SUMMARY or sha_bytes(summary)!=reg["source_original_summary_sha256"]:
        raise RuntimeError("original ME077 published fitted result SHA unexpectedly changed")
    original_summary=json.loads(summary)
    actual=[int(row["seed"]) for row in original_summary["seeds"]]
    if actual!=list(SEEDS):
        raise RuntimeError("five original model seeds/order differ")
    all_rows=[];identity=None;per_seed_sha={}
    for seed in SEEDS:
        source=SRC/f"seed-{seed}"/"ORIGINAL_PREDICTIONS.json"
        raw=original(source)
        per_seed_sha[str(seed)]=sha_bytes(raw)
        rows=json.loads(raw)
        if len(rows)!=960:raise RuntimeError("original ME077 real case count not 960")
        ids=[(r["task"],r["episode_index"],r["frame_index"]) for r in rows]
        if len(set(ids))!=len(ids):raise RuntimeError("original source contains duplicate real query")
        if identity is None:identity=ids
        elif ids!=identity:raise RuntimeError("frozen original case ID/order changes across model seeds")
        all_rows.append(rows)
    return all_rows,per_seed_sha,sha_bytes(summary)

def run():
    rows,sha,src_hash=load_original()
    labels=np.asarray([row["motion_rich"] for row in rows[0]],dtype=bool)
    if int(labels.sum())!=84 or len(labels)!=960:
        raise RuntimeError("historical exposed motion prevalence changed")
    episodes=[f'{r["task"]}:{r["episode_index"]}' for r in rows[0]]
    tasks=[r["task"] for r in rows[0]]
    if len(set(episodes))!=40:raise RuntimeError("changed real episode sample")
    model_p=[];component_p=[];errors=[]
    original_seed={}
    for seed,records in zip(SEEDS,rows):
        if [r["motion_rich"] for r in records]!=labels.tolist():
            raise RuntimeError("real labels not identical across original trained seeds")
        p=np.asarray([r["motion_probability"] for r in records],dtype=float)
        c=np.asarray([float(np.mean(r["mixture_probability"])) for r in records])
        e=np.asarray([r["physical_mae_model"] for r in records],dtype=float)
        if not (np.isfinite(p).all() and np.isfinite(c).all() and np.isfinite(e).all()
                and np.all((p>=0)&(p<=1)) and np.all((c>=0)&(c<=1))):
            raise RuntimeError("nonfinite or out-of-bounds model outputs")
        original_seed[str(seed)]={"motion":ranking(labels,p),
                                   "coordinate_mixture_mean":ranking(labels,c),
                                   "all_physical_MAE":float(e.mean()),
                                   "moving_physical_MAE":float(e[labels].mean()),
                                   "stationary_physical_MAE":float(e[~labels].mean())}
        model_p.append(p);component_p.append(c);errors.append(e)
    p=np.stack(model_p).mean(0);c=np.stack(component_p).mean(0)
    base=float(labels.mean());base_brier=base*(1.-base)
    task_rows={}
    for task in sorted(set(tasks)):
        mask=np.array([v==task for v in tasks])
        task_rows[task]={"n":int(mask.sum()),"moving":int(labels[mask].sum()),
                         "motion":ranking(labels[mask],p[mask]),
                         "base_rate":float(labels[mask].mean())}
    result={
        "study":"ME077A_ORIGINAL_REAL_MOTION_DISCRIMINATION_NO_FIT",
        "claim_tier":"EXPOSED_DEV_DIAGNOSTIC_ONLY",
        "trained_new_models":0,"new_optimizer_updates":0,
        "real_source":"BodenAI original historically exposed genuine 40 episodes / 960 next-state questions",
        "source_sha256":src_hash,
        "five_original_prediction_file_SHA256":sha,
        "trained_model_count":5,"original_real_queries":960,
        "real_episode_clusters":40,"motion_rich_queries":int(labels.sum()),
        "motion_base_rate":base,
        "hindsight_REPORT_climatology_Brier_NOT_DEPLOYABLE":base_brier,
        "original_5seed_mean_auxiliary_motion":ranking(labels,p),
        "original_5seed_mean_coordinate_mixture_proxy":ranking(labels,c),
        "motion_Brier_skill_against_hindsight_REPORT_climatology":
            1.-brier(labels,p)/base_brier,
        "fixed_thresholds":thresholds(labels,p),
        "fixed_calibration_bins":bins(labels,p),
        "original_five_individual_seed_statistics":original_seed,
        "original_task_statistics":task_rows,
        "episode_cluster_2000_bootstrap":bootstrap(labels,p,episodes),
        "outcome_limits":[
            "DEV historically exposed, not independent untouched real task or source",
            "Only motion defined by original 0.025 mean 14D displacement threshold, no hazard/event benefits verified",
            "Hindsight REPORT prevalence not a decision-time permitted prior",
            "Seed ensemble is a diagnostic of previously trained models, not a new deployable fitted model",
            "Coordinate-level mixture weights are not a global motion-event probability",
            "No new policy/authorization/causal or Woyengi claims",
            "No parameter/compute-matched GRU or Transformer head-to-head here"
        ]
    }
    dest=OUT/"ORIGINAL_REAL_MOTION_RANKING.json"
    OUT.mkdir(parents=True,exist_ok=True)
    if dest.exists():
        prior=json.loads(dest.read_text())
        if prior!=result:raise RuntimeError("original Git evidence changed; refuse overwrite")
        print("ME077A_NO_FIT_ORIGINAL_REPLAY_PASS",flush=True)
        return
    dest.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("ME077A_GENUINE_REAL_ORIGINAL_NO_FIT_AUDIT="+json.dumps({
        "motion_rate":base,
        "aux_Brier":result["original_5seed_mean_auxiliary_motion"]["Brier"],
        "hindsight_Brier":base_brier,
        "auc":result["original_5seed_mean_auxiliary_motion"]["ROC_AUC"],
        "AP":result["original_5seed_mean_auxiliary_motion"]["average_precision"],
        "top10_recall":result["original_5seed_mean_auxiliary_motion"]["top10_movement_recall"],
        "gt0_5":result["fixed_thresholds"]["0.5"],
        "new_training":0}),flush=True)

if __name__=="__main__":run()
