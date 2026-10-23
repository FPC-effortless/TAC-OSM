"""ME074 genuine BPI2012 exposed-DEV all-case baseline audit, NO optimizer steps.

Reuse exact Git-owned ME070 FIT-only n-gram counts and original ME073 trained
forecasts. Score an absorbing EOR/continuation outcome on ALL eligible cases,
not a different survivor population per horizon. Iterated 1-gram is exact
matrix propagation; 3/5-gram are registered finite-particle approximations.
This is not a matched-parameter neural comparison or new held-out evidence.
"""
from __future__ import annotations
import argparse, hashlib, json, math, subprocess
from pathlib import Path
import numpy as np

from scripts.me073_autoregressive_world import source as real_source
from scripts.me069_workflow_source import digest
from scripts.me070_ngram_real_workflow import predict, stable_bytes

ROOT=Path("research/mathematical-engineering/me074")
BUNDLE=Path("research/replay-bundles/ME074")
ORIGINAL=Path("research/replay-bundles/ME073")
NGRAM_PATH=Path("research/replay-bundles/ME070/practical-ngrams/FIT_NGRAM_STATE.json")
QUERY_AGES=(2,6,16) # 16 is small and descriptive only; 64 has only two cases.
HORIZONS=(1,2,4,8)
NGRAM_ORDERS=(1,3,5)
PARTICLES=256
EPS=1e-6
SAMPLE_SEED=74074

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def immutable_git_blob(path):
    actual=Path(path).read_bytes()
    original=subprocess.check_output(["git","show","HEAD:"+Path(path).as_posix()])
    if actual!=original:
        raise ValueError("uncommitted or modified original FIT/forecast artifact "+str(path))

def validate_original_sources(pre,receipt):
    immutable_git_blob(NGRAM_PATH)
    immutable_git_blob(ORIGINAL/"ORIGINAL_FORECASTS.npz")
    immutable_git_blob(ORIGINAL/"ORIGINAL_SCORES.json")
    count=json.loads(NGRAM_PATH.read_text())
    if count["real_source_sha256"]!=receipt["source_sha256"]:
        raise ValueError("original FIT ngram source identity mismatch")
    if count["vocabulary"]!=len(pre["activities"])+1:
        raise ValueError("frozen 5-gram and ME073 vocabulary differ")
    if not count["selected_without_DEV_or_TEST"]:
        raise ValueError("ngram state used outcomes")
    probability=np.asarray(count["first_order_transition_probability"],dtype=np.float64)
    if not np.isfinite(probability).all() or not np.allclose(probability.sum(1),1.,atol=2e-4):
        raise ValueError("original FIT transition probability not normalized")
    step_logits=np.asarray(pre["step_survival_logit"],dtype=np.float64)
    survival=1./(1.+np.exp(-step_logits))
    if len(survival)!=count["vocabulary"] or not np.isfinite(survival).all():
        raise ValueError("invalid FIT source EOR survival prior")
    return count,survival,probability

def iterated_bigram_matrix(prob,survival):
    n=len(survival)
    m=np.zeros((n+1,n+1),np.float64)
    m[:n,:n]=prob*survival[:,None]
    m[:n,n]=1.-survival
    m[n,n]=1. # absorbing source EOR, NOT independently verified completion
    if not np.allclose(m.sum(1),1.,atol=1e-8):
        raise ValueError("iterated bigram+EOR is not a Markov transition")
    return m

def exact_markov(current,m):
    p=np.zeros(len(m),np.float64);p[int(current)]=1.
    result={}
    for h in range(1,max(HORIZONS)+1):
        p=p@m
        if h in HORIZONS:result[h]=p.copy()
    return result

def sample_ngram(history,state,survival,order,case_key):
    """Actual autoregressive no-future-label sampled continuations.
    h1 analytical distribution; h2..8 are approximate 256 particle marginals.
    Monte Carlo stochasticity must be disclosed, and should not be mistaken
    for intrinsic trained model uncertainty or precise likelihood.
    """
    n=len(survival)
    h0=list(map(int,history))
    h1=np.asarray(predict(state,h0,order),np.float64)
    h1=h1/h1.sum()
    first=np.r_[h1*survival[h0[-1]],1.-survival[h0[-1]]]
    seed=int(hashlib.sha256(f"me074:{SAMPLE_SEED}:{order}:{case_key}".encode()).hexdigest()[:15],16)
    gen=np.random.default_rng(seed)
    sampled=[h0.copy() for _ in range(PARTICLES)]
    ended=np.zeros(PARTICLES,dtype=bool)
    results={}
    for h in range(1,max(HORIZONS)+1):
        counts=np.zeros(n+1,dtype=np.float64)
        for i,context in enumerate(sampled):
            if ended[i]:
                counts[n]+=1
                continue
            if gen.random()>=survival[context[-1]]:
                ended[i]=True
                counts[n]+=1
                continue
            p=np.asarray(predict(state,context,order),np.float64)
            p=p/p.sum()
            a=int(gen.choice(n,p=p))
            context.append(a)
            if len(context)>5:del context[:-5]
            counts[a]+=1
        if h in HORIZONS:
            raw=first if h==1 else counts/PARTICLES
            # Same mass-smoothing convention as reported ME073 later-QMC marginals.
            result=(raw+EPS)/(1.+(n+1)*EPS)
            results[h]=result
    return results

def score_prediction(p,truth,exists,n):
    if not np.isclose(p.sum(),1.,atol=1e-8):raise ValueError("non-normalized all-case EOR score")
    index=truth if exists else n
    log_loss=-math.log(max(1e-12,float(p[index])))
    brier=float(np.sum(p*p)-2*p[index]+1.)
    end_brier=float((p[n]-(0. if exists else 1.))**2)
    condition=p[:n]/max(1e-12,p[:n].sum())
    return log_loss,brier,end_brier,float(np.argmax(condition)==truth) if exists else None,(-math.log(max(1e-12,condition[truth]))) if exists else None

def summarise(matrix,exists,truth,n):
    # [case, horizon, vocabulary plus EOR]
    scores={}
    for k,h in enumerate(HORIZONS):
        eligible=exists[:,k]
        f=np.asarray([score_prediction(p,t,yes,n) for p,t,yes in zip(matrix[:,k],truth[:,k],eligible)],dtype=object)
        scores[str(h)]={
            "all_reached_cases":int(len(f)),
            "actual_future_record_cases":int(eligible.sum()),
            "source_EOR_cases":int((~eligible).sum()),
            "all_case_joint_activity_or_EOR_NLL":float(np.asarray(f[:,0],np.float64).mean()),
            "all_case_joint_Brier":float(np.asarray(f[:,1],np.float64).mean()),
            "EOR_binary_Brier":float(np.asarray(f[:,2],np.float64).mean()),
            "future_conditional_activity_accuracy":float(np.asarray(f[eligible,3],np.float64).mean()) if eligible.any() else None,
            "future_conditional_activity_NLL":float(np.asarray(f[eligible,4],np.float64).mean()) if eligible.any() else None,
        }
    return scores

def compare_original(original,rows,truth,exists,n):
    reference={}
    seed_keys=sorted(original)
    for age,data in rows.items():
        published=json.loads((ORIGINAL/"ORIGINAL_SCORES.json").read_text())["seeds"]
        ids=digest([c["id"] for c in data])
        for seed in seed_keys:
            if published[seed][str(age)]["IDs_sha256"]!=ids:
                raise ValueError("ME073 original per-age case IDs not in same order")
        with np.load(ORIGINAL/"ORIGINAL_FORECASTS.npz") as src:
            probs=[src[f"s{seed}_age{age}"][:,:,:n+1] for seed in seed_keys]
        m=np.stack(probs)
        if m.shape[1:]!=(len(data),len(HORIZONS),n+1):
            raise ValueError("ME073 original forecast dimensions changed")
        # Independently score exact frozen original five-seed arrays, then verify
        # their aggregates against the published ME073 original report.
        audits=[]
        for i,seed in enumerate(seed_keys):
            scored=summarise(m[i],exists[age],truth[age],n)
            for h in HORIZONS:
                public=published[seed][str(age)]["horizons"][str(h)]
                if abs(scored[str(h)]["all_case_joint_activity_or_EOR_NLL"]-public["QMC_activity_or_boundary_NLL"])>1e-5:
                    raise ValueError("frozen actual original ME073 EOR log loss not reproduced")
            audits.append(scored)
        reference[str(age)]={
            "seeds":[int(z) for z in seed_keys],
            "per_seed_original_scores":audits,
            "five_seed_all_case_mean_NLL":{str(h):float(np.mean([
                a[str(h)]["all_case_joint_activity_or_EOR_NLL"] for a in audits]))
                for h in HORIZONS},
            "five_seed_mean_prediction_NLL_descriptive":summarise(m.mean(0),exists[age],truth[age],n),
            "note":"Five trained seeds and repeated report queries are not independent institutional cases.",
        }
    return reference

def main(cache):
    cohorts,pre,receipt=real_source(cache)
    state,survival,prob=validate_original_sources(pre,receipt)
    n=int(state["vocabulary"])
    bigram=iterated_bigram_matrix(prob,survival)
    series={};truth={};exists={};baselines={}
    for age in QUERY_AGES:
        cases=[c for c in cohorts["REPORT"] if c["n"]>=age]
        if not cases:raise ValueError("no eligible real exposed cases")
        series[age]=cases
        targets=np.zeros((len(cases),len(HORIZONS)),np.int64)
        present=np.zeros_like(targets,dtype=bool)
        out={k:np.empty((len(cases),len(HORIZONS),n+1),np.float64) for k in NGRAM_ORDERS}
        for i,case in enumerate(cases):
            history=[int(a) for a in case["activity"][:age]]
            predicted_bigram=exact_markov(history[-1],bigram)
            for k,h in enumerate(HORIZONS):
                t=age+h-1
                present[i,k]=t<case["n"]
                targets[i,k]=int(case["activity"][t]) if present[i,k] else 0
                out[1][i,k]=predicted_bigram[h]
            for order in (3,5):
                res=sample_ngram(history,state,survival,order,case["id"])
                for j,h in enumerate(HORIZONS):out[order][i,j]=res[h]
            if (i+1)%200==0:print(json.dumps({"study":"ME074","age":age,"real_cases":i+1,"model_fits":0}),flush=True)
        truth[age]=targets;exists[age]=present
        baselines[str(age)]={str(order):summarise(out[order],present,targets,n) for order in NGRAM_ORDERS}
        print(json.dumps({"age":age,"cases":len(cases),
            "bigram_all_case_h8_NLL":baselines[str(age)]["1"]["8"]["all_case_joint_activity_or_EOR_NLL"],
            "5gram_all_case_h8_NLL":baselines[str(age)]["5"]["8"]["all_case_joint_activity_or_EOR_NLL"]}),flush=True)
    original=sorted(json.loads((ORIGINAL/"ORIGINAL_SCORES.json").read_text())["seeds"])
    neural=compare_original(original,series,truth,exists,n)
    result={
        "study":"ME074_EXPOSED_REAL_END_AWARE_PRACTICAL_BASELINES",
        "scope":"Original exposed BPI2012 DEV REPORT only; no new TEST or site or model fits",
        "source_original_sha256":receipt["source_sha256"],
        "original_fit_ngram_state_sha256":sha(NGRAM_PATH),
        "original_ME073_scores_sha256":sha(ORIGINAL/"ORIGINAL_SCORES.json"),
        "original_ME073_forecasts_sha256":sha(ORIGINAL/"ORIGINAL_FORECASTS.npz"),
        "n_neural_models_fitted":0,
        "n_synthetic_trained_models":0,
        "frozen_100k_checkpoint_models_restored":0,
        "methods":{
          "iterated_bigram":"exact N+1 absorbing EOR transition powers fitted only on original FIT",
          "order_3_and_5":"256 fixed-seed autoregressive particle paths per exposed real case; exact first step; later approximate with epsilon smoothing",
          "event_end":"last observed record source EOR hindsight proxy, not known business completion",
          "matching":"Identical real query-age case IDs and horizon labels to original ME073 5-seed forecast files",
          "information":"Baseline activity history only, while ME073 also has known resource/time. Practical baseline NOT matched-input or matched-param superiority test",
          "replay":"Original immutable FIT-count and ME073 model output bytes exactly compared to Git HEAD",
        },
        "ages":list(QUERY_AGES),
        "horizons":list(HORIZONS),
        "practical_baseline_metrics":baselines,
        "five_seed_original_ME073_metrics":neural,
        "primary_claim":"DESCRIPTIVE_RETROSPECTIVE_ONLY_NO_TAC_ARCHITECTURAL_WIN",
        "missing_matched_controls":["100k GRU autoregressive","100k causal Transformer","100k direct multi-horizon neural head"],
    }
    BUNDLE.mkdir(parents=True,exist_ok=True)
    output=BUNDLE/"REAL_DEV_END_AWARE_BASELINE.json"
    if output.exists():raise RuntimeError("original real baseline score already published; refuse rewrite")
    output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print("ME074_GENUINE_EXPOSED_REAL_BASELINE="+json.dumps({
        "query_cases":{str(a):len(rows) for a,rows in series.items()},
        "source_TEST_opened":False,"trained_new_models":0,
        "age6_h8_NLL":{name:data["8"]["all_case_joint_activity_or_EOR_NLL"] for name,data in baselines["6"].items()},
        "original_ME073_age6_h8_mean_NLL":neural["6"]["five_seed_all_case_mean_NLL"]["8"]},
        sort_keys=True),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--cache",type=Path,required=True);a=p.parse_args()
    main(a.cache)
