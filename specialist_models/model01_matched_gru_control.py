#!/usr/bin/env python3
"""Authentic UCI374 original-FIT control: parameter-matched standard GRU.

103k-scale conventional GRU, identical 3-source features, 48 observation
history, six-step target, 3 original seeds, 160 updates x 64 same minibatch
indices/loss/hyperparameters as TAC original. Only exposed FIT/CAL/DEV.
Different architectures have different FLOPs; no strict compute-matched win.
Full model+optimizer+RNG+draws original checkpoints go to Git and no-fit replay.
"""
from __future__ import annotations
import argparse,base64,gzip,hashlib,io,json,math,os,random,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import model01_native_tac_real_fit as tac

ROOT=Path(__file__).resolve().parents[1]
MODEL="MODEL01-GRU-103K-CONTROL-001"
TAC_ORIGINAL=ROOT/"research/replay-bundles/MODEL01-TAC-REAL-FIT-001/ORIGINAL_FIT.json"
TARGET_PARAMS=103252
EXPECTED_PARAMS=102883
MAX_REL_GAP=.01

class ControlError(ValueError):pass

class ControlGRU(nn.Module):
    """Single-layer 145 hidden-state recurrent GRU, same 3 source input features."""
    def __init__(self):
        super().__init__()
        self.rnn=nn.GRU(input_size=3,hidden_size=145,num_layers=1,batch_first=True)
        self.head=nn.Sequential(nn.Linear(145,256),nn.GELU(),nn.Linear(256,1))
    def forward(self,x):
        if x.ndim!=3 or x.shape[1:]!=(tac.HISTORY,3):
            raise ControlError("GRU requires identical original 48x3 causal input")
        z,_=self.rnn(x)
        return x[:,-1,0]+self.head(z[:,-1]).squeeze(-1)

def preflight():
    m=ControlGRU()
    n=tac.params(m)
    gap=abs(n-TARGET_PARAMS)/TARGET_PARAMS
    if n!=EXPECTED_PARAMS or gap>MAX_REL_GAP or not 95000<=n<=105000:
        raise ControlError("parameter match outside frozen preregistered tolerance")
    if tac.SEEDS!=(1701,1702,1703) or tac.UPDATES!=160 or tac.BATCH!=64 or tac.LR!=.001:
        raise ControlError("native training recipe mismatch")
    existing=json.loads(TAC_ORIGINAL.read_bytes())
    if (existing.get("study")!=tac.STUDY or existing.get("source_zip_sha256")!=tac.SOURCE
        or existing.get("trainable_params")!=TARGET_PARAMS or existing.get("seeds")!=list(tac.SEEDS)
        or existing.get("train_updates_per_seed")!=tac.UPDATES
        or existing.get("batch_size")!=tac.BATCH
        or existing.get("original_protected_tail_parsed") is not False
        or len(existing.get("runs",[]))!=3):
        raise ControlError("original TAC training/source contract changed")
    return existing,gap

def fit_seed(seed,fit,cal,dev,mean,std,original):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    net=ControlGRU()
    opt=torch.optim.AdamW(net.parameters(),lr=tac.LR,weight_decay=.01)
    x,y,_=fit
    sampler=np.random.default_rng(seed);draws=hashlib.sha256()
    net.train()
    for _ in range(tac.UPDATES):
        ids=sampler.integers(0,len(x),size=tac.BATCH)
        draws.update(ids.tobytes())
        prediction=net(x[ids])
        loss=F.smooth_l1_loss(prediction,y[ids],beta=.5)
        if not torch.isfinite(loss):raise ControlError("nonfinite real FIT loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
        opt.step()
    sh=draws.hexdigest()
    if sh!=original["sample_order_sha256"]:
        raise ControlError("exact original source minibatch indices differ")
    cal_metrics,cal_abs=tac.score(net,cal,mean,std)
    q=float(np.sort(cal_abs)[math.ceil((len(cal_abs)+1)*.9)-1])
    dev_metrics,_=tac.score(net,dev,mean,std,q)
    out=io.BytesIO()
    torch.save({"model":net.state_dict(),"optimizer":opt.state_dict(),
                "torch_rng":torch.get_rng_state(),
                "numpy_rng":sampler.bit_generator.state,
                "python_rng":random.getstate(),
                "seed":seed,"steps":tac.UPDATES,
                "sample_order_sha256":sh},out)
    raw=out.getvalue()
    return {"seed":seed,"cal":cal_metrics,"dev":dev_metrics,"q90_wh":q,
            "steps":tac.UPDATES,"sample_order_sha256":sh,
            "matched_native_checkpoint_sha256":original["checkpoint_sha256"],
            "checkpoint_sha256":hashlib.sha256(raw).hexdigest(),
            "checkpoint_b64_gzip":base64.b64encode(gzip.compress(raw,mtime=0)).decode()}

def replay_seed(bundle,cal,dev,mean,std,original):
    raw=gzip.decompress(base64.b64decode(bundle["checkpoint_b64_gzip"],validate=True))
    if hashlib.sha256(raw).hexdigest()!=bundle["checkpoint_sha256"]:
        raise ControlError("original GRU checkpoint SHA mismatch")
    d=torch.load(io.BytesIO(raw),map_location="cpu",weights_only=True)
    if (d["seed"]!=bundle["seed"] or d["steps"]!=tac.UPDATES
        or d["sample_order_sha256"]!=original["sample_order_sha256"]
        or bundle["matched_native_checkpoint_sha256"]!=original["checkpoint_sha256"]):
        raise ControlError("original GRU training sample-order/seed mismatch")
    net=ControlGRU()
    if tac.params(net)!=EXPECTED_PARAMS:
        raise ControlError("GRU source model architecture changed")
    net.load_state_dict(d["model"],strict=True)
    opt=torch.optim.AdamW(net.parameters(),lr=tac.LR,weight_decay=.01)
    opt.load_state_dict(d["optimizer"])
    cal_score,abs_err=tac.score(net,cal,mean,std)
    q=float(np.sort(abs_err)[math.ceil((len(abs_err)+1)*.9)-1])
    dev_score,_=tac.score(net,dev,mean,std,q)
    for name,real,pinned in (("CAL",cal_score,bundle["cal"]),("DEV",dev_score,bundle["dev"])):
        for key in real:
            if not math.isclose(real[key],pinned[key],rel_tol=1e-6,
                                abs_tol=(0.0004 if key=="width90_wh" else 1e-6)):
                # CPU kernel differences can move CAL 90% quantile by one
                # float32 representable step, <0.0002 Wh; do not affect source
                # outcome, model bytes, or observed 90% coverage.
                raise ControlError("source "+name+" replay difference "+key)
    if not math.isclose(q,bundle["q90_wh"],rel_tol=1e-6,abs_tol=0.0002):
        raise ControlError("source CAL quantile replay mismatch")
    return {"seed":bundle["seed"],"original_checkpoint_sha256":bundle["checkpoint_sha256"],
            "native_matched_sample_order_sha256":d["sample_order_sha256"],
            "original_CAL_DEV_replayed_with_declared_numeric_tolerances":True,
            "model_checkpoint_sha256_bitwise_verified":True,
            "quantile_absolute_tolerance_Wh":0.0002,
            "interval_width_absolute_tolerance_Wh":0.0004,
            "other_metric_relative_tolerance":1e-6,
            "trained_during_replay":False}

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--replay-from",type=Path)
    args=parser.parse_args(argv)
    try:
        if args.out.exists():raise ControlError("refuse to overwrite original original fit receipt")
        if not args.replay_from and os.environ.get("MODEL01_GRU_REAL_FIT_ENABLE")!="EXPLICIT_CPU_DEV_CONTROL":
            raise ControlError("explicit isolated FIT permission required")
        original,gap=preflight()
        source=tac.uci.fetch_official_zip()
        if tac.uci.digest(source)!=tac.SOURCE:
            raise ControlError("publisher real ZIP checksum mismatch")
        y=np.asarray(tac.uci.original_rows(source),dtype=np.float64)
        mu=float(y[:tac.uci.FIT_END].mean())
        std=float(y[:tac.uci.FIT_END].std())
        if std<=0 or not math.isfinite(std):raise ControlError("FIT-only normalization invalid")
        fit=tac.windows(y,range(tac.HISTORY-1,tac.uci.FIT_END-tac.uci.H),mu,std)
        cal=tac.windows(y,tac.uci.eligible_indices(tac.uci.FIT_END,tac.uci.CAL_END),mu,std)
        dev=tac.windows(y,tac.uci.eligible_indices(tac.uci.CAL_END,tac.uci.DEV_END),mu,std)
        torch.set_num_threads(2)
        natives={r["seed"]:r for r in original["runs"]}
        if args.replay_from:
            b=json.loads(args.replay_from.read_bytes())
            if (b.get("study")!=MODEL or b.get("source_zip_sha256")!=tac.SOURCE
                or b.get("gru_learned_parameters")!=EXPECTED_PARAMS or b.get("training_steps")!=160
                or len(b.get("runs",[]))!=3):
                raise ControlError("original GRU replay source mismatch")
            reports=[replay_seed(r,cal,dev,mu,std,natives[r["seed"]]) for r in b["runs"]]
            result={"study":MODEL,"mode":"ORIGINAL_GIT_CHECKPOINT_NO_REFIT_REPLAY",
                    "replay_reports":reports,
                    "scientific_tier":"EXPOSED_DEVELOPMENT_MATCHED_UPDATES_AND_PARAMS_ONLY",
                    "trained_during_replay":False}
        else:
            runs=[fit_seed(seed,fit,cal,dev,mu,std,natives[seed]) for seed in tac.SEEDS]
            result={"study":MODEL,"mode":"ORIGINAL_AUTHENTIC_UCI374_FIT_CAL_DEV",
                    "source_zip_sha256":tac.SOURCE,"source_doi":tac.uci.DOI,
                    "gru_learned_parameters":EXPECTED_PARAMS,
                    "native_learned_parameters":TARGET_PARAMS,
                    "relative_parameter_gap":gap,
                    "training_steps":tac.UPDATES,
                    "training_batch_size":tac.BATCH,
                    "training_usable_fit_only":True,
                    "real_training_samples_per_seed":tac.UPDATES*tac.BATCH,
                    "same_exact_source_minibatch_indices":True,
                    "same_optimizer_loss_lr_weight_decay_and_grad_clip":True,
                    "source_time_available_at_verified":False,
                    "protected_future_test_rows_parsed":False,
                    "training_FLOPs_matched":False,
                    "architectures_have_different_FLOPs":True,
                    "neural_architecture_superiority_claim":False,
                    "independent_evaluation":False,
                    "seeds":list(tac.SEEDS),
                    "runs":runs,
                    "scientific_tier":"REAL_REUSED_DEV_CONTROL_NOT_ARCHITECTURE_SUPERIORITY",
                    "warnings":[
                       "exact source, inputs, minibatches, updates, optimizer and parameter count match within 1%; FLOPs are not equal",
                       "original TAC was developed earlier against this reused DEV population; no new source confirmation",
                       "single household autocorrelated time-series windows are not distinct independent cases",
                       "no qualified probability interval reliability or physical/policy decision claim",
                       "memory module benefit is not established by comparing whole models",
                       "original native checkpoint and all model training artifacts must be in Git"
                    ]}
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
        if args.replay_from:
            summary={"study":MODEL,"replay_reports":result["replay_reports"],"trained_during_replay":False}
        else:
            summary={"study":MODEL,"status":result["scientific_tier"],
                "parameters":EXPECTED_PARAMS,"parameter_gap":gap,
                "training_FLOPs_matched":False,"scores":[
                    {"seed":x["seed"],"dev":x["dev"],
                     "checkpoint_sha256":x["checkpoint_sha256"]}
                    for x in result["runs"]]}
        print(json.dumps(summary,sort_keys=True,allow_nan=False))
        return 0
    except (ControlError,tac.StudyError,tac.uci.SourceError,OSError,KeyError,
            ValueError,TypeError,RuntimeError) as exc:
        print(json.dumps({"status":"BLOCKED_OR_FAILED","reason":str(exc)}),file=sys.stderr)
        return 2

if __name__=="__main__":
    sys.exit(main())
