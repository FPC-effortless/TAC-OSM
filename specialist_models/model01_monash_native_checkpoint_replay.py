#!/usr/bin/env python3
"""Clean original training checkpoint replay on real customers; NEVER refits."""
import argparse,base64,gzip,hashlib,io,json,math
from pathlib import Path
import torch
import model01_monash_native_multihorizon_fit as m

ROOT=Path(__file__).resolve().parents[1]

def verify(record_path):
    source_clients,source_receipt=m.source_data()
    data=json.loads(record_path.read_bytes())
    if (data.get("study")!=m.STUDY or
        data.get("source_zip_sha256")!=source_receipt["source_zip_sha256"] or
        data.get("native_learned_deployed_parameters")!=m.PARAMETERS or
        data.get("seeds")!=list(m.SEEDS) or
        data.get("sealed_test_target_values_parsed") is not False or
        len(data.get("runs",[]))!=len(m.SEEDS)):
        raise m.CapabilityError("original Git model source/identity mismatch")
    mean,std=m.fit_scaling(source_clients)
    if not math.isclose(mean,data["FIT_only_log1p_mean"],abs_tol=1e-10):
        raise m.CapabilityError("original FIT mean changed")
    if not math.isclose(std,data["FIT_only_log1p_std"],abs_tol=1e-10):
        raise m.CapabilityError("original FIT std changed")
    cal=m.eval_batches(source_clients,"CAL",mean,std,source_receipt)
    dev=m.eval_batches(source_clients,"DEV",mean,std,source_receipt)
    torch.set_num_threads(2)
    reports=[]
    def assert_same(a,b,where):
        if isinstance(a,dict):
            if not isinstance(b,dict) or set(a)!=set(b):
                raise m.CapabilityError("score field mismatch "+where)
            for k in a:assert_same(a[k],b[k],where+"/"+str(k))
        elif type(a) in (int,float) and type(b) in (int,float):
            if not math.isclose(a,b,rel_tol=3e-6,abs_tol=.005):
                raise m.CapabilityError(f"replay numeric source mismatch {where}")
        elif a!=b:
            raise m.CapabilityError("source identity/source outcome mismatched "+where)
    for r in data["runs"]:
        blob=gzip.decompress(base64.b64decode(r["checkpoint_b64_gzip"],validate=True))
        if hashlib.sha256(blob).hexdigest()!=r["checkpoint_sha256"]:
            raise m.CapabilityError("original model bytes SHA256 changed")
        state=torch.load(io.BytesIO(blob),map_location="cpu",weights_only=True)
        if (state["seed"]!=r["seed"] or state["steps"]!=m.UPDATES or
            state["sampler_sha256"]!=r["sampler_sha256"]):
            raise m.CapabilityError("seed/source sample-order changed")
        model=m.MultiHorizonTAC()
        if m.ancestral.params(model)!=m.PARAMETERS:
            raise m.CapabilityError("native learned model architecture changed")
        model.load_state_dict(state["model"],strict=True)
        optimizer=torch.optim.AdamW(model.parameters(),lr=m.ancestral.LR,weight_decay=.01)
        optimizer.load_state_dict(state["optimizer"])
        c,_=m.collect(model,cal,mean,std,r["cal_offsets90_kw"])
        d,per_client=m.collect(model,dev,mean,std,r["cal_offsets90_kw"])
        assert_same(c,r["cal_macro"],"CAL")
        assert_same(d,r["dev_macro"],"DEV")
        assert_same(per_client,r["dev_per_customer"],"DEV physical client scores")
        reports.append({"seed":r["seed"],"checkpoint_sha256":r["checkpoint_sha256"],
                        "original_weights_and_optimizer_SHA256_verified":True,
                        "original_sampler_seed_and_fit_scaler_verified":True,
                        "CAL_DEV_replayed_within_0p005_kw_absolute":True,
                        "trained_during_replay":False})
    return {"schema":"MODEL01-MONASH-NEW-NATIVE-CLEAN-REPLAY-v1",
            "study":m.STUDY,"source_zip_sha256":source_receipt["source_zip_sha256"],
            "training_steps_executed":0,
            "sealed_test_customer_targets_parsed":False,
            "independent_scientific_test_performed":False,
            "reports":reports}

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--from-original",dest="original",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise SystemExit("refusing replay overwrite")
    r=verify(args.original)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(r,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"schema":r["schema"],
                      "seeds":[x["seed"] for x in r["reports"]],
                      "no_fit":r["training_steps_executed"]==0,
                      "sealed_test_not_scored":True},sort_keys=True))
