#!/usr/bin/env python3
"""LANGMEMORY045A: strictly causal two-cheap-memory + gated verifier probe.

This is *not* TAC neural model training or a learned language model. It checks
whether complementarity is visible in actual publisher text before PNDS model
math, matched neural fitting or untouched confirmation are authorized.
"""
import argparse,hashlib,json,math,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from scripts.langlayer035_sampling import fit_sources

RECENT=64
BUCKETS=256
CONTEXT=3
MAX_OBSERVED=1536
START=128
STRIDE=4
VERIFY_MOD=5

def digest(data):return hashlib.sha256(data).hexdigest()

def bucket(key):
    h=2166136261
    for b in key:
        h=((h^b)*16777619)&0xffffffff
    return h%BUCKETS

def most_frequent(values):
    if not values:return None
    counts={}
    for v in values:counts[v]=counts.get(v,0)+1
    return min(counts,key=lambda v:(-counts[v],v))

def recent(raw,t,key):
    """Only already observed byte continuations inside the last 64 positions."""
    values=[raw[j] for j in range(max(CONTEXT,t-RECENT),t) if raw[j-CONTEXT:j]==key]
    return most_frequent(values),len(values)

def exact_history(raw,t,key):
    """Expensive exact scan, bounded to prior source bytes (not hidden labels)."""
    candidates=[raw[j] for j in range(CONTEXT,t) if raw[j-CONTEXT:j]==key]
    far=sum(raw[j-CONTEXT:j]==key for j in range(CONTEXT,max(CONTEXT,t-RECENT)))
    return most_frequent(candidates),len(candidates),far,t-CONTEXT

class BoundedSketch:
    """256 deterministic FNV buckets; Boyer-Moore candidate per hash bucket.

    Compact possible representation per bucket: uint8 candidate + uint16 vote
    + uint16 total = 5 bytes; capped source window length prevents overflow.
    Actual Python object overhead and source/verifier store are not counted as
    on-device model memory.
    """
    def __init__(self):
        self.candidate=[0]*BUCKETS
        self.vote=[0]*BUCKETS
        self.total=[0]*BUCKETS
        self.keys=[set() for _ in range(BUCKETS)] # diagnostic collision lens only
    def read(self,key):
        k=bucket(key)
        return self.candidate[k] if self.vote[k] else None
    def update(self,key,value):
        k=bucket(key);self.keys[k].add(key)
        if self.vote[k]==0:
            self.candidate[k]=value;self.vote[k]=1
        elif self.candidate[k]==value:self.vote[k]+=1
        else:self.vote[k]-=1
        self.total[k]+=1
    def collision_buckets(self):
        return sum(len(x)>1 for x in self.keys)

def sample(raw,owner):
    """Streaming prefix probe: never inspect or store y_t before prediction."""
    if not isinstance(raw,bytes):raise TypeError("original document must be raw bytes")
    n=min(len(raw),MAX_OBSERVED)
    sketch=BoundedSketch()
    arms={a:{"covered":0,"correct":0,"long_covered":0,"long_correct":0}
          for a in ("recent_A","long_sketch_B","cheap_fusion","gated_verify","always_verify")}
    cases=0;long_cases=0;verify_calls=0;compared=0;collisions_seen=0
    audit=hashlib.sha256()
    for t in range(CONTEXT,n):
        key=raw[t-CONTEXT:t]
        if t>=START and t%STRIDE==0:
            a,a_hits=recent(raw,t,key)
            b=sketch.read(key)
            v,v_hits,far,scan_cost=exact_history(raw,t,key)
            long_case=far>0
            gate=t%VERIFY_MOD==0 and (a is None or b is None or a!=b)
            fusion=a if a is not None else b
            gated=v if gate and v is not None else fusion
            candidates={"recent_A":a,"long_sketch_B":b,"cheap_fusion":fusion,
                        "gated_verify":gated,"always_verify":v if v is not None else fusion}
            if gate:verify_calls+=1;compared+=scan_cost
            for name,prediction in candidates.items():
                if prediction is not None:
                    arms[name]["covered"]+=1
                    arms[name]["correct"]+=int(prediction==raw[t])
                    if long_case:
                        arms[name]["long_covered"]+=1
                        arms[name]["long_correct"]+=int(prediction==raw[t])
            cases+=1;long_cases+=int(long_case)
            # Input identity, not outcome label. Caller cannot infer protected TEST.
            audit.update(owner.encode()+b":"+str(t).encode()+b":"+key)
        # The true next byte becomes part of history ONLY AFTER its evaluation.
        sketch.update(key,raw[t])
    return {"owner":owner,"source_sha256":digest(raw),"positions_tested":cases,
            "long_history_recoverable_positions":long_cases,"arms":arms,
            "verifier_calls":verify_calls,"verifier_fraction":verify_calls/max(1,cases),
            "comparison_positions_scanned":compared,
            "cheap_payload_bytes_compact_theoretical":RECENT+BUCKETS*5,
            "verifier_source_retained_bytes_at_limit":n,
            "sketch_collision_bucket_count":sketch.collision_buckets(),
            "case_identity_sha256":audit.hexdigest(),"document_bytes":len(raw)}

def aggregate(rows):
    names=("recent_A","long_sketch_B","cheap_fusion","gated_verify","always_verify")
    totals={k:{f:sum(r["arms"][k][f] for r in rows) for f in ("covered","correct","long_covered","long_correct")} for k in names}
    samples=sum(r["positions_tested"] for r in rows)
    long_samples=sum(r["long_history_recoverable_positions"] for r in rows)
    for name,data in totals.items():
        data["coverage"]=data["covered"]/max(1,samples)
        data["accuracy_given_coverage"]=data["correct"]/max(1,data["covered"])
        data["long_history_accuracy_given_coverage"]=data["long_correct"]/max(1,data["long_covered"])
        data["long_history_coverage"]=data["long_covered"]/max(1,long_samples)
    return {"positions":samples,"long_positions":long_samples,"arms":totals,
            "verified_fraction":sum(r["verifier_calls"] for r in rows)/max(1,samples),
            "verifier_source_comparisons":sum(r["comparison_positions_scanned"] for r in rows)}

def run():
    fit=fit_sources()
    if len(fit)!=14:raise ValueError("original publisher source owner count changed")
    rows=[sample(raw,owner) for owner,raw in sorted(fit.items())]
    a=aggregate(rows)
    if not a["positions"] or a["verified_fraction"]>0.20+1e-12:
        raise ValueError("empty or overbudget verification")
    return {"study":"LANGMEMORY045A","status":"ACTUAL_SOURCE_ONLY_MECHANISM_DIAGNOSTIC",
        "partition":"AUTHENTIC_PSF_CPYTHON_FIT_ONLY","documents":len(rows),
        "source_owners":rows,"summary":a,
        "geometry":{"context_bytes":CONTEXT,"recent_bytes":RECENT,"sketch_buckets":BUCKETS,
                    "window_bytes_cap":MAX_OBSERVED,"eval_start":START,"stride":STRIDE,
                    "verifier_gate":"t mod 5 == 0 and (A absent or B absent or A != B); <=20% of sampled queries"},
        "limitations":["NOT TAC neural model, trained predictor, source-grounded model benefit, calibrated NLL or memory winner",
            "narrow observed publisher FIT prefix only, deterministic first 1536 bytes/owner",
            "Sketch current candidate is approximate and collision-prone; diagnostic keys sets not production state",
            "Verifier accesses full source history O(T), must count this memory and scan cost",
            "No matched neural parameter/training compute, no untouched DEV/TEST generalization",
            "Future benchmark must measure error-complementarity, source-id retrieval and prediction-cost frontier"],
        "protected_TEST":"NOT_ACCESSED","gpu_minutes":0,"training_updates":0,
        "checkpoint_promotion":False,"independently_qualified":False}

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError("refuse to overwrite original evidence")
    result=run();a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"status":result["status"],"summary":result["summary"],
        "documents":result["documents"]},sort_keys=True),flush=True)

if __name__=="__main__":main()
