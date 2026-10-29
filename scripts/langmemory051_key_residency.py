#!/usr/bin/env python3
"""LANGMEMORY051 — source-causal key residency versus continuation entropy.

A no-fit original-publisher FIT decomposition of prior LANGMEMORY050 losses.
Oracle dictionaries below are explicitly UNBOUNDED diagnostics: they are
never passed into, or used by, fixed-bytearray predictors or their updates.
"""
from __future__ import annotations
import argparse,hashlib,json,math,sys,time
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from scripts.langlayer035_sampling import fit_sources
from scripts.langmemory045_source_probe import digest
from scripts.langmemory047_bounded import RawRing,FixedKeyTable
from scripts.langmemory050_multivalue import BoundedMultiContinuation,COHORTS

MODES=("mg2_way2_1344","mg3_way2_1344","mg2_way2_4096","mg3_way2_4096")
SLOTS={"k1_1344":160,"k2_1344":80,"k3_1344":64,
       "k1_4096":504,"k2_4096":252,"k3_4096":200}
# All candidate clocks remain within 8192, so no wrap or count saturation.

class NoEvictionMG:
    """Unbounded per-exact-key MG representation; DO NOT call a fixed-memory baseline."""
    def __init__(self,k):
        self.k=k;self.values={};self.decrements=0;self.updates=0
    def lookup(self,key):
        c=self.values.get(key)
        return min(c,key=lambda x:(-c[x],x)) if c else None
    def contains_key(self,key):return key in self.values
    def update(self,key,val):
        if key not in self.values:self.values[key]={}
        c=self.values[key]
        if val in c:c[val]+=1
        elif len(c)<self.k:c[val]=1
        else:
            self.decrements+=1
            for v in tuple(c):
                c[v]-=1
                if c[v]==0:del c[v]
        self.updates+=1

def new_arm():
    return {"total":0,"answered":0,"correct":0,"recoverable":0,
            "key_resident_recoverable":0,"key_missing_recoverable":0,
            "value_missing_given_resident":0,"key_present_but_abstained":0,
            "exact_match_correct_on_recoverable":0,"source_long_correct":0,
            "lookup_ns":0,"update_ns":0}
def record(stats,answer,truth,recoverable,key_present,long,elapsed):
    stats["total"]+=1;stats["lookup_ns"]+=elapsed
    if recoverable:
        stats["recoverable"]+=1
        if key_present:stats["key_resident_recoverable"]+=1
        else:stats["key_missing_recoverable"]+=1
    if key_present and answer is None:stats["key_present_but_abstained"]+=1
    if recoverable and key_present and answer is None:stats["value_missing_given_resident"]+=1
    if answer is not None:
        stats["answered"]+=1
        good=int(answer==truth)
        stats["correct"]+=good
        if recoverable:stats["exact_match_correct_on_recoverable"]+=good
        if long:stats["source_long_correct"]+=good

def prefix_entropy(hist):
    n=sum(hist.values())
    return -sum((c/n)*math.log2(c/n) for c in hist.values()) if n else 0.0

def one_owner(raw,owner,cap,stride):
    if not isinstance(raw,bytes):raise TypeError("original authentic source bytes required")
    length=min(len(raw),cap)
    raw64=RawRing(64)
    old={b:FixedKeyTable(b,2) for b in (1344,4096)}
    fixed={(b,k):BoundedMultiContinuation(b,k,2) for b in (1344,4096) for k in (2,3)}
    unlimited={k:NoEvictionMG(k) for k in (2,3)}
    # EVALUATION/ORACLE only: never accessible by fixed predictor.
    prefix_hist={}
    hist_last={}
    first_seen={}
    names=["exact_full_history","recent64","old_k1_1344","old_k1_4096"]
    names+=["mg_no_eviction_2","mg_no_eviction_3"]
    names+=["mg2_way2_1344","mg3_way2_1344","mg2_way2_4096","mg3_way2_4096"]
    arms={n:new_arm() for n in names}
    bins={"unique_context_queries":0,"repeat_context_queries":0,
          "repeat_with_one_value":0,"repeat_with_more_than_one_value":0,
          "repeat_with_three_plus_values":0,"older_than_64":0,
          "older_than_1024":0,"older_than_1344":0,"older_than_4096":0,
          "older_than_8192":0,"context_count_above_1":0,
          "oracle_most_common_correct":0,"oracle_latest_correct":0,
          "oracle_majority_better":0,"oracle_latest_better":0,
          "entropy_bits_sum_on_repeat_queries":0.0,
          "entropy_bits_max":0.0,"distinct_continuations_sum":0,
          "prior_key_count_sum":0,
          "source_positions":0}
    casehash=hashlib.sha256()
    for t in range(length):
        if t>=3:
            key=raw[t-3:t]
            if t>=128 and t%stride==0:
                # Only the EVALUATOR ever accesses the original label x[t].
                truth=raw[t]
                hist=prefix_hist.get(key)
                count=sum(hist.values()) if hist else 0
                recoverable=count>0
                distinct=len(hist) if hist else 0
                latest=hist and raw[hist_last[key]]
                long=key in first_seen and first_seen[key]<t-64
                bins["source_positions"]+=1
                casehash.update(owner.encode()+b":"+str(t).encode()+b":"+key)
                if not recoverable:bins["unique_context_queries"]+=1
                else:
                    bins["repeat_context_queries"]+=1
                    bins["context_count_above_1"]+=count>1
                    bins["prior_key_count_sum"]+=count
                    bins["distinct_continuations_sum"]+=distinct
                    bins["repeat_with_one_value"]+=distinct==1
                    bins["repeat_with_more_than_one_value"]+=distinct>1
                    bins["repeat_with_three_plus_values"]+=distinct>=3
                    h=prefix_entropy(hist)
                    bins["entropy_bits_sum_on_repeat_queries"]+=h
                    bins["entropy_bits_max"]=max(bins["entropy_bits_max"],h)
                    majority=min(hist,key=lambda v:(-hist[v],v))
                    good_majority=majority==truth
                    good_latest=latest==truth
                    bins["oracle_most_common_correct"]+=good_majority
                    bins["oracle_latest_correct"]+=good_latest
                    bins["oracle_majority_better"]+=good_majority and not good_latest
                    bins["oracle_latest_better"]+=good_latest and not good_majority
                if long:
                    bins["older_than_64"]+=1
                    delay=t-first_seen[key]
                    for boundary in (1024,1344,4096,8192):
                        bins[f"older_than_{boundary}"]+=delay>boundary
                now=time.perf_counter_ns()
                recent,_=raw64.lookup(key)
                recent_ns=time.perf_counter_ns()-now
                record(arms["recent64"],recent,truth,recoverable,recoverable and recent is not None,long,recent_ns)
                now=time.perf_counter_ns()
                full=min(hist,key=lambda v:(-hist[v],v)) if hist else None
                record(arms["exact_full_history"],full,truth,recoverable,recoverable,long,
                       time.perf_counter_ns()-now)
                for b,table in old.items():
                    now=time.perf_counter_ns()
                    found=table.find(key) is not None
                    answer=recent if recent is not None else table.read(key)
                    record(arms[f"old_k1_{b}"],answer,truth,recoverable,found,long,
                           time.perf_counter_ns()-now+recent_ns)
                for k,oracle in unlimited.items():
                    now=time.perf_counter_ns()
                    found=oracle.contains_key(key)
                    candidate=oracle.lookup(key)
                    answer=recent if recent is not None else candidate
                    record(arms[f"mg_no_eviction_{k}"],answer,truth,recoverable,found,long,
                           time.perf_counter_ns()-now+recent_ns)
                for (b,k),table in fixed.items():
                    name=f"mg{k}_way2_{b}"
                    now=time.perf_counter_ns()
                    # IMPORTANT: source key residency is queried before fallback.
                    found=table._key_offset(key) is not None
                    candidate=table.lookup(key)
                    answer=recent if recent is not None else candidate
                    record(arms[name],answer,truth,recoverable,found,long,
                           time.perf_counter_ns()-now+recent_ns)
            first_seen.setdefault(key,t)
            tick=time.perf_counter_ns()
            prefix_hist.setdefault(key,Counter())[raw[t]]+=1
            hist_last[key]=t
            arms["exact_full_history"]["update_ns"]+=time.perf_counter_ns()-tick
            for b,table in old.items():
                tick=time.perf_counter_ns();table.update(key,raw[t],t)
                arms[f"old_k1_{b}"]["update_ns"]+=time.perf_counter_ns()-tick
            for k,oracle in unlimited.items():
                tick=time.perf_counter_ns();oracle.update(key,raw[t])
                arms[f"mg_no_eviction_{k}"]["update_ns"]+=time.perf_counter_ns()-tick
            for (b,k),table in fixed.items():
                tick=time.perf_counter_ns();table.observe(key,raw[t],t)
                arms[f"mg{k}_way2_{b}"]["update_ns"]+=time.perf_counter_ns()-tick
        tick=time.perf_counter_ns()
        raw64.observe(raw[t])
        delta=time.perf_counter_ns()-tick
        arms["recent64"]["update_ns"]+=delta
        for n in names:
            if n not in ("exact_full_history","recent64"):
                arms[n]["update_ns"]+=delta
    if any(row["total"]!=bins["source_positions"] for row in arms.values()):
        raise ValueError("causal source positions or arm alignment changed")
    diag={}
    for b,k in fixed:
        table=fixed[(b,k)]
        diag[f"mg{k}_way2_{b}"]={"hash_evictions":table.evictions,
            "mg_decrements":table.lossy_updates,
            "fixed_payload_bytes":table.effective_state_payload_bytes(),
            "occupied_at_end":table.occupied(),"slots":table.slots}
    # Upper-bound source model is unbounded and NOT a deployable fixed-key cache.
    return {"owner":owner,"source_sha256":digest(raw),"prefix_bytes":length,
        "case_identity_sha256":casehash.hexdigest(),"source_analysis":bins,"arms":arms,
        "fixed_diagnostics":diag,
        "unbounded_diagnostics":{"distinct_keys":len(prefix_hist),
            "total_observations":sum(sum(c.values()) for c in prefix_hist.values()),
            "unlimited_k2_decrements":unlimited[2].decrements,
            "unlimited_k3_decrements":unlimited[3].decrements}}

def run():
    source=fit_sources()
    if len(source)!=14:raise ValueError("original owner cohort changed")
    cohorts={}
    for title,cap,stride in COHORTS:
        rows=[one_owner(raw,owner,cap,stride) for owner,raw in sorted(source.items())]
        aggregate={}
        for row in rows:
            for k,v in row["source_analysis"].items():aggregate[k]=aggregate.get(k,0)+v
        totals={}
        for name in rows[0]["arms"]:
            d={key:sum(r["arms"][name][key] for r in rows)
               for key in rows[0]["arms"][name]}
            d["accuracy_all_positions"]=d["correct"]/aggregate["source_positions"]
            d["coverage"]=d["answered"]/aggregate["source_positions"]
            d["key_residency_given_earlier_key"]=d["key_resident_recoverable"]/max(1,aggregate["repeat_context_queries"])
            d["lookup_ms"]=d.pop("lookup_ns")/1e6
            d["update_ms"]=d.pop("update_ns")/1e6
            totals[name]=d
        if title=="anchor1536":
            if aggregate["source_positions"]!=4928 or totals["old_k1_1344"]["correct"]!=781:
                raise ValueError("original 050 anchor/source prediction drift")
            if totals["exact_full_history"]["correct"]!=1465:
                raise ValueError("original exact-history reference changed")
        if title=="extension4096" and aggregate["source_positions"]!=13245:
            raise ValueError("extended original source position drift")
        if title=="horizon8192" and aggregate["source_positions"]!=11784:
            raise ValueError("long original source position drift")
        cohorts[title]={"summary":{"source_analysis":aggregate,"arms":totals,
            "unlimited_k_decrements":{str(k):sum(r["unbounded_diagnostics"][f"unlimited_k{k}_decrements"] for r in rows)
                                      for k in (2,3)},
            "fixed_hash_evictions":{name:sum(r["fixed_diagnostics"][name]["hash_evictions"] for r in rows)
                                      for name in rows[0]["fixed_diagnostics"]}},
            "owners":rows}
    return {"study":"LANGMEMORY051",
        "status":"ACTUAL_PUBLISHER_FIT_INTERNAL_CAUSAL_RESIDENCY_DIAGNOSTIC",
        "cohorts":cohorts,"source":"original 14 legally published CPython3.13.0 FIT owners, already exposed",
        "fixed_record_capacity":SLOTS,
        "limits":["No fresh previously untouched publisher owner, no neural model or trained checkpoint",
                  "No FIT/DEV neural calibration, protected TEST or GPU",
                  "Unbounded oracle K2/K3 per-key dictionaries cannot be called fixed-memory alternatives",
                  "Original frequency/entropy summary uses only prior actual observations at query time; current x[t] is score only",
                  "No causal proof that key loss alone explains degradation; key and value failures interact",
                  "Bytearray payload excludes Python objects/allocator; CPU measured source-only on GitHub runner",
                  "No mobile/VRAM or GRAPH011 recomputation benchmark"],
        "training_updates":0,"gpu_minutes":0,"protected_TEST":"NOT_ACCESSED",
        "qualified":False,"checkpoint_promotion":False}
def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError("preserve original receipt")
    out=run();args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"study":out["study"],"status":out["status"],
        "cohorts":{k:v["summary"] for k,v in out["cohorts"].items()}},sort_keys=True))

if __name__=="__main__":
    main()
