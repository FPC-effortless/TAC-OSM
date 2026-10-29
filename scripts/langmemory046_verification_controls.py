#!/usr/bin/env python3
"""LANGMEMORY046: honest matched-call verification and exact-index controls.

Same authentic publisher FIT histories as 045A, no trained model, no neural
performance metric. Every prediction uses only strictly earlier observations.
Precommitted random seeds; per-owner K is retrospectively matched using only
prediction-time gate decisions (NOT outcome correctness). This matching is a
diagnostic, not a deployable random policy.
"""
from __future__ import annotations
import argparse,hashlib,json,math,statistics,sys,time
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from scripts.langlayer035_sampling import fit_sources
from scripts.langmemory045_source_probe import (BoundedSketch,RECENT,CONTEXT,START,
    MAX_OBSERVED,STRIDE,VERIFY_MOD,exact_history,recent,most_frequent,digest)

RANDOM_SEEDS=(11,29,47,83,131)
POLICIES=("cheap","select_scan","select_index","random_all",
          "random_uncertain","periodic","always_index")
# Neither gate construction nor random ranking can depend on current target.
def random_rank(seed,owner,t):
    return hashlib.sha256(f"LANGMEMORY046|{seed}|{owner}|{t}".encode()).digest()

class ExactContextIndex:
    """Incremental per-owner context -> previous continuation counts.

    Full historical information is stored (growing state), never confused with
    TAC's fixed-size neural recurrence. Used as strong non-neural comparator.
    """
    def __init__(self):
        self.counts={}
        self.transitions=0
    def read(self,key):
        values=self.counts.get(key)
        if not values:return None,0
        return min(values,key=lambda v:(-values[v],v)),sum(values.values())
    def observe(self,key,value):
        if not isinstance(key,bytes) or len(key)!=CONTEXT or not (0<=value<=255):
            raise ValueError("bad causal source update")
        values=self.counts.setdefault(key,Counter())
        values[value]+=1
        self.transitions+=1
    def packed_lower_bound(self):
        # 3 bytes per key, 1 byte observed continuation, 4-byte count. No
        # hash table capacity, Python dict/object, allocator or runtime.
        return sum(CONTEXT+len(v)*5 for v in self.counts.values())
    def python_heap_bytes(self):
        # Shallow graph walk, no double-count of alias objects.
        seen=set()
        def traverse(obj):
            pointer=id(obj)
            if pointer in seen:return 0
            seen.add(pointer);total=sys.getsizeof(obj)
            if isinstance(obj,dict):
                for key,value in obj.items():total+=traverse(key)+traverse(value)
            elif isinstance(obj,(list,tuple,set)):
                for value in obj:total+=traverse(value)
            return total
        return traverse(self.counts)
    def distinct_contexts(self):return len(self.counts)

def record_case(raw,t,owner,a,b,exact,far,scan_cost,idx_value,idx_count):
    if exact!=idx_value:raise ValueError("causal incremental index mismatch with naive full prior history")
    cheap=a if a is not None else b
    uncertain=a is None or b is None or a!=b
    eligible=(t%VERIFY_MOD==0 and uncertain)
    return {"owner":owner,"position":t,"cheap":cheap,"full":exact,
            "truth":raw[t],"uncertain":uncertain,"periodic":t%VERIFY_MOD==0,
            "select":eligible,"long":far>0,"scan_work":scan_cost,
            "prior_exact_matches":idx_count}

def collect(raw,owner):
    """Evaluate a document online; B/index updates are AFTER scoring raw[t]."""
    if not isinstance(raw,bytes):raise TypeError("raw original source bytes required")
    n=min(len(raw),MAX_OBSERVED)
    sketch=BoundedSketch();index=ExactContextIndex()
    cases=[];index_update_ns=0;index_read_ns=0;scan_ns=0
    for t in range(CONTEXT,n):
        key=raw[t-CONTEXT:t]
        if t>=START and t%STRIDE==0:
            a,_=recent(raw,t,key)
            b=sketch.read(key)
            tick=time.perf_counter_ns()
            iv,count=index.read(key)
            index_read_ns+=time.perf_counter_ns()-tick
            tick=time.perf_counter_ns()
            v,_,far,cost=exact_history(raw,t,key)
            scan_ns+=time.perf_counter_ns()-tick
            cases.append(record_case(raw,t,owner,a,b,v,far,cost,iv,count))
        tick=time.perf_counter_ns()
        index.observe(key,raw[t])
        index_update_ns+=time.perf_counter_ns()-tick
        sketch.update(key,raw[t])
    return cases,{"owner":owner,"full_original_source_sha256":digest(raw),
        "bytes_processed":n,"indexed_transitions":index.transitions,
        "distinct_exact_contexts":index.distinct_contexts(),
        "exact_index_compact_lower_bound_bytes":index.packed_lower_bound(),
        "exact_index_python_heap_bytes":index.python_heap_bytes(),
        "raw_source_bytes_for_naive_scan":n,
        "index_read_timed_ns_all_queries":index_read_ns,
        "index_update_timed_ns_all_observations":index_update_ns,
        "naive_full_history_timed_ns_all_queries":scan_ns,
        "case_identity_sha256":hashlib.sha256(b"".join(
            f"{owner}:{r['position']}:{r['prior_exact_matches']};".encode() for r in cases)).hexdigest(),
        "cases":len(cases)}

def selections(cases,owner,seed):
    selected={i for i,r in enumerate(cases) if r["select"]}
    uncertain=[i for i,r in enumerate(cases) if r["uncertain"]]
    if len(uncertain)<len(selected):
        raise ValueError("insufficient uncertainty-matched candidates")
    def choose(indices,k):
        return set(sorted(indices,key=lambda i:(random_rank(seed,owner,cases[i]["position"]),i))[:k])
    K=len(selected)
    return {"cheap":set(),"select_scan":selected,"select_index":selected,
            "random_all":choose(range(len(cases)),K),
            "random_uncertain":choose(uncertain,K),
            "periodic":{i for i,r in enumerate(cases) if r["periodic"]},
            "always_index":set(range(len(cases)))}

def evaluate(cases,sets):
    report={}
    total=len(cases);long_total=sum(x["long"] for x in cases)
    for name,selected in sets.items():
        count={"answered":0,"correct":0,"long_answered":0,"long_correct":0,
               "gain_from_cheap":0,"repairs":0,"regressions":0,
               "verification_calls":len(selected),"scan_positions":0,
               "index_lookups":0}
        for i,r in enumerate(cases):
            predict=r["full"] if i in selected and r["full"] is not None else r["cheap"]
            if predict is not None:
                count["answered"]+=1;count["correct"]+=predict==r["truth"]
                if r["long"]:
                    count["long_answered"]+=1
                    count["long_correct"]+=predict==r["truth"]
            if i in selected:
                if name in ("select_index","always_index"):
                    count["index_lookups"]+=1
                else:
                    count["scan_positions"]+=r["scan_work"]
                before=r["cheap"]==r["truth"]
                after=predict==r["truth"]
                if not before and after:count["repairs"]+=1
                if before and not after:count["regressions"]+=1
        count["gain_from_cheap"]=count["repairs"]-count["regressions"]
        count["coverage"]=count["answered"]/max(1,total)
        count["success_per_all_positions"]=count["correct"]/max(1,total)
        count["accuracy_given_coverage"]=count["correct"]/max(1,count["answered"])
        count["long_coverage"]=count["long_answered"]/max(1,long_total)
        count["gain_per_100_queries"]=100*count["gain_from_cheap"]/max(1,total)
        report[name]=count
    if report["select_index"]["correct"]!=report["select_scan"]["correct"] or report["select_index"]["answered"]!=report["select_scan"]["answered"]:
        raise ValueError("index exact-source verifier changed results")
    return report

def summarize(rows):
    result={}
    for name in POLICIES:
        agg={k:sum(r["arms"][name][k] for r in rows) for k in
             ("answered","correct","long_answered","long_correct","gain_from_cheap",
              "repairs","regressions","verification_calls","scan_positions","index_lookups")}
        total=sum(r["cases"] for r in rows)
        long_total=sum(r["long_cases"] for r in rows)
        agg["coverage"]=agg["answered"]/max(1,total)
        agg["success_per_all_positions"]=agg["correct"]/max(1,total)
        agg["accuracy_given_coverage"]=agg["correct"]/max(1,agg["answered"])
        agg["long_coverage"]=agg["long_answered"]/max(1,long_total)
        agg["calls_fraction"]=agg["verification_calls"]/max(1,total)
        result[name]=agg
    return result

def run():
    fit=fit_sources()
    if len(fit)!=14:raise ValueError("immutable original publisher FIT cohort unexpected")
    source_rows=[];cached=[]
    for owner,raw in sorted(fit.items()):
        cases,meta=collect(raw,owner)
        source_rows.append(meta);cached.append((owner,cases))
    if not all(x[1] for x in cached):raise ValueError("incomplete source owner case cohort")
    by_seed={}
    for seed in RANDOM_SEEDS:
        doc_rows=[]
        for owner,cases in cached:
            masks=selections(cases,owner,seed)
            assert len(masks["select_scan"])==len(masks["random_all"])==len(masks["random_uncertain"])
            doc_rows.append({"owner":owner,"cases":len(cases),
                "long_cases":sum(x["long"] for x in cases),
                "arms":evaluate(cases,masks)})
        summary=summarize(doc_rows)
        for name in ("random_all","random_uncertain"):
            if summary[name]["verification_calls"]!=summary["select_scan"]["verification_calls"]:
                raise ValueError("random verification not call-matched")
        for name in ("select_scan","select_index"):
            if summary[name]["verification_calls"]!=summary["random_all"]["verification_calls"]:
                raise ValueError("call budget mismatch")
        by_seed[str(seed)]={"summary":summary,"owner_outcomes":doc_rows}
    base=by_seed[str(RANDOM_SEEDS[0])]["summary"]
    if (base["select_scan"]["correct"]!=base["select_index"]["correct"]
          or not base["select_scan"]["verification_calls"] or
          base["select_scan"]["verification_calls"]/sum(r["cases"] for r in by_seed[str(RANDOM_SEEDS[0])]["owner_outcomes"])>.2):
        raise ValueError("index/select mismatch or verification quota exceeded")
    costs={"exact_index_total_lower_bound_bytes":sum(r["exact_index_compact_lower_bound_bytes"] for r in source_rows),
           "exact_index_total_python_heap_bytes":sum(r["exact_index_python_heap_bytes"] for r in source_rows),
           "original_source_prefix_bytes_total":sum(r["raw_source_bytes_for_naive_scan"] for r in source_rows),
           "actual_index_update_timed_ms":sum(r["index_update_timed_ns_all_observations"] for r in source_rows)/1e6,
           "actual_index_lookup_timed_ms_all_queried":sum(r["index_read_timed_ns_all_queries"] for r in source_rows)/1e6,
           "actual_naive_full_history_scan_timed_ms_all_queried":sum(r["naive_full_history_timed_ns_all_queries"] for r in source_rows)/1e6,
           "timing_note":"Python per-query wall time, all 4928 queries, includes measurement overhead, not same-host paired model latency or production RAM"}
    return {"study":"LANGMEMORY046","status":"MEASURED_GENUINE_FIT_SOURCE_MATCHED_CALL_VERIFIER_CONTROLS",
        "source_partition":"AUTHENTIC_PSF_CPYTHON_FIT_ONLY","documents":len(cached),
        "cases":sum(len(v) for _,v in cached),"source_receipts":source_rows,
        "random_seeds":RANDOM_SEEDS,"runs":by_seed,
        "costs":costs,
        "protocol":{"query_context":3,"recent_window":64,"hash_buckets":256,
            "source_prefix_limit":1536,"evaluation_start":128,"stride":4,
            "select_gate":"t % 5 == 0 AND cheap A/B missing/disagreement",
            "matched_random":"within each owner choose exactly K=select calls by SHA256(seed, owner, t) without target correctness",
            "random_uncertain":"same K sampled only among missing/disagreement, no label",
            "random_control_caveat":"K retrospectively determined by prefix-based gate choices, not deployable online selection",
            "index_is_full_history":"per-owner exact context->continuation counters maintained after every observed byte; grows with history",
            "absolute_position_rule":"target byte not indexed/scanned until AFTER scoring at its original position"},
        "negative_scope":["No trained TAC/Transformer/SSM neural evaluation or neural next-byte NLL",
            "Source-only repeated publisher FIT and same first 1536 bytes previously used by 045A",
            "No independent untouched holdout; random selection controls posthoc budget but never target labels",
            "Compact index lower bound excludes hash allocator; Python heap is interpreter-specific",
            "Index saves scan cost at expense of growing memory and per-observed-token update",
            "Interpreting gains needs equal work/byte/storage controls, not call count alone",
            "No claim of mobile power, novel persistent intelligence, or architecture superiority",
            "No model weights, GPU, protected TEST, or checkpoint promotion"],
        "training_updates":0,"gpu_minutes":0,"protected_TEST":"NOT_ACCESSED",
        "independently_qualified":False,"checkpoint_promotion":False}

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError("preserve original results")
    result=run();args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({"study":result["study"],"status":result["status"],
        "cases":result["cases"],"random_seeds":RANDOM_SEEDS,
        "main_seed_summary":result["runs"][str(RANDOM_SEEDS[0])]["summary"],
        "index_costs":result["costs"]},sort_keys=True),flush=True)

if __name__=="__main__":main()
