#!/usr/bin/env python3
"""LANGMEMORY050: bounded exact-key, multi-continuation Misra-Gries source probe.

Internal authentic publisher FIT diagnostic only. Candidate counts are not
the exact n-gram histogram after decrement or key eviction. Never treat this
as a trained TAC neural model or unseen-source language NLL.
"""
from __future__ import annotations
import argparse,hashlib,json,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from scripts.langlayer035_sampling import fit_sources
from scripts.langmemory045_source_probe import BoundedSketch,recent,digest
from scripts.langmemory046_verification_controls import ExactContextIndex
from scripts.langmemory047_bounded import RawRing,FixedKeyTable,fnv

BUDGETS=(1344,4096)
VALUES=(2,3)
WAYS=(1,2)
CONTEXT=3
RECENT_BYTES=64
START=128
MAX_SOURCE=8192
COHORTS=(("anchor1536",1536,4),("extension4096",4096,4),
         ("horizon8192",8192,8))

class BoundedMultiContinuation:
    """One truly fixed-size bytearray holding exact keys and 2/3 value counters.

    Each record is 16 or 20 bytes. Key:3, valid:1, K*(value:1,count:uint16),
    last_observed_source_clock:uint16, rest padding. No unbounded dict,
    candidate cache, original-source history or per-key object inside read.
    Extra 64B of the total budget is reserved for the shared recent raw ring.
    """
    def __init__(self,budget,k,ways):
        if budget not in BUDGETS or k not in VALUES or ways not in WAYS:
            raise ValueError("unregistered memory geometry")
        self.budget=budget;self.k=k;self.ways=ways
        self.width=4+3*k+2+ (4 if k==2 else 5)
        assert self.width == (16 if k==2 else 20)
        self.payload=bytearray(budget-RECENT_BYTES)
        self.slots=(len(self.payload)//self.width)//ways*ways
        self.sets=self.slots//ways
        self.used_bytes=self.slots*self.width
        self.evictions=0
        self.lossy_updates=0
        self.key_hits=0
        self.key_misses=0
        self.update_probes=0
        self.read_probes=0
        self.writes=0
    def offsets(self,key):
        if not isinstance(key,bytes) or len(key)!=CONTEXT:
            raise ValueError("not a genuine 3-byte source context")
        start=(fnv(key)%self.sets)*self.ways*self.width
        return tuple(start+j*self.width for j in range(self.ways))
    def time(self,off):
        return int.from_bytes(self.payload[off+4+3*self.k:off+6+3*self.k],"little")
    def count(self,off,i):
        o=off+4+3*i
        return int.from_bytes(self.payload[o+1:o+3],"little")
    def value(self,off,i):
        return self.payload[off+4+3*i]
    def _write_counter(self,off,i,value,count):
        x=off+4+3*i
        self.payload[x]=value
        self.payload[x+1:x+3]=count.to_bytes(2,"little")
    def _zero(self,off):
        self.payload[off:off+self.width]=bytes(self.width)
    def _key_offset(self,key,probe_write=False):
        probes=0
        for off in self.offsets(key):
            probes+=1
            if self.payload[off+3] and self.payload[off:off+3]==key:
                if probe_write:self.update_probes+=probes
                else:self.read_probes+=probes
                return off
        if probe_write:self.update_probes+=probes
        else:self.read_probes+=probes
        return None
    def lookup(self,key):
        off=self._key_offset(key)
        if off is None:
            self.key_misses+=1
            return None
        self.key_hits+=1
        candidates=[(self.count(off,i),self.value(off,i)) for i in range(self.k)
                    if self.count(off,i)>0]
        if not candidates:return None
        n,val=min(candidates,key=lambda item:(-item[0],item[1]))
        return val
    def observe(self,key,value,clock):
        if not (isinstance(value,int) and 0<=value<=255 and 0<=clock<65535):
            raise ValueError("invalid observed source byte or 16-bit clock")
        off=self._key_offset(key,probe_write=True)
        if off is None:
            offsets=self.offsets(key)
            off=next((a for a in offsets if not self.payload[a+3]),None)
            if off is None:
                off=min(offsets,key=lambda a:(self.time(a),a))
                self.evictions+=1
            self._zero(off)
            self.payload[off:off+3]=key
            self.payload[off+3]=1
            self._write_counter(off,0,value,1)
        else:
            candidate=next((i for i in range(self.k)
                            if self.count(off,i)>0 and self.value(off,i)==value),None)
            if candidate is not None:
                n=self.count(off,candidate)
                self._write_counter(off,candidate,value,min(65535,n+1))
            else:
                free=next((i for i in range(self.k) if self.count(off,i)==0),None)
                if free is not None:
                    self._write_counter(off,free,value,1)
                else:
                    # Misra-Gries decrement: no new candidate is inserted.
                    # Every full K-value context loses 1 count per candidate.
                    self.lossy_updates+=1
                    for i in range(self.k):
                        self._write_counter(off,i,self.value(off,i),self.count(off,i)-1)
        self.payload[off+4+3*self.k:off+6+3*self.k]=clock.to_bytes(2,"little")
        self.writes+=1
    def occupied(self):
        return sum(self.payload[j+3]!=0 for j in range(0,self.used_bytes,self.width))
    def effective_state_payload_bytes(self):
        return RECENT_BYTES+len(self.payload)

def empty():
    return {"queries":0,"covered":0,"correct":0,"long_covered":0,
            "long_correct":0,"lookup_ns":0,"update_ns":0,"work":0}
def record(s,answer,label,long_case,work,elapsed):
    s["queries"]+=1;s["work"]+=work;s["lookup_ns"]+=elapsed
    if answer is not None:
        s["covered"]+=1;s["correct"]+=int(answer==label)
        if long_case:
            s["long_covered"]+=1;s["long_correct"]+=int(answer==label)

def arm_name(b,k,way):
    return f"mg{k}_way{way}_{b}"

def owner_probe(raw,owner,cap,stride):
    if not isinstance(raw,bytes):raise TypeError("original publisher bytes required")
    length=min(cap,len(raw))
    recent_ring=RawRing(RECENT_BYTES)
    exact=ExactContextIndex()
    legacy_sketch=BoundedSketch()
    old={b:FixedKeyTable(b,2) for b in BUDGETS}
    rings={b:RawRing(b) for b in BUDGETS}
    systems={(b,k,w):BoundedMultiContinuation(b,k,w)
             for b in BUDGETS for k in VALUES for w in WAYS}
    names=["legacy_a_plus_b","exact_unbounded",
           "recent64"]+[f"old_table2_{b}" for b in BUDGETS]+[
           f"raw_ring_{b}" for b in BUDGETS]+[
           arm_name(*geo) for geo in systems]
    metrics={name:empty() for name in names}
    prior_first={}
    cases=long_cases=0
    ids=hashlib.sha256()
    for t in range(length):
        if t>=CONTEXT:
            key=raw[t-CONTEXT:t]
            if t>=START and t%stride==0:
                label=raw[t] # EVALUATOR ONLY, AFTER all states reflect x[0:t]
                long_case=key in prior_first and prior_first[key]<t-RECENT_BYTES
                cases+=1;long_cases+=int(long_case)
                ids.update(owner.encode()+b":"+str(t).encode()+b":"+key)
                tick=time.perf_counter_ns()
                a,work=recent_ring.lookup(key)
                a_ns=time.perf_counter_ns()-tick
                record(metrics["recent64"],a,label,long_case,work,a_ns)
                tick=time.perf_counter_ns()
                olda,_=recent(raw,t,key)
                leg=olda if olda is not None else legacy_sketch.read(key)
                record(metrics["legacy_a_plus_b"],leg,label,long_case,1,
                       time.perf_counter_ns()-tick)
                tick=time.perf_counter_ns()
                target,_=exact.read(key)
                record(metrics["exact_unbounded"],target,label,long_case,1,
                       time.perf_counter_ns()-tick)
                for b,ring in rings.items():
                    tick=time.perf_counter_ns()
                    answer,work=ring.lookup(key)
                    record(metrics[f"raw_ring_{b}"],answer,label,long_case,work,
                           time.perf_counter_ns()-tick)
                for b,model in old.items():
                    tick=time.perf_counter_ns()
                    answer=a if a is not None else model.read(key)
                    record(metrics[f"old_table2_{b}"],answer,label,long_case,2,
                           time.perf_counter_ns()-tick+a_ns)
                for geo,model in systems.items():
                    tick=time.perf_counter_ns()
                    candidate=model.lookup(key)
                    answer=a if a is not None else candidate
                    record(metrics[arm_name(*geo)],answer,label,long_case,
                           geo[-1]*geo[1],time.perf_counter_ns()-tick+a_ns)
            prior_first.setdefault(key,t)
            # Observed byte inserted into memory ONLY after its prediction.
            tick=time.perf_counter_ns()
            exact.observe(key,raw[t])
            metrics["exact_unbounded"]["update_ns"]+=time.perf_counter_ns()-tick
            tick=time.perf_counter_ns()
            legacy_sketch.update(key,raw[t])
            metrics["legacy_a_plus_b"]["update_ns"]+=time.perf_counter_ns()-tick
            for b,model in old.items():
                tick=time.perf_counter_ns();model.update(key,raw[t],t)
                metrics[f"old_table2_{b}"]["update_ns"]+=time.perf_counter_ns()-tick
            for geo,model in systems.items():
                tick=time.perf_counter_ns();model.observe(key,raw[t],t)
                metrics[arm_name(*geo)]["update_ns"]+=time.perf_counter_ns()-tick
        tick=time.perf_counter_ns();recent_ring.observe(raw[t])
        elapsed=time.perf_counter_ns()-tick
        metrics["recent64"]["update_ns"]+=elapsed
        for b in BUDGETS:metrics[f"old_table2_{b}"]["update_ns"]+=elapsed
        for geo in systems:metrics[arm_name(*geo)]["update_ns"]+=elapsed
        for b,ring in rings.items():
            tick=time.perf_counter_ns();ring.observe(raw[t])
            metrics[f"raw_ring_{b}"]["update_ns"]+=time.perf_counter_ns()-tick
    if cases==0:raise ValueError("no original event candidates")
    if any(s["queries"]!=cases for s in metrics.values()):
        raise ValueError("incomplete source aligned evaluations")
    if any(model.effective_state_payload_bytes()!=geo[0] for geo,model in systems.items()):
        raise ValueError("bypassed fixed byte budget")
    return {"owner":owner,"source_sha256":digest(raw),"limit":length,"cases":cases,
            "genuine_far_context_cases":long_cases,"case_ids_sha256":ids.hexdigest(),
            "arms":metrics,
            "multi_diagnostics":{arm_name(*geo):{
                "total_payload_bytes":model.effective_state_payload_bytes(),
                "index_bytes":len(model.payload),"reserved_recent_bytes":64,
                "record_width":model.width,"value_slots_per_key":model.k,
                "record_slots":model.slots,"unused_index_padding_bytes":len(model.payload)-model.used_bytes,
                "occupied":model.occupied(),"hash_evictions":model.evictions,
                "misra_gries_decrement_events":model.lossy_updates,
                "key_hits":model.key_hits,"key_misses":model.key_misses,
                "update_slot_probes":model.update_probes,"read_slot_probes":model.read_probes,
                "source_observation_updates":model.writes,"bytearray_python_heap":sys.getsizeof(model.payload)}
                for geo,model in systems.items()},
            "unbounded_exact_index":{"compact_lower_bound_bytes":exact.packed_lower_bound(),
                "python_heap_bytes":exact.python_heap_bytes()}}

def aggregate(rows):
    cases=sum(r["cases"] for r in rows)
    far=sum(r["genuine_far_context_cases"] for r in rows)
    arms={}
    for name in rows[0]["arms"]:
        x={k:sum(r["arms"][name][k] for r in rows) for k in
           ("queries","covered","correct","long_covered","long_correct","work",
            "lookup_ns","update_ns")}
        x["success_all_positions"]=x["correct"]/cases
        x["coverage"]=x["covered"]/cases
        x["far_coverage"]=x["long_covered"]/max(1,far)
        x["far_success_all_far_positions"]=x["long_correct"]/max(1,far)
        x["total_lookup_ms"]=x.pop("lookup_ns")/1e6
        x["total_update_ms"]=x.pop("update_ns")/1e6
        arms[name]=x
    return {"cases":cases,"genuine_far_context_cases":far,"arms":arms,
        "multi_diagnostics":{name:{
            k:sum(r["multi_diagnostics"][name][k] for r in rows)
            for k in ("hash_evictions","misra_gries_decrement_events",
                      "key_hits","key_misses","update_slot_probes","read_slot_probes",
                      "source_observation_updates")}
            for name in rows[0]["multi_diagnostics"]}}

def run():
    fit=fit_sources()
    if len(fit)!=14:raise ValueError("original publisher source count changed")
    cohorts={}
    for name,cap,stride in COHORTS:
        rows=[owner_probe(raw,owner,cap,stride) for owner,raw in sorted(fit.items())]
        a=aggregate(rows)
        # Retain original benchmark anchors from separate 047 source code.
        if name=="anchor1536":
            if a["cases"]!=4928 or a["arms"]["legacy_a_plus_b"]["correct"]!=933:
                raise ValueError("frozen original 045A source anchor mismatched")
            if a["arms"]["raw_ring_1344"]["correct"]!=1461:
                raise ValueError("frozen original 047 raw-ring anchor changed")
            if a["arms"]["old_table2_1344"]["correct"]!=781:
                raise ValueError("frozen original 047 two-way table changed")
        if name=="extension4096":
            if a["cases"]!=13245 or a["arms"]["raw_ring_1344"]["correct"]!=4503:
                raise ValueError("original 047 longer FIT source anchor changed")
        if name=="horizon8192":
            if a["cases"]!=11784 or a["arms"]["raw_ring_1344"]["correct"]!=4152:
                raise ValueError("original 048 long-prefix raw comparator changed")
        cohorts[name]={"summary":a,"source_owners":rows}
    return {"study":"LANGMEMORY050",
        "status":"INTERNAL_AUTHENTIC_SOURCE_BOUNDED_MULTI_CONTINUATION_DIAGNOSTIC",
        "cohorts":cohorts,"budgets_bytes":BUDGETS,"value_slots":VALUES,"ways":WAYS,
        "source":"original publisher CPython3.13.0 14 FIT owner documents; previously exposed",
        "memory_math":{"record2_bytes":16,"record3_bytes":20,"source_context_bytes":3,
            "candidate_value_bytes":1,"candidate_counter_bits":16,
            "max_source_clock":8192,
            "algorithm":"exact key + K Misra-Gries candidates per surviving key; decrement all K on previously unseen continuation",
            "guarantee_scope":"ideal MG frequency bound requires no key eviction or counter saturation; neither global guarantee nor exact majority is claimed",
            "budget_includes":"64B real recent ring and one fixed index bytearray; wrappers/clock excluded"},
        "limitations":["No model training, original checkpoint use, protected DEV/TEST, GPU, or independent source confirmation",
            "A real bytearray budget is not whole-process RSS; original documents and diagnostic exact index not accessible to bounded predictions",
            "Hash eviction and candidate decrements mean these counts are NOT exact frequencies",
            "Most-frequent candidate by MG count need not be most-frequent true continuation",
            "Equal stored bytes does not control actual read/write work, CPU hardware, source length or predictor semantics",
            "Horizon8192 is on the same previously exposed publisher FIT owners",
            "No neural TAC, SSM, BDH, Transformer language NLL or model ability has been measured here"],
        "training_updates":0,"gpu_minutes":0,"protected_TEST":"NOT_ACCESSED",
        "checkpoint_promotion":False,"independently_qualified":False}
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    a=parser.parse_args()
    if a.output.exists():raise FileExistsError("original run evidence cannot be overwritten")
    r=run();a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(r,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"study":r["study"],"status":r["status"],
        "summaries":{k:v["summary"] for k,v in r["cohorts"].items()}},sort_keys=True),flush=True)

if __name__=="__main__":main()
