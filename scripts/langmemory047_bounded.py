#!/usr/bin/env python3
"""LANGMEMORY047: byte-budgeted original-source retrieval, not neural TAC.

Compare actual fixed-bytearray memories against an equally sized exact raw
byte ring, legacy A/B sketch, and full growing original-source exact index.
No training, protected TEST, neural NLL, or model checkpoint use.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from scripts.langlayer035_sampling import fit_sources
from scripts.langmemory045_source_probe import BoundedSketch,most_frequent,recent,digest
from scripts.langmemory046_verification_controls import ExactContextIndex

KEY_BYTES=3
RECENT_BYTES=64
SLOT_BYTES=8
TOTAL_BYTES=(1344,4096)
MAX_PRIMARY=1536
MAX_EXTENDED=4096
START=128
STRIDE=4

def fnv(key):
    h=2166136261
    for value in key:
        h=((h^value)*16777619)&0xffffffff
    return h

class RawRing:
    """Real fixed bytearray: contains only the most recently observed B bytes."""
    def __init__(self,capacity):
        if capacity<4:raise ValueError("ring capacity too small")
        self.buffer=bytearray(capacity)
        self.capacity=capacity
        self.clock=0
    def observe(self,value):
        if not (0<=value<=255):raise ValueError("nonbyte source")
        self.buffer[self.clock%self.capacity]=value
        self.clock+=1
    def at(self,position):
        if not (max(0,self.clock-self.capacity)<=position<self.clock):
            raise IndexError("outside retained source ring")
        return self.buffer[position%self.capacity]
    def lookup(self,key):
        if len(key)!=KEY_BYTES:raise ValueError("context size mismatch")
        # j is a previous observed continuation; j-3..j-1 also in ring.
        earliest=max(KEY_BYTES,self.clock-self.capacity+KEY_BYTES)
        counts=Counter()
        compared=0
        for j in range(earliest,self.clock):
            compared+=1
            if all(self.at(j-KEY_BYTES+k)==key[k] for k in range(KEY_BYTES)):
                counts[self.at(j)]+=1
        if not counts:return None,compared
        return min(counts,key=lambda v:(-counts[v],v)),compared

class FixedKeyTable:
    """Exact three-byte keys, approximate majority byte, bounded fixed allocation.

    Layout per record: [3 key bytes, 1 candidate, 1 vote, 2 last_seen clock,
    1 occupied]. Source clock is below 4096, fits uint16. No Python
    per-key dictionary, diagnostic key sets, unbounded linked lists, or
    unmetered stored source. 1-way and 2-way set associative variants.
    """
    def __init__(self,total_bytes,ways):
        if total_bytes not in TOTAL_BYTES or ways not in (1,2):
            raise ValueError("unregistered budget or associativity")
        table_bytes=total_bytes-RECENT_BYTES
        if table_bytes%SLOT_BYTES or (table_bytes//SLOT_BYTES)%ways:
            raise ValueError("not exactly representable in fixed budget")
        self.buffer=bytearray(table_bytes)
        self.ways=ways
        self.sets=table_bytes//SLOT_BYTES//ways
        self.total_bytes=total_bytes
        self.evictions=0
        self.exact_key_hits=0
        self.writes=0
    def offsets(self,key):
        base=(fnv(key)%self.sets)*self.ways*SLOT_BYTES
        return tuple(base+i*SLOT_BYTES for i in range(self.ways))
    def find(self,key):
        for offset in self.offsets(key):
            if self.buffer[offset+7] and self.buffer[offset:offset+3]==key:
                return offset
        return None
    def read(self,key):
        if not isinstance(key,bytes) or len(key)!=KEY_BYTES:
            raise ValueError("only full observed 3-byte keys")
        offset=self.find(key)
        if offset is None or self.buffer[offset+4]==0:return None
        return self.buffer[offset+3]
    def update(self,key,value,clock):
        if not isinstance(key,bytes) or len(key)!=KEY_BYTES or not (0<=value<=255) or not (0<=clock<65536):
            raise ValueError("invalid causal update")
        offsets=self.offsets(key)
        present=self.find(key)
        if present is None:
            empty=next((o for o in offsets if not self.buffer[o+7]),None)
            if empty is not None:offset=empty
            else:
                offset=min(offsets,key=lambda o:(int.from_bytes(self.buffer[o+5:o+7],"little"),o))
                self.evictions+=1
            self.buffer[offset:offset+3]=key
            self.buffer[offset+3]=value
            self.buffer[offset+4]=1
            self.buffer[offset+7]=1
        else:
            offset=present
            self.exact_key_hits+=1
            vote=self.buffer[offset+4]
            if vote==0:
                self.buffer[offset+3]=value;self.buffer[offset+4]=1
            elif self.buffer[offset+3]==value:
                self.buffer[offset+4]=min(vote+1,255)
            else:self.buffer[offset+4]=vote-1
        self.buffer[offset+5:offset+7]=clock.to_bytes(2,"little")
        self.writes+=1
    def occupied(self):
        return sum(self.buffer[i+7]!=0 for i in range(0,len(self.buffer),SLOT_BYTES))
    def actual_payload_bytes(self):return len(self.buffer)+RECENT_BYTES

def empty_metric():
    return {"covered":0,"correct":0,"long_covered":0,"long_correct":0,
            "query_work":0,"lookup_ns":0,"update_ns":0,"calls":0}

def _record(stats,answer,label,long_case,work=0,elapsed_ns=0):
    stats["calls"]+=1
    stats["query_work"]+=work
    stats["lookup_ns"]+=elapsed_ns
    if answer is not None:
        stats["covered"]+=1
        stats["correct"]+=int(answer==label)
        if long_case:
            stats["long_covered"]+=1
            stats["long_correct"]+=int(answer==label)

def study_owner(raw,owner,max_clock=MAX_PRIMARY,return_predictions=False):
    """The next target is never passed to lookup/update before scoring."""
    if not isinstance(raw,bytes) or not raw:raise TypeError("genuine raw bytes expected")
    limit=min(len(raw),max_clock)
    names=("legacy_a_plus_b","recent64","full_exact",
           "table1_1344","table2_1344","table1_4096","table2_4096",
           "ring_1344","ring_4096")
    stats={name:empty_metric() for name in names}
    recent_ring=RawRing(RECENT_BYTES)
    raw_rings={budget:RawRing(budget) for budget in TOTAL_BYTES}
    tables={(budget,ways):FixedKeyTable(budget,ways) for budget in TOTAL_BYTES for ways in (1,2)}
    original_sketch=BoundedSketch()
    exact_index=ExactContextIndex()
    earliest_key={}
    evaluated=0
    long_cases=0
    predictions=[]
    casehash=hashlib.sha256()
    for t in range(limit):
        if t>=KEY_BYTES:
            key=raw[t-KEY_BYTES:t]
            if t>=START and t%STRIDE==0:
                label=raw[t] # EVALUATION ONLY; not accessed by policy.
                long_case=key in earliest_key and earliest_key[key]<t-RECENT_BYTES
                long_cases+=int(long_case)
                evaluated+=1
                casehash.update(owner.encode()+b":"+str(t).encode()+b":"+key)
                legacy_a,_=recent(raw,t,key)
                legacy_b=original_sketch.read(key)
                legacy=legacy_a if legacy_a is not None else legacy_b
                _record(stats["legacy_a_plus_b"],legacy,label,long_case)
                now=time.perf_counter_ns()
                a,a_work=recent_ring.lookup(key)
                a_ns=time.perf_counter_ns()-now
                _record(stats["recent64"],a,label,long_case,a_work,a_ns)
                now=time.perf_counter_ns()
                truth,_=exact_index.read(key)
                exact_ns=time.perf_counter_ns()-now
                _record(stats["full_exact"],truth,label,long_case,0,exact_ns)
                # Two memory components at a fixed byte budget: recent64 and table.
                for (budget,ways),table in tables.items():
                    name=f"table{ways}_{budget}"
                    now=time.perf_counter_ns()
                    b=table.read(key)
                    answer=a if a is not None else b
                    elapsed=time.perf_counter_ns()-now
                    _record(stats[name],answer,label,long_case,ways,elapsed+a_ns)
                for budget,ring in raw_rings.items():
                    name=f"ring_{budget}"
                    now=time.perf_counter_ns()
                    answer,work=ring.lookup(key)
                    elapsed=time.perf_counter_ns()-now
                    _record(stats[name],answer,label,long_case,work,elapsed)
                # Full index is the source-faithful non-neural reference, not
                # a plausible constant-state mobile TAC memory.
                if return_predictions:
                    predictions.append((t,legacy,a,truth,
                        *(a if a is not None else tables[(budget,ways)].read(key)
                          for budget in TOTAL_BYTES for ways in (1,2)),
                        *(raw_rings[budget].lookup(key)[0] for budget in TOTAL_BYTES)))
            if key not in earliest_key:earliest_key[key]=t
            tick=time.perf_counter_ns()
            exact_index.observe(key,raw[t])
            stats["full_exact"]["update_ns"]+=time.perf_counter_ns()-tick
            tick=time.perf_counter_ns()
            original_sketch.update(key,raw[t])
            stats["legacy_a_plus_b"]["update_ns"]+=time.perf_counter_ns()-tick
            for (budget,ways),table in tables.items():
                tick=time.perf_counter_ns()
                table.update(key,raw[t],t)
                stats[f"table{ways}_{budget}"]["update_ns"]+=time.perf_counter_ns()-tick
        # Observe the original byte after all predictions at t.
        tick=time.perf_counter_ns()
        recent_ring.observe(raw[t])
        recent_ns=time.perf_counter_ns()-tick
        stats["recent64"]["update_ns"]+=recent_ns
        for budget,ring in raw_rings.items():
            tick=time.perf_counter_ns()
            ring.observe(raw[t])
            stats[f"ring_{budget}"]["update_ns"]+=time.perf_counter_ns()-tick
    if evaluated==0:raise ValueError("original source owner yielded no evaluable positions")
    for budget in TOTAL_BYTES:
        for ways in (1,2):
            key=(budget,ways);name=f"table{ways}_{budget}"
            # The recent64 observation overhead is part of two-memory cost.
            stats[name]["update_ns"]+=stats["recent64"]["update_ns"]
            if tables[key].actual_payload_bytes()!=budget:raise ValueError("memory budget not enforced")
    if max_clock==MAX_PRIMARY and limit==1536:
        # historical 045A comparability; no neural score.
        pass
    result={"owner":owner,"source_sha256":digest(raw),
            "limit_bytes":limit,"cases":evaluated,"long_history_cases":long_cases,
            "case_identity_sha256":casehash.hexdigest(),"arms":stats,
            "allocated_state_bytes":{"legacy_a_plus_b":"NOMINAL_1344_NOT_ACTUALLY_FIXED",
                "recent64":64,"full_exact":"GROWING_UNBOUNDED",
                **{f"table{ways}_{budget}":budget for budget in TOTAL_BYTES for ways in (1,2)},
                **{f"ring_{budget}":budget for budget in TOTAL_BYTES}},
            "table_diagnostics":{f"table{ways}_{budget}":{
                "slots":len(table.buffer)//SLOT_BYTES,"sets":table.sets,
                "occupied":table.occupied(),"evictions":table.evictions,
                "exact_key_hits":table.exact_key_hits,"writes":table.writes,
                "bytearray_actual_bytes":len(table.buffer),
                "bytearray_heap_bytes":sys.getsizeof(table.buffer)}
                for (budget,ways),table in tables.items()},
            "unbounded_exact_index":{"compact_lower_bound_bytes":exact_index.packed_lower_bound(),
                "python_heap_bytes":exact_index.python_heap_bytes(),
                "distinct_context_keys":exact_index.distinct_contexts()},
            "ring_bytearray_heap_bytes":{str(b):sys.getsizeof(r.buffer) for b,r in raw_rings.items()}}
    return (result,predictions) if return_predictions else result

def aggregate(rows):
    names=list(rows[0]["arms"])
    total=sum(r["cases"] for r in rows)
    long_total=sum(r["long_history_cases"] for r in rows)
    summary={}
    for name in names:
        d={key:sum(r["arms"][name][key] for r in rows) for key in
           ("covered","correct","long_covered","long_correct","query_work",
            "lookup_ns","update_ns","calls")}
        d["coverage"]=d["covered"]/total
        d["accuracy_all_queries"]=d["correct"]/total
        d["accuracy_when_answered"]=d["correct"]/max(1,d["covered"])
        d["long_coverage"]=d["long_covered"]/max(1,long_total)
        d["total_lookup_ms"]=d.pop("lookup_ns")/1e6
        d["total_update_ms"]=d.pop("update_ns")/1e6
        summary[name]=d
    return {"cases":total,"long_history_cases":long_total,"arms":summary,
        "total_exact_index_lower_bound_bytes":sum(r["unbounded_exact_index"]["compact_lower_bound_bytes"] for r in rows),
        "total_exact_index_python_heap_bytes":sum(r["unbounded_exact_index"]["python_heap_bytes"] for r in rows)}

def run():
    fit=fit_sources()
    if len(fit)!=14:raise ValueError("original FIT 14-source owners changed")
    cohorts={}
    for label,cap in (("anchor_1536",MAX_PRIMARY),("extension_up_to_4096",MAX_EXTENDED)):
        rows=[study_owner(raw,owner,cap) for owner,raw in sorted(fit.items())]
        result=aggregate(rows)
        if label=="anchor_1536":
            if result["cases"]!=4928:
                raise ValueError("prior authentic 045/046 anchor geometry changed")
            if result["arms"]["legacy_a_plus_b"]["correct"]!=933:
                raise ValueError("unmodified original cheap A/B baseline changed")
        if any(a["calls"]!=result["cases"] for a in result["arms"].values()):
            raise ValueError("not all memory arms evaluated at identical source queries")
        cohorts[label]={"summary":result,"owners":rows}
    return {"study":"LANGMEMORY047","status":"ACTUAL_AUTHENTIC_FIT_BOUNDED_MEMORY_RETRIEVAL_DIAGNOSTIC",
        "sources":"original publisher CPython3.13.0 14 FIT document owners",
        "cohorts":cohorts,
        "fixed_state_budgets_bytes":TOTAL_BYTES,
        "budget_meaning":"exact allocated bytearray payload: recent64+8-byte record table, or raw ring of exactly same capacity; Python wrapper/runtime excluded",
        "baseline_warning":"legacy A+B old sketch is a Python-object diagnostic with only nominal packed 1344B; full exact index unbounded",
        "scope_warnings":["Original FIT owners previously explored; no unseen source qualification",
            "No held-out DEV or protected TEST; source-only 3-byte exact-key continuation lookup, not TAC neural language NLL",
            "Exact reference full index and diagnostic earliest-key/legacy provenance structures may grow; not part of bounded arms",
            "Python microtiming noisy, object/wrapper overhead excluded; raw ring scan work counted",
            "Read/write work and memory are different resource budgets; do not infer phone/mobile energy",
            "Artificial future-poison constructed tests only; not real-world observed outcomes",
            "No training, original model checkpoint use, GPU, or model promotion"],
        "training_updates":0,"gpu_minutes":0,"protected_TEST":"NOT_ACCESSED",
        "independently_qualified":False,"checkpoint_promotion":False}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    a=parser.parse_args()
    if a.output.exists():raise FileExistsError("do not overwrite research receipt")
    r=run()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(r,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"study":r["study"],"status":r["status"],
        "cohort_summaries":{k:v["summary"] for k,v in r["cohorts"].items()}},
        sort_keys=True),flush=True)

if __name__=="__main__":main()
