#!/usr/bin/env python3
"""LANGMEMORY048: single fixed-bytearray ring + full-key verified cyclic locator.

Original publisher FIT source only. This is a no-fit source-mechanism probe,
not a model, efficient neural attention, calibrated gate, or heldout result.
"""
from __future__ import annotations
import argparse,hashlib,json,math,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from scripts.langlayer035_sampling import fit_sources
from scripts.langmemory047_bounded import RawRing,FixedKeyTable,KEY_BYTES,fnv
from scripts.langmemory045_source_probe import BoundedSketch,recent,digest
from scripts.langmemory046_verification_controls import ExactContextIndex

BUDGETS={1344:1024,4096:3072}  # ring and locator share ONE fixed buffer.
SLOT_BYTES=8
STRIDE={1536:4,4096:4,8192:8}
COHORTS=(("anchor1536",1536),("extension4096",4096),("horizon8192",8192))

class VerifiedRingLocator:
    """One contiguous allocated bytearray incl ring and 8-byte offset records.

    Each record: uint16 FNV fingerprint, uint16 absolute j, valid byte,
    3 reserved zero bytes. Fixed 2-way possible; actual stored key checked
    using original bytes IN this bounded ring. No candidate value or source
    history is stored outside the buffer.
    """
    def __init__(self,budget,ways):
        if budget not in BUDGETS or ways not in (1,2):
            raise ValueError("unregistered byte budget / ways")
        self.payload=bytearray(budget)
        self.capacity=BUDGETS[budget]
        self.ways=ways
        self.slots=(budget-self.capacity)//SLOT_BYTES
        if self.slots%ways:raise ValueError("index slot geometry")
        self.sets=self.slots//ways
        self.clock=0
        self.evictions=0
        self.stale_hits=0
        self.fingerprint_collisions=0
        self.valid_hits=0
        self.hash_slot_reads=0
        self.fallback_scanned=0
        self.updated=0
    def at(self,pos):
        if not (max(0,self.clock-self.capacity)<=pos<self.clock):
            raise IndexError("source byte evicted")
        return self.payload[pos%self.capacity]
    def current_key(self):
        if self.clock<KEY_BYTES:raise ValueError("not enough causal history")
        return bytes(self.at(j) for j in range(self.clock-KEY_BYTES,self.clock))
    def fingerprint(self,key):
        return fnv(key)&0xffff
    def slots_for(self,key):
        group=fnv(key)%self.sets
        return tuple(self.capacity+(group*self.ways+i)*SLOT_BYTES for i in range(self.ways))
    def entry(self,start):
        return (int.from_bytes(self.payload[start:start+2],"little"),
                int.from_bytes(self.payload[start+2:start+4],"little"),
                self.payload[start+4])
    def valid_record(self,start,key):
        fingerprint,pos,valid=self.entry(start)
        if not valid:return False
        # Need the continuation byte AND its 3 immediately prior context bytes.
        if pos-KEY_BYTES<max(0,self.clock-self.capacity) or pos>=self.clock:
            self.stale_hits+=1
            return False
        if fingerprint!=self.fingerprint(key):
            return False
        if bytes(self.at(j) for j in range(pos-KEY_BYTES,pos))!=key:
            self.fingerprint_collisions+=1
            return False
        return True
    def _write_entry(self,off,key,pos):
        self.payload[off:off+2]=self.fingerprint(key).to_bytes(2,"little")
        self.payload[off+2:off+4]=pos.to_bytes(2,"little")
        self.payload[off+4]=1
        self.payload[off+5:off+8]=b"\x00\x00\x00"
    def observe(self,value):
        """Updates source from previously observed byte, never a future answer."""
        if not (0<=value<=255) or self.clock>=65535:raise ValueError("source or clock out of bounds")
        key=self.current_key() if self.clock>=KEY_BYTES else None
        pos=self.clock
        # Source retained before locator points at pos; verify stale afterwards.
        self.payload[pos%self.capacity]=value
        self.clock+=1
        if key is None:return
        starts=self.slots_for(key)
        # Prefer refreshing actual same exact key already present.
        same=None;free=None
        for off in starts:
            self.hash_slot_reads+=1
            fp,j,valid=self.entry(off)
            if not valid or (j-KEY_BYTES<max(0,self.clock-self.capacity)):
                if free is None:free=off
                continue
            if fp==self.fingerprint(key) and self.valid_record(off,key):
                same=off;break
        if same is not None:target=same
        elif free is not None:target=free
        else:
            # Deterministic bounded local oldest-record replacement.
            target=min(starts,key=lambda off:self.entry(off)[1])
            self.evictions+=1
        self._write_entry(target,key,pos)
        self.updated+=1
    def lookup(self,key,scan_on_miss=0):
        """Return latest truly matching source continuation, else optional scan.

        Caller-supplied key must equal current observed last3 bytes. The
        locator finds one latest record, NOT majority continuation. Scan on
        miss explores only bounded raw ring bytes, never an oracle source.
        """
        if key!=self.current_key():
            raise ValueError("query key not equal to actually retained observed prefix")
        candidates=[]
        inspected=0
        for off in self.slots_for(key):
            inspected+=1
            if self.valid_record(off,key):
                candidates.append(self.entry(off)[1])
        self.hash_slot_reads+=inspected
        if candidates:
            self.valid_hits+=1
            pos=max(candidates)
            return self.at(pos),inspected,False
        if scan_on_miss:
            # Full or truncated source prefix, counts exact-key continuations.
            max_scan=min(scan_on_miss,self.capacity-KEY_BYTES,self.clock-KEY_BYTES)
            earliest=max(KEY_BYTES,self.clock-max_scan)
            values={}
            for j in range(earliest,self.clock):
                self.fallback_scanned+=1
                inspected+=1
                if j-KEY_BYTES<max(0,self.clock-self.capacity):continue
                if bytes(self.at(k) for k in range(j-KEY_BYTES,j))==key:
                    value=self.at(j)
                    values[value]=values.get(value,0)+1
            if values:
                return min(values,key=lambda v:(-values[v],v)),inspected,True
        return None,inspected,bool(scan_on_miss)

def record(stats,answer,label,long_case,lookups,lookup_ns):
    stats["queries"]+=1
    stats["work_units"]+=lookups
    stats["lookup_ns"]+=lookup_ns
    if answer is not None:
        stats["covered"]+=1
        stats["correct"]+=int(answer==label)
        if long_case:
            stats["long_covered"]+=1
            stats["long_correct"]+=int(answer==label)

def fresh_stats():
    return {"queries":0,"covered":0,"correct":0,"long_covered":0,
            "long_correct":0,"work_units":0,"lookup_ns":0,"update_ns":0}

def process_owner(raw,owner,cap):
    if not isinstance(raw,bytes):raise TypeError("unmodified authentic original source bytes required")
    stride=STRIDE[cap];limit=min(cap,len(raw))
    loc={(b,w):VerifiedRingLocator(b,w) for b in BUDGETS for w in (1,2)}
    rings={b:RawRing(b) for b in BUDGETS}
    # 1024 and 3072 are *ring-only* controls for what the locator replaces.
    rings.update({b+100000:RawRing(BUDGETS[b]) for b in BUDGETS})
    exact=ExactContextIndex()
    sketch=BoundedSketch()
    legacy_recent=RawRing(64)
    legacy_table={b:FixedKeyTable(b,2) for b in BUDGETS}
    names=["exact_unbounded","legacy_sketch","table2_1344","table2_4096"]
    names += [f"raw_{b}" for b in BUDGETS]
    names += [f"raw_sub_{b}" for b in BUDGETS]
    for b in BUDGETS:
        for w in (1,2):
            names += [f"locator{w}_{b}","locator2_{}_scan64".format(b)] if w==2 else [f"locator{w}_{b}"]
    names=list(dict.fromkeys(names))
    stats={n:fresh_stats() for n in names}
    first_seen={}
    cases=long_total=0
    for t in range(limit):
        if t>=KEY_BYTES:
            key=raw[t-KEY_BYTES:t]
            if t>=128 and t%stride==0:
                label=raw[t] # evaluated only after all policy logits/answers fixed
                long_case=key in first_seen and first_seen[key]<t-64
                long_total+=long_case
                cases+=1
                tick=time.perf_counter_ns()
                e,_=exact.read(key)
                record(stats["exact_unbounded"],e,label,long_case,1,time.perf_counter_ns()-tick)
                tick=time.perf_counter_ns()
                a,_=recent(raw,t,key)
                b=sketch.read(key)
                fallback=a if a is not None else b
                record(stats["legacy_sketch"],fallback,label,long_case,1,time.perf_counter_ns()-tick)
                for budget,table in legacy_table.items():
                    now=time.perf_counter_ns()
                    a2,_=legacy_recent.lookup(key)
                    pred=a2 if a2 is not None else table.read(key)
                    record(stats[f"table2_{budget}"],pred,label,long_case,2,time.perf_counter_ns()-now)
                for code,ring in rings.items():
                    name=f"raw_sub_{code-100000}" if code>=100000 else f"raw_{code}"
                    now=time.perf_counter_ns()
                    p,steps=ring.lookup(key)
                    record(stats[name],p,label,long_case,steps,time.perf_counter_ns()-now)
                for (budget,ways),model in loc.items():
                    name=f"locator{ways}_{budget}"
                    if model.current_key()!=key:raise ValueError("causality mismatch for index query")
                    now=time.perf_counter_ns()
                    p,work,_=model.lookup(key)
                    record(stats[name],p,label,long_case,work,time.perf_counter_ns()-now)
                    if ways==2:
                        now=time.perf_counter_ns()
                        q,work,_=model.lookup(key,scan_on_miss=64)
                        record(stats[f"locator2_{budget}_scan64"],q,label,long_case,work,time.perf_counter_ns()-now)
            if key not in first_seen:first_seen[key]=t
            tick=time.perf_counter_ns()
            exact.observe(key,raw[t])
            stats["exact_unbounded"]["update_ns"]+=time.perf_counter_ns()-tick
            tick=time.perf_counter_ns()
            sketch.update(key,raw[t])
            stats["legacy_sketch"]["update_ns"]+=time.perf_counter_ns()-tick
            for budget,table in legacy_table.items():
                tick=time.perf_counter_ns();table.update(key,raw[t],t)
                stats[f"table2_{budget}"]["update_ns"]+=time.perf_counter_ns()-tick
        # Change mutable state only after scoring the source byte.
        for (budget,ways),model in loc.items():
            tick=time.perf_counter_ns();model.observe(raw[t])
            delta=time.perf_counter_ns()-tick
            stats[f"locator{ways}_{budget}"]["update_ns"]+=delta
            if ways==2:stats[f"locator2_{budget}_scan64"]["update_ns"]+=delta
        for code,ring in rings.items():
            name=f"raw_sub_{code-100000}" if code>=100000 else f"raw_{code}"
            tick=time.perf_counter_ns();ring.observe(raw[t])
            stats[name]["update_ns"]+=time.perf_counter_ns()-tick
        tick=time.perf_counter_ns();legacy_recent.observe(raw[t])
        delta=time.perf_counter_ns()-tick
        for budget in BUDGETS:stats[f"table2_{budget}"]["update_ns"]+=delta
    if not cases:raise ValueError("empty observed original-source cohort")
    for name,item in stats.items():
        if item["queries"]!=cases:raise ValueError(f"not equal evaluation records: {name}")
    return {"owner":owner,"source_sha256":digest(raw),"observed_prefix_bytes":limit,
        "positions":cases,"genuine_older_than64":long_total,
        "stats":stats,"locators":{f"locator{w}_{b}":{
            "bytes":len(m.payload),"ring_bytes":m.capacity,"index_bytes":b-m.capacity,
            "sets":m.sets,"ways":w,"evictions":m.evictions,
            "stale_checked":m.stale_hits,"fingerprint_false_positives":m.fingerprint_collisions,
            "exact_verified_hits":m.valid_hits,"hash_slot_reads":m.hash_slot_reads,
            "scan_fallback_positions":m.fallback_scanned,
            "updates":m.updated,"bytearray_python_heap":sys.getsizeof(m.payload)}
            for (b,w),m in loc.items()},
        "growing_exact_reference":{
            "packed_lower_bound_bytes":exact.packed_lower_bound(),
            "python_heap_bytes":exact.python_heap_bytes(),
            "distinct_keys":exact.distinct_contexts()}}

def aggregate(rows):
    total=sum(r["positions"] for r in rows)
    long_count=sum(r["genuine_older_than64"] for r in rows)
    names=rows[0]["stats"]
    report={}
    for name in names:
        out={k:sum(r["stats"][name][k] for r in rows) for k in
             ("queries","covered","correct","long_covered","long_correct",
              "work_units","lookup_ns","update_ns")}
        out["success_all_positions"]=out["correct"]/total
        out["coverage"]=out["covered"]/total
        out["long_coverage"]=out["long_covered"]/max(1,long_count)
        out["total_lookup_ms"]=out.pop("lookup_ns")/1e6
        out["total_update_ms"]=out.pop("update_ns")/1e6
        report[name]=out
    return {"positions":total,"genuine_older_than64":long_count,
        "arms":report,"original_index_bytes_lower_bound":sum(r["growing_exact_reference"]["packed_lower_bound_bytes"] for r in rows),
        "original_index_python_heap":sum(r["growing_exact_reference"]["python_heap_bytes"] for r in rows),
        "locator_diagnostics":{f"locator{w}_{b}":{
            k:sum(r["locators"][f"locator{w}_{b}"][k] for r in rows) for k in
            ("evictions","stale_checked","fingerprint_false_positives","exact_verified_hits",
             "hash_slot_reads","scan_fallback_positions","updates")}
            for b in BUDGETS for w in (1,2)}}

def run():
    fit=fit_sources()
    if len(fit)!=14:raise ValueError("original authentic publisher cohort mismatch")
    cohorts={}
    for label,limit in COHORTS:
        rows=[process_owner(raw,owner,limit) for owner,raw in sorted(fit.items())]
        summary=aggregate(rows)
        if label=="anchor1536":
            if summary["positions"]!=4928:raise ValueError("original query geometry drift")
            if summary["arms"]["legacy_sketch"]["correct"]!=933:raise ValueError("original A+B reference changed")
            if summary["arms"]["raw_1344"]["correct"]!=1461:raise ValueError("original same-byte ring reference changed")
        if label=="extension4096":
            if summary["positions"]!=13245:raise ValueError("extended source query drift")
            if summary["arms"]["raw_1344"]["correct"]!=4503:raise ValueError("original extended 1344B reference changed")
        cohorts[label]={"summary":summary,"owners":rows}
    return {"study":"LANGMEMORY048","status":"SOURCE_ONLY_FIXED_BUFFER_INDEX_ACTUAL_MEASUREMENT",
        "source":"Original CPython3.13.0 publisher FIT owners, previously exposed in 045-047",
        "cohorts":cohorts,
        "geometry":{"total_budget_bytes":list(BUDGETS),
            "ring_bytes_by_budget":{str(k):v for k,v in BUDGETS.items()},
            "index_entry_bytes":8,"index_offset":"u16 absolute 0..65534; verify full original key and ring range",
            "output":"latest matching verified continuation, NOT exact majority",
            "fallback":"optional scan last64 only if locator misses; explicitly counted",
            "actual_bytearray_payload":"single bytearray per locator, ring+index; Python object metadata not included"},
        "limits":["same genuine FIT owners already used; no new independent source or protected TEST",
            "source-only ngram continuation, not TAC neural next-character NLL or learned retrieval",
            "raw window shortened by index capacity; same-total-byte parity but not same accessible raw history",
            "index lookup O(ways) worst-case, fallback O(64), not O(1) for exact majority",
            "16-bit source offset limits this prototype to <=65535 observed bytes per stream",
            "full index and prefix historical first_seen dictionaries diagnostic only, unbounded and not accessible to locator",
            "CPU microtimings vary across runs; actual reads, writes and workload differ between arms",
            "no trained model, GPU use, energy/mobile measurement or checkpoint promotion"],
        "training_updates":0,"gpu_minutes":0,"protected_TEST":"NOT_ACCESSED",
        "independently_qualified":False,"checkpoint_promotion":False}

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True)
    x=p.parse_args()
    if x.output.exists():raise FileExistsError("original evidence never overwritten")
    result=run()
    x.output.parent.mkdir(parents=True,exist_ok=True)
    x.output.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"study":result["study"],"status":result["status"],
        "cohorts":{k:v["summary"] for k,v in result["cohorts"].items()}},sort_keys=True),flush=True)

if __name__=="__main__":main()
