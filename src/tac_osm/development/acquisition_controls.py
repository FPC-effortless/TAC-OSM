"""DEV-057 hand-coded acquisition references, not a learned C5 challenger.

Counters measure declared API/record operations, not CPU instructions. Python
dict internals are deliberately not called measured key comparisons. Fixed
stores retain ONLY their 64x3 int32 array; counters are evaluator instruments.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
import sys
import numpy as np

ARMS=('full_scan','indexed_acquisition','equal_bits_scalar','hash_direct','hash_two_way')


@dataclass
class Work:
    build_input_reads:int=0
    build_state_probes:int=0
    record_writes:int=0
    build_address_operations:int=0
    build_key_comparisons:int=0
    record_probes:int=0
    address_operations:int=0
    key_comparisons:int=0
    structural_operations:int=0
    verification_reads:int=0
    verification_operations:int=0


def resident_size(obj):
    seen=set()
    def size(value):
        if id(value) in seen:return 0
        seen.add(id(value));total=sys.getsizeof(value)
        if isinstance(value,dict):total+=sum(size(k)+size(v) for k,v in value.items())
        elif isinstance(value,(tuple,list)):total+=sum(map(size,value))
        return total
    return size(obj)


def hashed(key,slots):return ((int(key)*2654435761)&0xffffffff)%slots


class AcquisitionStore:
    def __init__(self,arm,history_size):
        if arm not in ARMS:raise ValueError('unregistered arm')
        self.arm=arm;self.work=Work()
        if arm=='indexed_acquisition':self.records={}
        else:
            self.records=np.zeros((history_size if arm=='full_scan' else 64,3),dtype=np.int32)
            self.records[:,0]=-1

    def write(self,key,tick,bit):
        if not 0<=key<2**31 or not 1<=tick<2**31 or bit not in (0,1):raise ValueError('invalid public write')
        w=self.work;w.build_input_reads+=1
        if self.arm=='indexed_acquisition':
            w.build_address_operations+=1
            previous=self.records.get(key)
            if previous is not None:
                w.build_state_probes+=1
                if previous[0]>tick:return
            self.records[key]=(tick,bit);w.record_writes+=1
            return
        if self.arm=='full_scan':position=tick-1
        elif self.arm=='equal_bits_scalar':position=(tick-1)%64
        elif self.arm=='hash_direct':
            w.build_address_operations+=1
            position=hashed(key,64);w.build_state_probes+=1;w.build_key_comparisons+=1
            if self.records[position,0]==key and self.records[position,1]>tick:return
        else:
            w.build_address_operations+=1
            start=hashed(key,32)*2
            slots=self.records[start:start+2]
            w.build_state_probes+=2;w.build_key_comparisons+=2
            matches=np.flatnonzero(slots[:,0]==key)
            if len(matches):
                position=start+int(matches[0])
                if self.records[position,1]>tick:return
            else:
                empty=np.flatnonzero(slots[:,0]<0)
                position=start+(int(empty[0]) if len(empty) else int(slots[:,1].argmin()))
        if position>=len(self.records):raise ValueError('public history capacity exceeded')
        self.records[position]=[key,tick,bit];w.record_writes+=1

    def acquire(self,keys):
        if len(keys)!=8 or len(set(keys))!=8:raise ValueError('eight distinct public keys required')
        result={};w=self.work
        if self.arm=='full_scan':
            wanted=set(keys);w.address_operations+=8
            for key,tick,bit in self.records:
                w.record_probes+=1;w.address_operations+=1
                key,tick,bit=int(key),int(tick),int(bit)
                if key in wanted and (key not in result or tick>result[key][0]):result[key]=(tick,bit)
        elif self.arm=='indexed_acquisition':
            for key in keys:
                w.address_operations+=1
                value=self.records.get(key)
                if value is not None:w.record_probes+=1;result[key]=value
        else:
            for key in keys:
                if self.arm=='equal_bits_scalar':positions=range(64)
                else:
                    w.address_operations+=1
                    start=hashed(key,64) if self.arm=='hash_direct' else 2*hashed(key,32)
                    positions=range(start,start+(1 if self.arm=='hash_direct' else 2))
                for position in positions:
                    k,tick,bit=map(int,self.records[position]);w.record_probes+=1;w.key_comparisons+=1
                    if k==key and (key not in result or tick>result[key][0]):result[key]=(tick,bit)
        return [result.get(key) for key in keys]

    @property
    def payload_bytes(self):return len(self.records)*12 if isinstance(self.records,dict) else self.records.nbytes

    @property
    def resident_container_bytes(self):return resident_size(self.records)


def execute_xor(facts,work):
    if len(facts)!=8:raise ValueError('eight circuit inputs required')
    if any(f is None for f in facts):return None
    values=[int(f[1]) for f in facts]
    # Seven actual operations, with a public fixed balanced circuit.
    while len(values)>1:
        values=[values[i]^values[i+1] for i in range(0,len(values),2)]
        work.structural_operations+=len(values)
    return values[0]


def benchmark(seed,history_size,queries=64):
    if history_size%4 or history_size//4<8:raise ValueError('four epochs and >=8 keys required')
    rng=np.random.default_rng(np.random.SeedSequence([seed,history_size,571]))
    keys=rng.choice(2**30,size=history_size//4,replace=False).tolist()
    events=[];latest={}
    for _ in range(4):
        for index in rng.permutation(len(keys)):
            key=int(keys[index]);tick=len(events)+1;bit=int(rng.integers(2))
            events.append((key,tick,bit));latest[key]=(tick,bit)
    requested=[list(map(int,rng.choice(keys,size=8,replace=False))) for _ in range(queries)]
    return events,requested,latest


def measure(seed,history_size,queries,arm):
    import time
    events,requested,latest=benchmark(seed,history_size,queries)
    started=time.perf_counter()
    store=AcquisitionStore(arm,history_size)
    for event in events:store.write(*event)
    build_seconds=time.perf_counter()-started
    build_work=asdict(store.work)
    rows=[];started=time.perf_counter()
    for keys in requested:
        facts=store.acquire(keys)
        answer=execute_xor(facts,store.work)
        gold=[latest[k] for k in keys]
        store.work.verification_reads+=8
        gold_work=Work();expected=execute_xor(gold,gold_work)
        store.work.verification_operations+=gold_work.structural_operations+1
        missing=sum(f is None for f in facts)
        stale=sum(f is not None and f!=g for f,g in zip(facts,gold))
        rows.append({'keys':keys,'facts':facts,'answer':answer,'missing':missing,'stale':stale,
                     'covered':missing==0,'exact':missing==0 and stale==0 and answer==expected})
    elapsed=time.perf_counter()-started
    work=asdict(store.work)
    return {'seed':seed,'history':history_size,'queries':queries,'arm':arm,'rows':rows,
            'success':sum(r['exact'] for r in rows)/queries,'coverage':sum(r['covered'] for r in rows)/queries,
            'missing_facts':sum(r['missing'] for r in rows),'stale_facts':sum(r['stale'] for r in rows),
            'payload_bytes':store.payload_bytes,'resident_container_bytes':store.resident_container_bytes,
            'build_work':build_work,'work':work,'amortized_declared_operations':sum(work.values())/queries,
            'dict_internals_unmeasured':arm=='indexed_acquisition','build_seconds':build_seconds,'query_seconds':elapsed}
