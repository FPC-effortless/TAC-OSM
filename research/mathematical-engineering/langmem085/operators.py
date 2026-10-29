"""LANGMEM085: candidate 8x8 associative source/delta/selective write operators.

All arms fixed normalized SHA(context4) key and 8-bit injective signed unit
byte code. Only update differs, no learned parameters/labels/next-byte. A
bank stores exactly 64 FP32 coordinates, i.e.256B, excluding off-model source
key/clock index used by the evaluator and original 4160B frozen producer.
"""
from __future__ import annotations
import hashlib
import math
import torch
import struct
from torch.nn import functional as F

def context_key(context):
    if type(context) is not bytes or len(context)!=4:raise ValueError("4 observed previous bytes")
    digest=hashlib.sha256(b'LANGMEM084-key-v1'+bytes([0])+context).digest()
    values=[((x+.5)/2**32)*2-1 for x in struct.unpack('<8I',digest)]
    return F.normalize(torch.tensor(values,dtype=torch.float32),dim=0)

def byte_book():
    bits=((torch.arange(256)[:,None]>>torch.arange(8)[None,:])&1).float()
    return (2*bits-1)/(8**.5)

def decode(matrix,key,book):
    read=matrix@key
    sims=book@F.normalize(read,dim=0,eps=1e-12)
    return int(sims.argmax()),read

ARMS=("source","overwrite","partial025","residual_clip035","selective")
AGES=(8,32,128,256)
WINDOW=16

def write(memory,key,value,arm,*,rho,beta,epsilon):
    if arm not in ARMS or memory.shape!=(8,8) or key.shape!=(8,) or value.shape!=(8,):
        raise ValueError("fixed source 8x8 arm geometry")
    if not torch.isfinite(memory).all() or not torch.isfinite(key).all() or not torch.isfinite(value).all():
        raise ValueError("nonfinite original observed source value")
    if not .0<rho<1 or not 0<=beta<=.05 or not 0<=epsilon<=.01:
        raise ValueError("original frozen trained slow source gate contract")
    if abs(float(key.norm())-1)>1e-5 or abs(float(value.norm())-1)>1e-5:
        raise ValueError("all hash keys and explicit byte code values unit norm")
    rhoM=rho*memory
    residual=value-rhoM@key
    norm=float(torch.linalg.vector_norm(residual).item())
    if arm=="source":
        new=rhoM-rho*beta*(memory@key)[:,None]*key[None,:]+epsilon*value[:,None]*key[None,:]
        eta=epsilon
    else:
        eta={"overwrite":1.,"partial025":.25,
             "residual_clip035":min(1.,.35/(norm+1e-9)),
             "selective":1. if norm<=.75 else .25}[arm]
        new=rhoM+eta*residual[:,None]*key[None,:]
    if new.shape!=(8,8) or not torch.isfinite(new).all():
        raise ValueError("bounded real source candidate produced nonfinite matrix")
    return new,float(eta),norm

def prospective_wrong_keys(prefix,keys,a,t,owner):
    """No answer-bearing wrong context, prospectively same-age local clocks.

    Input is only original observed source prefix0..t, and already computed
    hash keys of events <t. Historical target byte raw[a] does NOT enter
    selector or keys; index exclusion prevents its occurrence in b-4:b.
    """
    if not isinstance(prefix,bytes) or len(prefix)!=t+1 or not 4<=a<t:
        raise ValueError("original observation source prefix/clock mismatch")
    original=prefix[a-4:a]
    pool=[b for b in range(max(4,a-WINDOW),min(t-1,a+WINDOW)+1)
          if b!=a and abs(b-a)>=5 and prefix[b-4:b]!=original]
    if not pool:raise ValueError("no clean same-source age-local wrong history")
    if any(not (b<t and abs(b-a)>=5 and abs(b-a)<=WINDOW and
                prefix[b-4:b]!=original and not(a<b<=a+4)) for b in pool):
        raise ValueError("historical answer position leaks into negative context")
    if any(b not in keys for b in pool) or a not in keys:
        raise ValueError("old original source keys missing")
    cos={b:float(torch.dot(keys[b],keys[a])) for b in pool}
    hard=max(pool,key=lambda b:(cos[b],b))
    seed=hashlib.sha256(f"LANGMEM085|{owner}|{t}|{a}".encode()).digest()
    uniform=pool[int.from_bytes(seed[:8],"big")%len(pool)]
    return hard,uniform,len(pool)

def fixed_fit_prior(fit):
    """Frequency-only original CPython FIT control; no Rust data is fitted."""
    if len(fit)!=14:raise ValueError("original original FIT owners not 14")
    counts=torch.zeros((4,256),dtype=torch.int64)
    for _,raw in sorted(fit.items()):
        if not isinstance(raw,bytes) or len(raw)<128:raise ValueError("original FIT source not genuine bytes")
        for t in range(128,min(640,len(raw)),4):
            for i,age in enumerate(AGES):
                a=t-age
                if a>=4:counts[i,raw[a]]+=1
    return counts

def load_frozen_new_rust(folder,source_receipt):
    import json
    from pathlib import Path
    reg=json.loads(Path(source_receipt).read_text())
    if (reg.get("repository")!="rust-lang/book" or reg.get("commit")!="2771df165f092a4fbb9f46b5b2e7450f19010adb"
        or len(reg.get("documents",[]))!=2):
        raise ValueError("registered rust source/commit different")
    docs={}
    for x in reg["documents"]:
        path=Path(folder)/x["raw_filename"]
        if not path.is_file():raise ValueError("frozen Rust publisher source missing")
        raw=path.read_bytes()
        sha=hashlib.sha256(raw).hexdigest()
        blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+bytes([0])+raw).hexdigest()
        if len(raw)!=x["bytes"] or sha!=x["sha256"] or blob!=x["git_blob"]:
            raise ValueError("original frozen MIT publisher source bytes/blob mismatch")
        docs[x["path"]]=raw
    if len(docs)!=2 or len(list(Path(folder).iterdir()))!=2:
        raise ValueError("unregistered source documents present")
    return docs
