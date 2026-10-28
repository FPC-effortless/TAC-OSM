"""GRAPH015-TEST001: frozen first-open CMake 3.31 SOURCE REPLAY only.

These CMake documents were first opened on 2026-10-28 after the exact source
names, windows and metrics were preregistered in GitHub. They are NOW EXPOSED.
This runner never creates a second independently unseen TEST.
"""
from __future__ import annotations
import argparse,hashlib,io,json,math,sys
from pathlib import Path
from zipfile import ZipFile
import numpy as np,torch
from graph014_selective_ssm_500k import SelectiveDiagonalSSM500K
from graph015_delta_tac_500k import Graph015DeltaTAC500K
from bayesian_cache_scan_v5 import ScannedCountCache
from learned_context_query_v6 import LearnedContextQuery
from tac_wave_ced_500k import WaveCED500K

ARCHIVE_SHA256="cde9a91be40a4e613975c0bd8caa4c19ac24e3d1d0e993db5cacfbd002fecc32"
SEEDS=(1701,1702,1703,1704,1705)
FROZEN_CMAKE_SOURCE_SHA256={
    "cmake-buildsystem.7.rst": "43a59fc95e9b85327dcc32732664ce3d899ea0f7393b7ab6a7dbad63d0917982",
    "cmake-compile-features.7.rst": "961b4fc5c202d8c1230a75b1ff07fa724693d9381dd59244daa722ca0adbbd65",
    "cmake-configure-log.7.rst": "3eda1853943ca5096ab7b8bd867d46287091940803496e1eb8a93e1045b7459e",
    "cmake-developer.7.rst": "057e73143bf5b746fc240e75c94bf7c7db934fa00d58c5fb16a4db3c3e8be363",
    "cmake-file-api.7.rst": "d2dbd5907e294169231336e07439b15ae68b0473fe776281dc7b53cac1d0111b",
    "cmake-generator-expressions.7.rst": "2c1cc52ab08b5d1b8939d461b592643449c7507f9a7c819a8a62cf50165502c9",
    "cmake-language.7.rst": "6151c106f1a9f90562674c085f1795d8eaaadb365b86427498808614f33eb1df",
    "cmake-packages.7.rst": "3522aad1233e5263aa1d59b6ef5925939d3f3714f5550f66404f58ceca7ed502c9"
}

def _sha(raw):
    return hashlib.sha256(raw).hexdigest()

def _safe(raw):
    with torch.serialization.safe_globals([
        np._core.multiarray._reconstruct,np._core.multiarray.scalar,
        np.ndarray,np.dtype,np.dtypes.UInt32DType,np.dtypes.Float64DType]):
        return torch.load(io.BytesIO(raw),weights_only=True,map_location="cpu")

def score(tac_archive,query_archive,ssm_checkpoints,delta_checkpoints,source_root,output):
    if _sha(tac_archive.read_bytes())!=ARCHIVE_SHA256:
        raise ValueError("original source/checkpoint archive hash mismatch")
    torch.set_num_threads(2)
    with ZipFile(tac_archive) as old,ZipFile(query_archive) as qzip:
        qmanifest=json.loads(qzip.read("graph011/GRAPH011_ARCHIVE_MANIFEST.json"))
        source=json.loads(old.read("tac_language_500k/SOURCE_PARTITION_MANIFEST.json"))["source"]
        windows=[]
        for name, frozen_sha in FROZEN_CMAKE_SOURCE_SHA256.items():
            raw=(source_root/name).read_bytes()
            if _sha(raw)!=frozen_sha or len(raw)<129:
                raise ValueError("frozen first-open CMake publisher source SHA mismatch")
            a=np.frombuffer(raw,dtype=np.uint8).astype(np.int64).copy()
            a[a>=128]=63
            cuts=np.linspace(0,len(a)-129,8,dtype=np.int64)
            windows.append((name,torch.stack([
                torch.tensor(a[int(i):int(i)+129],dtype=torch.long) for i in cuts])))
        if len(windows)!=8:raise ValueError("registered CMake source list changed")
        counter=ScannedCountCache(512)
        results=[]
        for seed in SEEDS:
            tac_ck=_safe(old.read(f"tac_language_500k/seed_{seed}.pt"))
            tac=WaveCED500K().eval()
            if tac.parameter_count()!=499290:raise ValueError("original TAC architecture mismatch")
            tac.load_state_dict(tac_ck["model"],strict=True)
            ssm_raw=(ssm_checkpoints/f"ssm500k_seed_{seed}.pt").read_bytes()
            ssm_ck=_safe(ssm_raw)
            ssm=SelectiveDiagonalSSM500K().eval()
            ssm.load_state_dict(ssm_ck["model"],strict=True)
            if ssm_ck["seed"]!=seed or ssm_ck["updates"]!=120:
                raise ValueError("new SSM checkpoint identity invalid")
            delta_raw=(delta_checkpoints/f"delta_tac500k_seed_{seed}.pt").read_bytes()
            delta_ck=_safe(delta_raw)
            delta=Graph015DeltaTAC500K().eval()
            delta.load_state_dict(delta_ck["model"],strict=True)
            if delta_ck["seed"]!=seed or delta_ck["updates"]!=120 or delta_ck["samples_sha256"]!=tac_ck["samples_sha256"]:
                raise ValueError("new delta TAC checkpoint identity or original FIT sample schedule mismatch")
            qname=f"run/graph010_query_{seed}.pt"
            qraw=qzip.read("graph011/"+qname)
            if _sha(qraw)!=qmanifest[qname]["sha256"]:
                raise ValueError("original GRAPH011 query checkpoint SHA mismatch")
            qs=_safe(qraw)["model"]
            prior=qs["fit_prior"]
            query=LearnedContextQuery(prior,capacity=512,topk=8,
                                      temperature=.25,tau=.05).eval()
            if query.learned_parameters()!=2816:
                raise ValueError("frozen GRAPH011 query parameter mismatch")
            query.load_state_dict(qs,strict=True)
            sums={f"{name}_{stage}":0. for name in ("tac","delta","ssm","prior")
                  for stage in ("native","exact","both")}
            hits={"exact":0,"approx":0,"none":0}
            n=0
            documents=[]
            with torch.no_grad():
                for filename,batch in windows:
                    x,y=batch[:,:-1],batch[:,1:]
                    owners=tuple(f"{filename}::window::{i}" for i in range(8))
                    ca,_=counter.scan_counts(x,start=0,source_ids=owners)
                    c=ca["counts"][:,:,1].to(torch.float32)
                    nt=ca["totals"][:,:,1].to(torch.float32)
                    exact_hit=nt>0
                    exact_p=(c+.05*prior[None,None,:])/(nt[...,None]+.05)
                    q=query.predict_window(x)
                    if not torch.equal(q["exact_hit"],exact_hit):
                        raise ValueError("shared source-only exact-hit providers disagree")
                    if bool((exact_hit & q["hit"]).any()):
                        raise ValueError("approximate query read on an exact hit")
                    hits["exact"]+=int(exact_hit.sum())
                    hits["approx"]+=int(q["hit"].sum())
                    hits["none"]+=int((~exact_hit & ~q["hit"]).sum())
                    bases={"tac":torch.softmax(tac(x),-1),
                           "delta":torch.softmax(delta(x),-1),
                           "ssm":torch.softmax(ssm(x),-1),
                           "prior":prior[None,None,:].expand(8,128,128)}
                    doc={"filename":filename,"losses":{},"targets":int(y.numel())}
                    for name,p in bases.items():
                        exact=torch.where(exact_hit[...,None],.4*p+.6*exact_p,p)
                        both=torch.where(q["hit"][...,None],.9*exact+.1*q["posterior"],exact)
                        for stage,law in (("native",p),("exact",exact),("both",both)):
                            if (not bool(torch.isfinite(law).all()) or
                                not torch.allclose(law.sum(-1),torch.ones_like(y,dtype=law.dtype),
                                                   atol=4e-5)):
                                raise ValueError("invalid probability law")
                            value=float((-law.gather(-1,y[...,None]).squeeze(-1).log()).sum())
                            sums[f"{name}_{stage}"]+=value
                            doc["losses"][f"{name}_{stage}"]=value/y.numel()
                    n+=y.numel()
                    documents.append(doc)
            if n!=8192:raise ValueError("unexpected frozen CMake target denominator")
            means={k:v/n for k,v in sums.items()}
            # Original DEV NLL MUST NOT be used as an expected score for new CMake sources.
            results.append(dict(seed=seed,means=means,documents=documents,routing=hits,
                                tac_checkpoint_sha256=_sha(old.read(f"tac_language_500k/seed_{seed}.pt")),
                                ssm_checkpoint_sha256=_sha(ssm_raw),
                                query_checkpoint_sha256=_sha(qraw),
                                delta_checkpoint_sha256=_sha(delta_raw)))
        totals={k:math.fsum(row["means"][k] for row in results)/len(results)
                for k in results[0]["means"]}
        result=dict(study="GRAPH015_TEST001_CMAKE31_FROZEN_FIRST_OPEN_REPLAY",
                    evidence_status="FIRST_OPEN_SOURCE_NOW_EXPOSED_REPLAY_NOT_ARCHITECTURE_QUALIFIED",
                    target_count_per_seed=8192,source_documents=8,source_sha256=FROZEN_CMAKE_SOURCE_SHA256,results=results,means=totals,
                    gates=dict(training_compute_parity="NOT_ESTABLISHED",
                               repository_checkpoint_replay="BLOCKED",
                               independent_test="NOT_RUN",
                               novel_semantic_memory="NOT_TESTED"))
        output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
        return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tac-archive",type=Path,required=True)
    p.add_argument("--query-archive",type=Path,required=True)
    p.add_argument("--ssm-checkpoints",type=Path,required=True)
    p.add_argument("--delta-checkpoints",type=Path,required=True)
    p.add_argument("--source-root",type=Path,default=Path("/usr/share/cmake-3.31/Help/manual"))
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    r=score(a.tac_archive,a.query_archive,a.ssm_checkpoints,a.delta_checkpoints,a.source_root,a.output)
    print("GRAPH015 CMake frozen source replay, NOT new independent TEST:",
          r["means"]["tac_both"],r["means"]["delta_both"],r["means"]["ssm_both"],
          "(not qualified architectural superiority)")

if __name__=="__main__":main()
