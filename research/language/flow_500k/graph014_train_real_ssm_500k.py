"""GRAPH014 real FIT-only 500K SSM training with original TAC sample schedule.

This is a prospective NEW comparator, not recovered original SSM weights.
Training-FLOP parity with original TAC remains UNVERIFIED; DEV is exposed.
Generated .pt files must be added to Git/LFS for REPO_REPLAY=PASS.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import torch
from torch.nn import functional as F
from graph014_selective_ssm_500k import SelectiveDiagonalSSM500K

ARCHIVE_SHA256="cde9a91be40a4e613975c0bd8caa4c19ac24e3d1d0e993db5cacfbd002fecc32"
SAMPLE_SEEDS=(1701,1702,1703,1704,1705)
ORIGINAL_TAC_FIT_SAMPLE_SHA256={
    1701: "8071294fdc5e89989cca20c868304377b8c1ffeacc3eea4f3f2993de89c316cc",
    1702: "fb6df34f252e7fb7af88dec9dbc6f712d6decf8800d6072fbb2c555912e58ae5",
    1703: "56e7d1836119d4f6faa75083476a8757af46d1a78c03b5c155766e72b9f76ea5",
    1704: "e9599e54c248adfe9e4d2ea416daa95ec3bbe5f5dd200bc55bde3fd5b99ee2d0",
    1705: "b6248a48961243ac4cfa62d0eff0ef25cbcbee774e86909c9a1525f56914b23f",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()

def source_windows(archive,source_root):
    if sha(archive.read_bytes())!=ARCHIVE_SHA256:
        raise ValueError("expected independently pinned original TAC archive")
    with ZipFile(archive) as z:
        protocol=json.loads(z.read("tac_language_500k/SOURCE_PARTITION_MANIFEST.json"))["source"]
    parts=protocol["doc_partitions"]
    if len(parts["fit"])!=30 or len(parts["dev"])!=6:
        raise ValueError("source partition identity changed")
    docs={}
    for name in parts["fit"]+parts["dev"]:
        raw=(source_root/name).read_bytes()
        if sha(raw)!=protocol["byte_sha256"][name]:
            raise ValueError("Vim publisher original hash mismatch: "+name)
        a=np.frombuffer(raw,dtype=np.uint8).astype(np.int64).copy()
        a[a>=128]=63
        docs[name]=a
    windows=[]
    for name in parts["dev"]:
        a=docs[name]
        for off in np.linspace(0,len(a)-129,8,dtype=np.int64):
            windows.append(torch.tensor(a[off:off+129],dtype=torch.long))
    if len(windows)!=48:raise ValueError("DEV windows differ from original")
    return parts,docs,windows,sha(json.dumps(protocol,sort_keys=True).encode())

def evaluate(model,windows):
    model.eval()
    total=0.;correct=0;n=0
    with torch.no_grad():
        for i in range(0,len(windows),8):
            batch=torch.stack(windows[i:i+8])
            targets=batch[:,1:]
            logits=model(batch[:,:-1])
            total+=float(F.cross_entropy(logits.reshape(-1,128),
                                         targets.reshape(-1),reduction="sum"))
            correct+=int((logits.argmax(-1)==targets).sum())
            n+=targets.numel()
    if n!=6144:raise ValueError("unexpected DEV label count")
    return {"nll":total/n,"ppl":float(np.exp(total/n)),
            "accuracy":correct/n,"targets":n}

def run(archive,source_root,output_dir,seeds,updates=120):
    if updates!=120:raise ValueError("do not alter frozen 120-update protocol")
    if not seeds or any(s not in SAMPLE_SEEDS for s in seeds) or len(set(seeds))!=len(seeds):
        raise ValueError("registered seed IDs must be unique and known")
    torch.set_num_threads(2)
    parts,docs,dev,source_sha=source_windows(archive,source_root)
    output_dir.mkdir(parents=True,exist_ok=True)
    results=[]
    for seed in seeds:
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        rng=np.random.default_rng(seed)
        sample_hash=hashlib.sha256()
        model=SelectiveDiagonalSSM500K()
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        last=0.;started=time.perf_counter()
        for step in range(updates):
            source_batch=[]
            for _ in range(8):
                name=parts["fit"][int(rng.integers(30))]
                a=docs[name]
                offset=int(rng.integers(len(a)-129))
                sample_hash.update(f"{name}:{offset};".encode())
                source_batch.append(torch.tensor(a[offset:offset+129],dtype=torch.long))
            batch=torch.stack(source_batch)
            logits=model(batch[:,:-1])
            loss=F.cross_entropy(logits.reshape(-1,128),batch[:,1:].reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if step==0 and sum(p.numel() for p in model.parameters() if p.grad is not None)!=498288:
                raise ValueError("unused model parameters in first backward")
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            last=float(loss.detach())
        if sample_hash.hexdigest()!=ORIGINAL_TAC_FIT_SAMPLE_SHA256[seed]:
            raise ValueError('sample order differs from original WaveCED500K source receipts')
        score=evaluate(model,dev)
        ck={"model":model.state_dict(),"optimizer":optimizer.state_dict(),
            "torch_rng":torch.get_rng_state(),"numpy_rng":np.random.get_state(),
            "python_rng":random.getstate(),"sampler_rng":rng.bit_generator.state,
            "seed":seed,"samples_sha256":sample_hash.hexdigest(),
            "dev":score,"updates":updates,"fit_source_protocol_sha256":source_sha}
        path=output_dir/f"ssm500k_seed_{seed}.pt"
        torch.save(ck,path)
        # Cold local restoration: original bytes unmodified, no extra optimizer steps.
        with torch.serialization.safe_globals([
            np._core.multiarray._reconstruct,np._core.multiarray.scalar,np.ndarray,
            np.dtype,np.dtypes.UInt32DType,np.dtypes.Float64DType]):
            loaded=torch.load(path,map_location="cpu",weights_only=True)
        replica=SelectiveDiagonalSSM500K()
        replica.load_state_dict(loaded["model"],strict=True)
        rescored=evaluate(replica,dev)
        if abs(rescored["nll"]-score["nll"])>1e-6:
            raise ValueError("cold checkpoint score mismatch")
        results.append(dict(seed=seed,train_last_nll=last,dev=score,
                            fit_sample_sha256=sample_hash.hexdigest(),
                            checkpoint=path.name,checkpoint_sha256=sha(path.read_bytes()),
                            elapsed_seconds=time.perf_counter()-started))
        print(f"seed={seed} DEV NLL={score['nll']:.6f} PPL={score['ppl']:.4f} replay=PASS",flush=True)
    receipt={"study":"GRAPH014_500K_SELECTIVE_DIAGONAL_SSM_COMPARATOR",
             "evidence":"authentic Vim9.1 FIT, previously examined DEV; no sealed TEST",
             "version":"NEW prospective comparator, not historical SSM",
             "parameter_count":498288,"target_original_tac_parameters":499290,
             "training_updates_per_seed":120,"batch":8,"context":128,
             "active_parameters_checked":True,"optimizer":"AdamW(lr=0.001, default wd)",
             "objective":"next-character categorical CE only",
             "fit_source_protocol_sha256":source_sha,"source_chapters_verified":36,
             "training_compute_equal_to_tac":"NOT_ESTABLISHED",
             "comparison_classification":"EXPLORATORY_NOT_FAIR_ARCHITECTURE_SUPERIORITY",
             "repo_cold_replay":"BLOCKED until checkpoint originals Git-hosted",
             "independent_test":"NOT_RUN","results":results}
    (output_dir/"graph014_result.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    return receipt

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive",type=Path,required=True)
    p.add_argument("--source-root",type=Path,default=Path("/usr/share/vim/vim91/doc"))
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--seeds",type=int,nargs="+",default=list(SAMPLE_SEEDS))
    args=p.parse_args()
    run(args.archive,args.source_root,args.output_dir,args.seeds)

if __name__=="__main__":main()
