#!/usr/bin/env python3
"""LANGRECOMP052: original-trained TAC inference state vs full-prefix replay.

NOT the original GRAPH011 implementation; neither train-time activation
checkpointing nor peak CUDA VRAM is measured here. Original weights/source
and per-step logits must replay before any timing evidence is accepted.
"""
from __future__ import annotations
import argparse,hashlib,io,json,math,platform,statistics,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"src"),str(ROOT/"research/mathematical-engineering/langstream043")]
from incremental import IncrementalTAC043
from scripts.langlayer035_sampling import fit_sources,schedule_receipt
from scripts.langlayer041_real_context_probe import model_for

CHECKPOINT_SHA="0cbc1b89f53ea2a0fe1371a179a9ff14cc35ae84d09f76bdab1dffeac4ac86cb"
SOURCE_COMMIT="60403a5409ff2c3f3b07dd2ca91a7a3e096839c7"
SOURCE_ORDER_SHA="45985ef86e7df644a3acfeee73fe781bbd8d2cb290ac6465dfe670579e6f702f"
ATOL=3e-5
RTOL=1e-4

def sha(raw):return hashlib.sha256(raw).hexdigest()

def one(folder,pattern):
    files=list(folder.rglob(pattern))
    if len(files)!=1 or files[0].is_symlink():raise ValueError("missing/duplicate original research file "+pattern)
    return files[0]

def load_original(folder,fit):
    path=one(folder,"*.pt")
    blob=path.read_bytes()
    receipt=json.loads(one(folder,"FIT_AND_EXPOSED_DEV_RECEIPT.json").read_text())
    if sha(blob)!=CHECKPOINT_SHA or receipt["checkpoint_sha256"]!=CHECKPOINT_SHA:
        raise ValueError("original TAC checkpoint hash mismatch")
    x=torch.load(io.BytesIO(blob),map_location="cpu",weights_only=True)
    schedule=schedule_receipt(fit,1701,steps=320)["sample_order_sha256"]
    if schedule!=SOURCE_ORDER_SHA:raise ValueError("source scheduling cohort unexpectedly changed")
    if (x["study"],x["arm"],x["seed"],x["updates"],x["source_commit"],x["sample_order_sha256"])!=(
        "LANGLAYER040","state48_no_graph",1701,320,SOURCE_COMMIT,SOURCE_ORDER_SHA):
        raise ValueError("checkpoint source/training identity mismatch")
    if not x["optimizer"]["state"] or any(k not in x for k in ("rng_torch","rng_numpy","rng_python")):
        raise ValueError("missing original training replay details")
    model=model_for("h48_plain").eval()
    model.load_state_dict(x["model"],strict=True)
    if sum(p.numel() for p in model.parameters())!=499623:
        raise ValueError("unexpected model parameter count")
    return model,{"sha256":CHECKPOINT_SHA,"original_source_commit":SOURCE_COMMIT,
                  "source_sample_order_sha256":SOURCE_ORDER_SHA,"optimizer_and_rng_checked":True}

def assert_equal(a,b):
    if a.shape!=b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("recomputed logits invalid")
    err=(a-b).abs().max().item()
    if not torch.allclose(a,b,atol=ATOL,rtol=RTOL):
        raise ValueError(f"non-equivalent recomputed logits max_abs={err}")
    if not torch.equal(a.argmax(-1),b.argmax(-1)):
        raise ValueError("different generated next-token prediction despite tolerances")
    return err

@torch.no_grad()
def original_incremental(engine,raw,owner,steps=16):
    prefix=torch.tensor([list(raw)],dtype=torch.long)
    y,state=engine.prefill(prefix,owners=(owner,))
    initial=y.clone()
    generated=[]
    logits=[]
    for _ in range(steps):
        tok=y.argmax(-1)
        generated.append(int(tok[0]))
        y,state=engine.step(tok,state,owners=(owner,))
        logits.append(y.clone())
    return {"initial":initial,"generated":bytes(generated),"logits":logits,
            "state_bytes":state.tensor_bytes()}

@torch.no_grad()
def replay_parity(engine,raw,owner,baseline):
    x=bytearray(raw)
    first,st=engine.prefill(torch.tensor([list(x)],dtype=torch.long),owners=(owner,))
    maxerr=assert_equal(first,baseline["initial"])
    assert st.tensor_bytes()==baseline["state_bytes"]
    for tok,reference in zip(baseline["generated"],baseline["logits"]):
        x.append(tok)
        y,state=engine.prefill(torch.tensor([list(x)],dtype=torch.long),owners=(owner,))
        maxerr=max(maxerr,assert_equal(y,reference))
        if state.clock!=len(x):raise ValueError("replay clock changed")
        if state.tensor_bytes()!=baseline["state_bytes"]:raise ValueError("replay has different state bytes")
        del state
    del st
    return {"max_abs_logit":maxerr,
            "source_bytes_after_generation":len(x),
            "incremental_tensor_state_bytes":baseline["state_bytes"],
            "no_saved_state_after_replay":True,
            "source_prefix_sha256":sha(raw),
            "generated_sha256":sha(baseline["generated"])}

def time_trial(engine,raw,owner,generated,arm):
    x=bytearray(raw)
    tok=torch.tensor([list(x)],dtype=torch.long)
    tick=time.perf_counter_ns()
    y,state=engine.prefill(tok,owners=(owner,))
    prefill_ms=(time.perf_counter_ns()-tick)/1e6
    times=[]
    for b in generated:
        tick=time.perf_counter_ns()
        if arm=="incremental":
            y,state=engine.step(torch.tensor([b],dtype=torch.long),state,owners=(owner,))
        elif arm=="recompute_every_step":
            x.append(b)
            # This is a transient full-prefix tensor. Source bytes retained
            # between queries, but large transient model activations exist.
            y,temporary_state=engine.prefill(torch.tensor([list(x)],dtype=torch.long),owners=(owner,))
            del temporary_state
        else:raise ValueError("unknown frozen policy")
        times.append((time.perf_counter_ns()-tick)/1e6)
    state_bytes=state.tensor_bytes() if arm=="incremental" else 0
    if arm=="recompute_every_step" and len(x)!=len(raw)+len(generated):
        raise ValueError("prefix length mismatch")
    return {"prefill_ms":prefill_ms,"decode_step_ms":times,
            "retained_model_state_tensor_bytes":state_bytes,
            "retained_raw_prefix_bytes":len(x) if arm=="recompute_every_step" else 0}

def median(values):
    return statistics.median(values)

def dist(values):
    vals=sorted(values)
    return {"median_ms":median(vals),
            "p95_ms":vals[max(0,math.ceil(.95*len(vals))-1)],
            "raw_ms":values}

def run(folder):
    if torch.cuda.is_initialized():raise ValueError("registered study is CPU only")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    fit=fit_sources()
    model,receipt=load_original(folder,fit)
    owner=next(x for x in sorted(fit) if len(fit[x])>=544)
    engine=IncrementalTAC043(model,"cap_all")
    rows=[]
    with torch.inference_mode():
        for n in (128,512):
            raw=fit[owner][:n]
            reference=original_incremental(engine,raw,owner)
            check=replay_parity(engine,raw,owner,reference)
            timings={k:[] for k in ("incremental","recompute_every_step")}
            for i in range(7):
                order=("incremental","recompute_every_step") if i%2==0 else ("recompute_every_step","incremental")
                for arm in order:
                    result=time_trial(engine,raw,owner,reference["generated"],arm)
                    if i>=2:timings[arm].append(result)
            for arm,t in timings.items():
                prefill=dist([s["prefill_ms"] for s in t])
                steps=dist([v for s in t for v in s["decode_step_ms"]])
                total=dist([s["prefill_ms"]+sum(s["decode_step_ms"]) for s in t])
                rows.append({"arm":arm,"original_prompt_bytes":n,"warmups":2,"repeats":5,
                    "step_count":len(reference["generated"]),
                    "prefill":prefill,"per_generated_byte":steps,"total_prefill_plus_16":total,
                    "source_proof":check,"retained_state_tensor_bytes":t[0]["retained_model_state_tensor_bytes"],
                    "retained_source_prefix_bytes":t[0]["retained_raw_prefix_bytes"]})
    if any(x["source_proof"]["max_abs_logit"]>ATOL for x in rows):
        raise ValueError("original source parity gate failed")
    return {"study":"LANGRECOMP052",
        "status":"ORIGINAL_TRAINED_TAC_CPU_RECOMPUTATION_REPLAY_MEASURED",
        "original_checkpoint_receipt":receipt,
        "trained_parameters":499623,
        "original_trained_seed":1701,"training_updates":0,
        "source_partition":"ORIGINAL_CP3.13_FIT_ONLY","source_owner":owner,
        "platform":platform.platform(),"torch":torch.__version__,
        "settings":"two CPU threads, FP32 eager, batch1, 16 greedy token updates, two warmups five measured",
        "results":rows,"gpu_minutes":0,"protected_TEST":"NOT_ACCESSED",
        "whole_model_peak_vram":"NOT_MEASURED_CPU_ONLY",
        "training_activation_recomputation":"NOT_TESTED",
        "graph011_exact_original_implementation":"UNVERIFIED_NOT_IMPORTED",
        "retained_state_definition":"state tensors outside active replay calls, excludes weights/transient memory; bytearray authentic prefix retained only by recompute policy",
        "limits":["Not GRAPH011 original verified implementation, not activation rematerialization during gradient checkpoint training",
                  "No GPU VRAM measurement: 5.51MB to0.0014MB and 99% claims NOT VERIFIED",
                  "Eager CPU transient tensor memory/whole process RSS/weight tensors excluded from retained-state counts",
                  "Recomputation trades CPU inference speed for fewer persistent state tensors, exact source bytes grow with context",
                  "Same previously explored publisher FIT owners and previously trained original checkpoint; not fresh language NLL comparison",
                  "No training, checkpoints modified, optimizer/reward evaluation, DEV/TEST, or independent scientific promotion"],
        "independently_qualified":False,"checkpoint_promotion":False}
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--tac-dir",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists():raise FileExistsError("source evidence must not overwrite")
    report=run(a.tac_dir)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({"study":report["study"],"status":report["status"],
        "measurements":[{"arm":r["arm"],"prompt":r["original_prompt_bytes"],
                         "median_decode_ms":r["per_generated_byte"]["median_ms"],
                         "state":r["retained_state_tensor_bytes"],
                         "raw_source":r["retained_source_prefix_bytes"],
                         "max_abs_error":r["source_proof"]["max_abs_logit"]}
                        for r in report["results"]]},sort_keys=True))

if __name__=="__main__":main()
