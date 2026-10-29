#!/usr/bin/env python3
"""LANGMEM081 label-blind true observed publisher source-context retrieval.

IMPORTANT: Historical context side-table holds past raw bytes and all
observed original source writer keys (tens/hundreds of entries). It is a
non-neural diagnostic; never report as part of native 4160B fixed state,
or as model's ability to reconstruct past source bytes from learned memory.

No fit/optimizer, no hidden TEST; four already EXPOSED real PSF DEV docs.
"""
from __future__ import annotations
import argparse,hashlib,json,math,sys,time
from collections import Counter
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
for n in ("langmem081","langmem080","langmem079","langmem078","langscale075",
          "langscale070","context026","langmemory042","langbase024"):
    p=str(ROOT/"research/mathematical-engineering"/n)
    if p not in sys.path:sys.path.insert(0,p)
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from dual_streaming import DualNativeStreamingTAC079
from scripts.langmem079_carried_eval import restore_original
from scripts.langmem080_event_trace import source_factors
from scripts.langstream076_carried_source_eval import score
from scripts.langlayer035_sampling import fit_sources
from byte_next_reference import original_source

LENGTHS=(2,3,4)
AGES=(32,128,256)
CLOCK_START=128
CLOCK_STOP=320
SEEDS=(1701,1702,1703)
TAU=8
VOCAB=256
CELLS=4*(CLOCK_STOP-CLOCK_START)*len(LENGTHS)*len(AGES)


def select_past(observed_prefix,owner,writer_keys,query,*,context_len,age_cap):
    """Frozen label-blind selection from raw<=t and keys written at a<t.

    observed_prefix contains *only* source bytes indexed 0..t inclusive.
    It is forbidden to pass raw[t+1] into this function. Past writer at a
    was derived from observed context through a-1 and actual event raw[a].
    Return clocks and ranking, but NO real future target or scoring.
    """
    if type(observed_prefix) is not bytes or not 129<=len(observed_prefix)<=320:
        raise ValueError("exact original observed prefix 129..320 required")
    if type(owner) is not str or not owner or not isinstance(writer_keys,dict):
        raise ValueError("invalid source owner/writer source index")
    if context_len not in LENGTHS or age_cap not in AGES:
        raise ValueError("unregistered causal context/horizon")
    if query.shape!=(8,) or query.dtype not in (torch.float32,torch.float64) or not torch.isfinite(query).all():
        raise ValueError("current learned native source query missing/invalid")
    t=len(observed_prefix)-1
    lo=max(4,t-age_cap)
    indices=tuple(range(lo,t))
    if not indices or any(a not in writer_keys for a in indices):
        raise ValueError("missing previous observed source writer keys")
    if any(a>=t or a<4 for a in indices):
        raise ValueError("future or invalid original source event included")
    stack=torch.stack([writer_keys[a] for a in indices])
    if stack.shape!=(len(indices),8) or not torch.isfinite(stack).all():
        raise ValueError("wrong source writer embeddings")
    norm=torch.linalg.vector_norm(stack,dim=1)
    if not ((norm>0.999).all() and (norm<1.001).all()):
        raise ValueError("source original writer addresses not normalized")
    cosine=(stack@query).tolist()
    order=sorted(range(len(indices)),key=lambda i:(-cosine[i],-indices[i]))
    current=observed_prefix[t-context_len+1:t+1]
    positives=tuple(i for i,a in enumerate(indices)
                    if observed_prefix[a-context_len:a]==current)
    wrong=tuple(i for i in range(len(indices)) if i not in positives)
    newest_positive=(indices[max(positives)] if positives else None)
    hardest_wrong=(indices[max(wrong,key=lambda i:(cosine[i],indices[i]))]
                   if wrong else None)
    # SHA selection depends on source identity and already observed time only.
    # Not trained, no dependency on source raw[t+1] or model seed.
    seed=hashlib.sha256(f"LANGMEM081|{owner}|{t}|{context_len}|{age_cap}".encode()).digest()
    random_index=int.from_bytes(seed[:8],"big")%len(indices)
    first=order[0]
    best_positive=min((order.index(i)+1 for i in positives),default=None)
    most_recent=indices[-1]
    matches=tuple(indices[i] for i in positives)
    counts=Counter(observed_prefix[a] for a in matches)
    # The source cache counts actual HISTORICAL observed bytes raw[a], no
    # current next byte label. Laplace/Dirichlet smoothing is frozen tau=8.
    return {
      "query_clock":t,"context_len":context_len,"age_cap":age_cap,
      "candidate_count":len(indices),"matching_clocks":matches,
      "matching_count":len(matches),"has_historical_match":bool(matches),
      "model_top_clock":indices[first],
      "model_top_context_match":bool(first in positives),
      "model_top_cosine":float(cosine[first]),
      "best_matching_rank":best_positive,
      "best_matching_reciprocal_rank":1/best_positive if best_positive else None,
      "best_matching_cosine":max((cosine[i] for i in positives),default=None),
      "hardest_wrong_clock":hardest_wrong,
      "hardest_wrong_cosine":max((cosine[i] for i in wrong),default=None),
      "historical_last_match_clock":newest_positive,
      "random_clock":indices[random_index],
      "recent_clock":most_recent,
      "observed_history_counts":dict(counts),
      "historical_context_observed":tuple(current),
      "source_clock_indices":indices,
    }


def add_outcome(chosen,observed_prefix,actual_nextbyte,model_score):
    """Score only AFTER original current-source model prediction and lookup.

    The target is intentionally a separate argument, absent from select_past.
    """
    if not isinstance(chosen,dict) or type(actual_nextbyte) is not int or not 0<=actual_nextbyte<=255:
        raise ValueError("genuine source withheld target or selector invalid")
    if type(observed_prefix) is not bytes or len(observed_prefix)!=chosen["query_clock"]+1:
        raise ValueError("scorer original source prefix mismatch")
    # Source raw[t+1] is not represented in observed_prefix.
    t=chosen["query_clock"]
    if t!=len(observed_prefix)-1:
        raise ValueError("source clock invalid")
    source_bytes={k:(observed_prefix[chosen[k]] if chosen[k] is not None else None)
                  for k in ("model_top_clock","hardest_wrong_clock",
                            "historical_last_match_clock","random_clock","recent_clock")}
    if any(chosen[k] is not None and not 4<=chosen[k]<t for k in source_bytes):
        raise ValueError("historic candidate accessed future/own current token")
    counts=chosen["observed_history_counts"]
    n=sum(counts.values())
    if n!=chosen["matching_count"]:
        raise ValueError("source context count inconsistent")
    prob=(counts.get(actual_nextbyte,0)+TAU/VOCAB)/(n+TAU)
    if not 0<prob<=1 or not math.isfinite(prob):
        raise ValueError("invalid precommitted source-history posterior")
    return {
      "owner":None,"clock":t,"context_len":chosen["context_len"],
      "age_cap":chosen["age_cap"],
      "query_context_observed":chosen["historical_context_observed"],
      "candidate_count":chosen["candidate_count"],
      "positive_match_count":n,"positive_exists":chosen["has_historical_match"],
      "model_writer_context_hit":chosen["model_top_context_match"],
      "model_best_matching_reciprocal_rank":chosen["best_matching_reciprocal_rank"],
      "model_best_matching_rank":chosen["best_matching_rank"],
      "model_top_writer_clock":chosen["model_top_clock"],
      "source_positive_recent_clock":chosen["historical_last_match_clock"],
      "hardest_wrong_writer_clock":chosen["hardest_wrong_clock"],
      "model_writer_cosine_margin_to_hard_wrong":(
          chosen["best_matching_cosine"]-chosen["hardest_wrong_cosine"]
          if chosen["best_matching_cosine"] is not None and
             chosen["hardest_wrong_cosine"] is not None else None),
      "model_writer_observed_continuation_correct":int(source_bytes["model_top_clock"]==actual_nextbyte),
      "hardest_wrong_observed_continuation_correct":(
          int(source_bytes["hardest_wrong_clock"]==actual_nextbyte)
          if source_bytes["hardest_wrong_clock"] is not None else None),
      "exact_recent_match_observed_continuation_correct":(
          int(source_bytes["historical_last_match_clock"]==actual_nextbyte)
          if source_bytes["historical_last_match_clock"] is not None else None),
      "uniform_random_observed_continuation_correct":int(source_bytes["random_clock"]==actual_nextbyte),
      "most_recent_any_observed_continuation_correct":int(source_bytes["recent_clock"]==actual_nextbyte),
      "exact_history_smoothed_source_nll":-math.log(prob),
      "actual_original_next_byte_target":actual_nextbyte,
      "original_trained_TAC_nextbyte_nll":model_score["nll"],
      "original_trained_TAC_nextbyte_top1":model_score["top1"]
    }


@torch.no_grad()
def evaluate(seed,original_ckpt,original_verified):
    torch.set_num_threads(2)
    fit=fit_sources()
    _,dev,_=original_source(ROOT)
    if len(fit)!=14 or len(dev)!=4 or set(fit)&set(dev):
        raise ValueError("original publisher owner partitions changed")
    model,receipt=restore_original(seed,original_ckpt,original_verified)
    e=DualNativeStreamingTAC079(model,"cap_all")
    if model.parameter_count()!=499627:
        raise ValueError("trained original model parameter identity")
    started=time.process_time()
    rows=[]
    targets_sha=hashlib.sha256()
    observations_sha=hashlib.sha256()
    max_prefix_logit_abs_error=0.
    for owner in sorted(dev):
        raw=dev[owner]
        if len(raw)<CLOCK_STOP+1:raise ValueError("genuine source owner shorter than preregistered clocks")
        keys={}
        state=e.start((owner,))
        for t in range(CLOCK_STOP):
            if state.clock!=t or state.tensor_bytes()!=4160:
                raise ValueError("original source model clock/ten-field state mismatch")
            current=torch.tensor([raw[t]],dtype=torch.long)
            # This frozen factor function only consumes prior state and current
            # observed source byte; current source target raw[t+1] NOT accessed.
            f=source_factors(model,state,current)
            model_logits,state=e.step(current,state,owners=(owner,))
            if 4<=t:
                keys[t]=f["write_key"].squeeze(0).clone()
            if t==127:
                prefix=torch.tensor(list(raw[:128]),dtype=torch.long).reshape(1,-1)
                original_full=model(prefix)[:,-1]
                diff=float((model_logits-original_full).abs().max())
                max_prefix_logit_abs_error=max(max_prefix_logit_abs_error,diff)
                if diff>3e-5:
                    raise ValueError("source original graph 128 prefix mismatch")
            if t<CLOCK_START:continue
            prefix=bytes(raw[:t+1])
            query=f["query"].squeeze(0)
            # All nine selectors receive causal observed source prefix ONLY;
            # real raw[t+1] is read below, strictly after choices have frozen.
            choices=[select_past(prefix,owner,keys,query,context_len=L,age_cap=H)
                for L in LENGTHS for H in AGES]
            # Model is also frozen. After this point source truth is read.
            y=int(raw[t+1])
            model_metrics=score(model_logits,y)
            targets_sha.update(owner.encode()+b":"+str(t).encode()+b":"+bytes([y])+b";")
            observations_sha.update(owner.encode()+b":"+str(t).encode()+b":"+
                bytes(prefix[t-3:t+1])+b";")
            for c in choices:
                z=add_outcome(c,prefix,y,model_metrics)
                z["owner"]=owner
                rows.append(z)
        if state.clock!=CLOCK_STOP:
            raise ValueError("source owner boundary broken")
    if len(rows)!=CELLS:
        raise ValueError("original source target cell grid incomplete")
    return {"study":"LANGMEM081","seed":seed,
      "status":"ORIGINAL_PUBLISHER_SOURCE_CONTEXT_KEY_MATCH_DIAGNOSTIC_NOT_NEURAL_RECALL",
      "original_model_checkpoint_sha256":receipt["checkpoint_sha256"],
      "original_fit_sample_order_sha256":receipt["source_sample_order_sha256"],
      "trained_original_parameters":499627,"native_state_fp32_bytes_per_owner":4160,
      "history_side_table":"NOT_PART_OF_4160BYTE_PERSISTENT_MODEL_STATE",
      "source_document_owners":tuple(sorted(dev)),
      "source_context_lengths":LENGTHS,"source_history_caps":AGES,
      "original_query_clock_begin":CLOCK_START,
      "original_query_clock_end_exclusive":CLOCK_STOP,
      "source_observed_nextbyte_targets_per_seed":len(dev)*(CLOCK_STOP-CLOCK_START),
      "source_observed_cohort_cells":CELLS,
      "actual_source_nextbyte_sha256":targets_sha.hexdigest(),
      "original_observed_context_sha256":observations_sha.hexdigest(),
      "original_graph_128_prefix_max_abs_logit_error":max_prefix_logit_abs_error,
      "original_model_no_new_fit":True,"optimizer_updates":0,
      "gpu_minutes":0,"protected_TEST":"NOT_ACCESSED",
      "rows":rows,"cpu_process_seconds":time.process_time()-started,
      "independently_qualified":False,
      "limitations":[
        "actual PSF publisher DEV original owners previously EXPOSED and 078 model/source postselected",
        "side-table holds actual past bytes and all original trained keys; NOT learned native 4160B state retrieval",
        "context match is equality of observed preceding bytes; it is not guaranteed same future continuation",
        "trained learned original address top writer may not represent model downstream causal contribution",
        "exact-history posterior tau8 and recent-match historical continuation are deterministic non-neural controls",
        "no model training, no protected TEST, no independent new publisher source generalization or mobile score",
        "source complete IEEE AD full model Lean proofs remain OPEN"]
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,choices=SEEDS,required=True)
    p.add_argument("--checkpoint-dir",type=Path,required=True)
    p.add_argument("--original-aggregate",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError("original source read-only receipt overwrite refused")
    result=evaluate(a.seed,a.checkpoint_dir,a.original_aggregate)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print("LANGMEM081_ACTUAL_CONTEXT_DIAGNOSTIC "+json.dumps({
      "seed":a.seed,"original_source_target_count":result["source_observed_nextbyte_targets_per_seed"],
      "source_cells":len(result["rows"]),
      "source_label_sha":result["actual_source_nextbyte_sha256"],
      "positive_coverage":{f"L{L}_H{H}":sum(x["positive_exists"] for x in result["rows"]
          if x["context_len"]==L and x["age_cap"]==H)/(4*192)
          for L in LENGTHS for H in AGES}
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
