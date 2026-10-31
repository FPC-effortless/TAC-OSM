#!/usr/bin/env python3
"""SENSE-TOK-005 reproducible publisher-matched source-only mechanism diagnostic.

No training, no optimizer, no held-out TEST, no paid compute or capability claim.
"""
from __future__ import annotations
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter_ns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from diagnose_adaptive_byte_token_nofit import SOURCE_BLOBS,SOURCE_COMMIT,SOURCE_TAG,git_blob


def diagnose(source_root: Path, *, max_bytes=256, seeds=(0,1,2)):
    import torch
    from tac_osm.adaptive_codebook_token import AdaptiveCodebookByteTokenGRU,codebook_parameter_receipt
    from tac_osm.causal_byte_probe import predecision_examples
    if not (32 <= max_bytes <= 512):
        raise ValueError("source-only diagnostic size limited to 32..512 bytes")
    if not seeds or len(seeds)!=len(set(seeds)) or any(type(x) is not int or x<0 for x in seeds):
        raise ValueError("invalid frozen seed set")
    torch.set_num_threads(1)
    originals=[]
    for path,pinned in SOURCE_BLOBS.items():
        file=source_root/path
        if not file.is_file() or file.is_symlink():
            raise RuntimeError("publisher original absent: "+path)
        raw=file.read_bytes()
        if git_blob(raw)!=pinned:
            raise RuntimeError("publisher blob identity invalid: "+path)
        originals.append((path,sha256(raw).hexdigest(),raw[:max_bytes]))
    rows=[]
    for seed in seeds:
        torch.manual_seed(seed)
        model=AdaptiveCodebookByteTokenGRU(code_count=8).eval()
        receipt=codebook_parameter_receipt(model)
        if not receipt["within_100k_band"] or not receipt["within_1pct_s004"]:
            raise RuntimeError("parameter-parity precondition failed")
        for path,source_hash,prefix in originals:
            ids,_,_=predecision_examples(prefix,"fixed")
            for force in (None,"byte","token"):
                with torch.no_grad():
                    start=perf_counter_ns()
                    logits,(audit,),cost=model(ids[None],force=force,audit=True)
                    duration=perf_counter_ns()-start
                audit.verify(prefix[:-1])
                if logits.shape!=(1,len(prefix),256) or not bool(torch.isfinite(logits).all()):
                    raise AssertionError("source-to-logits failure")
                counts=Counter(c.code_id for c in audit.codes)
                rows.append({
                    "file":path,"seed":seed,"mode":force or "untrained_adaptive",
                    "publisher_full_sha256":source_hash,
                    "observed_prefix_sha256":sha256(prefix).hexdigest(),
                    "byte_readouts":audit.base.byte_readouts,
                    "observed_bytes":audit.base.observed_bytes,
                    "recurrent_updates":audit.recurrent_updates,
                    "token_ids_committed":len(audit.codes),
                    "active_token_code_ids":audit.active_code_count,
                    "dominant_code_share":audit.dominant_code_share,
                    "code_usage":dict(sorted(counts.items())),
                    "pending_bytes":len(audit.base.pending),
                    "mean_gate_p_byte_untrained":sum(audit.base.byte_probabilities)/max(1,len(audit.base.byte_probabilities)),
                    "proxy_cost_not_flops":float(cost.detach()),
                    "cpu_time_ms_uncontrolled":duration/1e6,
                    "passed_causal_reconstruction":True,
                })
    summary={}
    for name in ("byte","token","untrained_adaptive"):
        subset=[r for r in rows if r["mode"]==name]
        summary[name]={
            "total_readouts":sum(r["byte_readouts"] for r in subset),
            "recurrent_updates":sum(r["recurrent_updates"] for r in subset),
            "completed_token_ids":sum(r["token_ids_committed"] for r in subset),
        }
    return {
        "study_id":"SENSE-TOK-005",
        "class":"NO_FIT_REAL_SOURCE_IDENTITY_ONLY",
        "publisher":"python/cpython",
        "tag":SOURCE_TAG,
        "commit":SOURCE_COMMIT,
        "files":len(originals),
        "seeds":list(seeds),
        "trained_models":0,
        "optimizer_steps":0,
        "sealed_test_read":False,
        "superiority_eligible":False,
        "independent_evaluator_attestation":"ABSENT",
        "full_PNDS_math":"OPEN",
        "parameters":receipt,
        "summary":summary,
        "rows":rows,
        "warning":"Untrained arbitrary code IDs; no learned skill, end-to-end compute savings or semantic tokenization claim."
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source-root",type=Path,default=Path("/usr/lib/python3.13"))
    ap.add_argument("--max-bytes",type=int,default=256)
    ap.add_argument("--output",type=Path)
    args=ap.parse_args()
    report=diagnose(args.source_root,max_bytes=args.max_bytes)
    output=json.dumps(report,sort_keys=True,indent=2)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(output,encoding="utf-8")
    else: print(output,end="")
if __name__=="__main__":
    main()
