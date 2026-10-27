"""NATIVELATENT003: verify actual original native memory retention transfer law.

Five SHA-restored ME111 original trained models, original real source FIT only.
The operator P schedule is captured from the actual native forward, not
guessed or inferred from an alternative architecture. State difference after
zeroing only the *previous* associative memory must equal M_prefix @ Π P_t
(up to FP32 rounding), conditional on all other state/source tensors unchanged.
No targets, new fits, protected/test rows or causal policy inference.
"""
import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

from scripts import me111_conditional_training as native
from scripts.native_latent_001_diagnosis import checked_index,read_frame,model_for,SEEDS,ORIGIN
from tac_osm.variants.parallel_affine_memory import associative_affine_schedule


def original_gate_transfer(model, full):
    T=full[0].shape[1]
    rows={}
    for n in (1,16,64,128):
        s=n
        pre=[x[:,:-s] if i!=4 and x.ndim>=2 else x for i,x in enumerate(full)]
        aft=[x[:,-s:] if i!=4 and x.ndim>=2 else x for i,x in enumerate(full)]
        _,past=model(*pre)
        captured={}
        def address_hook(_,__,out):
            if out.ndim==3:captured["key"]=F.normalize(out.detach().clone(),dim=-1,eps=1e-6)
        def content_hook(_,__,out):
            if out.ndim==3:captured["value"]=F.normalize(out.detach().clone(),dim=-1,eps=1e-6)
        def gate_hook(_,__,out):
            if out.ndim==3:captured["logits"]=out.detach().clone()
        hooks=[model.address.register_forward_hook(address_hook),
               model.content.register_forward_hook(content_hook),
               model.memory_gates.register_forward_hook(gate_hook)]
        try:
            forecast,unmodified=model(*aft,state=past)
        finally:
            for h in hooks:h.remove()
        memory_removed=replace(past,memory=torch.zeros_like(past.memory))
        other,removed=model(*aft,state=memory_removed)
        assert set(captured)=={"key","value","logits"}
        key=captured["key"];val=captured["value"]
        keep,write=captured["logits"].sigmoid().chunk(2,-1)
        alpha=keep*(1-write/4)
        usable=aft[1].any(-1,keepdim=True)
        P,Q=associative_affine_schedule(key,val,alpha,write,usable)
        # The exact chronological suffix memory transport of any prefix memory
        # is right multiplication by P0 P1 ... P_(n-1).
        transfer=torch.eye(8,dtype=P.dtype).expand(P.shape[0],8,8).clone()
        for t in range(n):
            transfer=transfer@P[:,t]
        expected=past.memory@transfer
        actual=unmodified.memory-removed.memory
        torch.testing.assert_close(actual,expected,atol=3e-5,rtol=5e-5)
        ratio=actual.flatten(1).norm(dim=1)/past.memory.flatten(1).norm(dim=1).clamp_min(1e-8)
        spectral=torch.linalg.matrix_norm(transfer,ord=2)
        loss=abs(forecast.log_location-other.log_location).mean()
        rows[str(n)]={
            "effective_alpha_mean":float(alpha.mean()),
            "controlled_write_beta_mean":float(write.mean()),
            "historical_memory_transfer_ratio_mean":float(ratio.mean()),
            "transfer_operator_spectral_norm_mean":float(spectral.mean()),
            "max_native_affine_memory_transfer_abs_error":float((expected-actual).abs().max()),
            "mean_abs_forecast_location_difference_after_prefix_memory_zeroed":float(loss),
            "sources_writable_fraction":float(usable.float().mean()),
            "operator_identity_checked_in_native_fp32":True,
        }
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    a=parser.parse_args()
    if a.output.exists():raise RuntimeError("original operator audit exists; no rewrite")
    torch.set_num_threads(1)
    d=native.fit_data()
    if d["phase"]!="fit" or hashlib.sha256(native.data.ARCHIVE.read_bytes()).hexdigest()!=native.data.SHA:
        raise RuntimeError("source SHA/phase discrepancy")
    i=checked_index(native)
    cases=read_frame(d,[(station,ORIGIN) for station in range(12)],512)
    models=[]
    with torch.no_grad():
        for seed in SEEDS:
            m,path,sha=model_for(native,d,seed,i[seed])
            versions=m.versions()
            transfer=original_gate_transfer(m,cases)
            if m.versions()!=versions or hashlib.sha256(path.read_bytes()).hexdigest()!=sha:
                raise RuntimeError("model mutated during read-only audit")
            row={"seed":seed,"original_checkpoint_sha256":sha,
                 "direct_fp32_native_memory_transfer":transfer}
            models.append(row)
            print("NATIVE_MEMORY_OPERATOR_SEED_JSON="+json.dumps(row,sort_keys=True),flush=True)
    out={"study":"NATIVELATENT003","source_sha256":native.data.SHA,
         "population":"12 authentic original Beijing 2014 FIT stations, fixed hour12000, source-only H512, no targets",
         "exact_scalar_theorem_scope":"M_out difference equals M_prefix times chronological product of source-conditioned affine P matrices, conditional on other source state fixed",
         "float32_result_type":"source-toleranced (3e-5 abs,5e-5 relative) original trained numerical comparison; NOT Lean IEEE proof",
         "no_optimizer":True,"no_original_weights_modified":True,
         "no_2015_dev_or_protected_numeric":True,
         "does_not_prove_key_retrieval_accuracy":True,
         "models":models}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print("NATIVE_MEMORY_OPERATOR_STATUS=PASS",flush=True)

if __name__=="__main__":main()
