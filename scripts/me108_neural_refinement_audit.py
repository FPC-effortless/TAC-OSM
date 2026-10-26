"""ME108 read-only source audit for PNDS exact-real neural-adjoint boundary.

Runs real FIT-only data and authentic Git-owned trained ME108 weights. Checks
the 3 102->16->12/12/6 role FFN parameter/input VJPs against an independent
NumPy implementation, and censored probability-law boundary behavior. This is
not a neural-network proof, IEEE verification or training/evaluation.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import json
import math
import numpy as np
import torch
from scripts import me108_role_training as run

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"research/mathematical-engineering/neural-refinement/ME108_SOURCE_AUDIT.json"

def independent_ffn(x,p,dy):
    """Reference uses ONLY NumPy arithmetic and source parameter copies."""
    w1=p.input.weight.detach().cpu().numpy().astype(np.float64)
    b1=p.input.bias.detach().cpu().numpy().astype(np.float64)
    w2=p.output.weight.detach().cpu().numpy().astype(np.float64)
    b2=p.output.bias.detach().cpu().numpy().astype(np.float64)
    z=x@w1.T+b1
    hidden=np.tanh(z)
    out=hidden@w2.T+b2
    local=(dy@w2)*(1.0-hidden**2)
    return out,dict(
        input=local@w1,
        first_weight=local.T@x,
        first_bias=local.sum(axis=0),
        second_weight=dy.T@hidden,
        second_bias=dy.sum(axis=0),
    )

def audit():
    torch.set_num_threads(1)
    d=run.fit_data()  # UCI501 FIT 2013-2014 only; protected 2016-17 unopened.
    idx=run.budget_audit()
    if len(idx["models"])!=5:
        raise RuntimeError("all five ORIGINAL source-registered model seeds needed")
    # Same FIT-only fixed observations for every original checkpoint; no model fit.
    choices=np.asarray(d["general"][:8])
    x0=run.tensor_batch(d,choices,64,np.random.default_rng(108777))
    records=[]
    for row in idx["models"]:
        seed=row["seed"]
        filename=run.OUT/"role_guarded"/f"seed-{seed}"/"final.pt"
        if run.sha(filename)!=row["checkpoint_sha256"]:
            raise RuntimeError("immutable trained neural checkpoint SHA changed")
        original=torch.load(filename,map_location="cpu",weights_only=False)
        if original["seed"]!=seed or original["steps"]!=4096:
            raise RuntimeError("checkpoint lineage mismatch")
        for dtype,rtol,atol in [(torch.float32,2e-5,2e-6),(torch.float64,1e-9,1e-10)]:
            model=run.initialize(d,seed).to(dtype)
            model.load_state_dict(original["model"],strict=True)
            model.eval()
            x=[v.to(dtype) if v.is_floating_point() else v for v in x0]
            forecast,_,trace=model(*x[:6],return_trace=True)
            features=trace["width_features"].detach()
            if features.shape!=(8,102):
                raise AssertionError("source ME108 features must be eight 102-vectors")
            if any(not torch.isfinite(q).all() for q in (
                    forecast.log_location,forecast.left_scale,forecast.right_scale,features)):
                raise AssertionError("no nonfinite real model or source values")
            role_records=[]
            for label,head,outdim in (
                ("reconstruct",model.current_head,12),
                ("width",model.width_head,12),
                ("median",model.median_head,6),
            ):
                fx=features.clone().requires_grad_(True)
                seed_grad=torch.linspace(-.5,.7,outdim,dtype=dtype)[None,:].expand(len(fx),-1)
                y=head(fx)
                grads=torch.autograd.grad((y*seed_grad).sum(),
                    (fx,head.input.weight,head.input.bias,
                     head.output.weight,head.output.bias),
                    allow_unused=False,retain_graph=False)
                reference,rg=independent_ffn(fx.detach().cpu().numpy().astype(np.float64),
                    head,seed_grad.detach().cpu().numpy().astype(np.float64))
                expected=(rg["input"],rg["first_weight"],rg["first_bias"],
                          rg["second_weight"],rg["second_bias"])
                torch.testing.assert_close(y,torch.tensor(reference,dtype=dtype),
                    rtol=rtol,atol=atol)
                largest=0.0
                for got,expect in zip(grads,expected):
                    truth=torch.tensor(expect,dtype=dtype)
                    torch.testing.assert_close(got,truth,rtol=rtol,atol=atol)
                    largest=max(largest,float((got-truth).abs().max()))
                role_records.append(dict(role=label,output_dimension=outdim,
                    original_source_parameter_VJP_verified=True,
                    original_source_feature_VJP_verified=True,
                    max_abs_gradient_error=largest))
            # The source mathematical CDF explicitly adds a boundary atom at y=0.
            # This tests source numeric behavior, NOT measure normalization.
            zero=torch.zeros_like(forecast.log_location)
            negative=-torch.ones_like(zero)
            p0=model.cdf(forecast,zero)
            below=model.cdf(forecast,negative)
            if not(torch.isfinite(p0).all() and ((p0>=0)&(p0<=1)).all()
                   and torch.equal(below,torch.zeros_like(below))):
                raise AssertionError("invalid censored atom or negative support")
            quantile=model.raw_quantiles(forecast)
            if quantile.shape!=(8,6,3) or not (quantile[:,:,1]>=0).all():
                raise AssertionError("invalid source quantile contract")
            # For censored medians Phi(quantile_50)>=0.5, with equality away
            # from a zero atom. Exact numeric equality is NOT claimed.
            pmedian=model.cdf(forecast,quantile[:,:,1])
            if not (pmedian>=(0.5-1e-6)).all():
                raise AssertionError("median inverse CDF contract failed")
            records.append(dict(seed=seed,dtype=str(dtype),
                original_checkpoint_sha256=row["checkpoint_sha256"],roles=role_records,
                source_censored_negative_support_verified=True,
                source_zero_atom_range_verified=True,
                source_median_quantile_checked=True,
                no_training_or_optimizer_steps=True))
    report=dict(status="ACTUAL_SOURCE_NUMERICAL_TEST_NOT_FORMAL_PROOF",
        source="UCI501 original FIT 2013-2014; five Git-frozen fitted ME108 checkpoints",
        only_original_fit_rows=True,protected_2016_2017_opened=False,
        checkpoint_count=5,precision_checks=10,role_vjp_checks=30,
        source_shape_102_16_12_12_6=True,
        real_source_IEEE_full_refinement=False,
        real_source_auto_differentiation_formal_refinement=False,
        normal_law_measure_normalization_formal_refinement=False,
        trained_checkpoint_weights_unchanged=True,records=records)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    if OUT.exists():
        old=json.loads(OUT.read_text())
        # Source owner is free to replay; never mutate a recorded original.
        if old!=report:
            raise RuntimeError("immutable original source audit differs on replay")
        print("ME108_NEURAL_REFINEMENT_AUDIT_REPLAY=PASS")
    else:
        OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
        print("ME108_NEURAL_REFINEMENT_AUDIT_ORIGINAL=PASS")
    return report

if __name__=="__main__":
    argparse.ArgumentParser().parse_args()
    audit()
