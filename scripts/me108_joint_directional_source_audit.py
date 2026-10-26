"""ME108 source-bound one-shot joint-input/ALL-role-parameters AD derivative audit.

No neural optimizer calls. Five actual original ME108 fitted SHA-pinned
checkpoints, genuine FIT-only environmental source, double-precision
independent NumPy joint directional derivative and source Torch autograd.
Finite differences are NUMERICAL evidence, not a Lean/PyTorch correctness proof.
"""
import json
from pathlib import Path

import numpy as np
import torch
from scripts import me108_role_training as run

OUT=Path(__file__).resolve().parents[1]/"research/mathematical-engineering/neural-refinement/ME108_JOINT_AD_RECEIPT.json"

def source_out(x,w1,b1,w2,b2):
    return np.tanh(x@w1.T+b1)@w2.T+b2

def joint_adjoint(x,w1,b1,w2,b2,dx,dw1,db1,dw2,db2,dy):
    hidden=np.tanh(x@w1.T+b1)
    dpre=dx@w1.T+x@dw1.T+db1
    dhidden=(1-hidden**2)*dpre
    jvp=dhidden@w2.T+hidden@dw2.T+db2
    dz=(dy@w2)*(1-hidden**2)
    vjp=dict(
        input=dz@w1,
        input_weight=dz.T@x,
        input_bias=dz.sum(axis=0),
        output_weight=dy.T@hidden,
        output_bias=dy.sum(axis=0))
    return jvp,vjp

def check():
    torch.set_num_threads(1)
    fit=run.fit_data()
    originals=run.budget_audit()["models"]
    if len(originals)!=5 or len({z["seed"] for z in originals})!=5:
        raise AssertionError("requires every actual ME108 source checkpoint")
    source_events=np.asarray(fit["general"][:8])
    samples=run.tensor_batch(fit,source_events,64,np.random.default_rng(108777))
    records=[]
    for identity in originals:
        seed=identity["seed"]
        ckpath=run.OUT/"role_guarded"/f"seed-{seed}"/"final.pt"
        if run.sha(ckpath)!=identity["checkpoint_sha256"]:
            raise AssertionError("original fitted SHA changed")
        ck=torch.load(ckpath,map_location="cpu",weights_only=False)
        if ck["seed"]!=seed or ck["steps"]!=4096:
            raise AssertionError("original trained lineage/steps mismatch")
        model=run.initialize(fit,seed).double()
        model.load_state_dict(ck["model"],strict=True)
        model.eval()
        xs=[v.to(torch.float64) if v.is_floating_point() else v for v in samples]
        with torch.no_grad():
            _,_,trace=model(*xs[:6],return_trace=True)
        features=trace["width_features"].detach().cpu().numpy().astype(np.float64)
        if features.shape!=(8,102) or not np.isfinite(features).all():
            raise AssertionError("invalid real source role input")
        for name,role in [("reconstruction",model.current_head),
                          ("width",model.width_head),("median",model.median_head)]:
            w1=role.input.weight.detach().numpy().astype(np.float64)
            b1=role.input.bias.detach().numpy().astype(np.float64)
            w2=role.output.weight.detach().numpy().astype(np.float64)
            b2=role.output.bias.detach().numpy().astype(np.float64)
            rng=np.random.default_rng(seed*100+{"reconstruction":1,"width":2,"median":3}[name])
            dx=rng.normal(size=features.shape)*.2
            dw1=rng.normal(size=w1.shape)*.2
            db1=rng.normal(size=b1.shape)*.2
            dw2=rng.normal(size=w2.shape)*.2
            db2=rng.normal(size=b2.shape)*.2
            dy=rng.normal(size=(features.shape[0],w2.shape[0]))*.2
            predicted,vjp=joint_adjoint(
                features,w1,b1,w2,b2,dx,dw1,db1,dw2,db2,dy)
            step=1e-4
            positive=source_out(features+step*dx,w1+step*dw1,
                b1+step*db1,w2+step*dw2,b2+step*db2)
            negative=source_out(features-step*dx,w1-step*dw1,
                b1-step*db1,w2-step*dw2,b2-step*db2)
            approximate=(positive-negative)/(2*step)
            max_error=float(np.max(np.abs(approximate-predicted)))
            np.testing.assert_allclose(approximate,predicted,rtol=1e-5,atol=2e-7)
            primal=source_out(features,w1,b1,w2,b2)
            xf=torch.from_numpy(features.copy()).requires_grad_(True)
            output=role(xf)
            torch.testing.assert_close(output,torch.tensor(primal),
                                       rtol=1e-10,atol=1e-10)
            grads=torch.autograd.grad((output*torch.from_numpy(dy)).sum(),
                (xf,role.input.weight,role.input.bias,
                    role.output.weight,role.output.bias))
            expected=(vjp["input"],vjp["input_weight"],vjp["input_bias"],
                      vjp["output_weight"],vjp["output_bias"])
            max_vjp_error=0.
            for got,expected_array in zip(grads,expected):
                torch.testing.assert_close(got,torch.tensor(expected_array),
                    rtol=1e-9,atol=1e-10)
                max_vjp_error=max(max_vjp_error,
                    float((got-torch.tensor(expected_array)).abs().max()))
            lhs=float(np.sum(dy*predicted))
            rhs=float(sum(np.sum(a*b) for a,b in zip(expected,
                         (dx,dw1,db1,dw2,db2))))
            if abs(lhs-rhs)>1e-9*max(1,abs(lhs),abs(rhs)):
                raise AssertionError("joint VJP is not adjoint of joint JVP")
            records.append(dict(seed=seed,head=name,real_fit_features=8,
                full_input_weight_bias_joint_linearization=True,
                independent_finite_difference_max_abs_error=max_error,
                numerical_5block_adjoint_abs_error=abs(lhs-rhs),
                torch_autograd_to_numpy_max_abs_error=max_vjp_error,
                original_checkpoint_sha256=identity["checkpoint_sha256"]))
    report=dict(kind="ME108_ORIGINAL_SOURCE_JOINT_AD_CHECK_NO_FORMAL_SOURCE_PROOF",
        source="UCI501 first eight FIT-only 2013-2014 origins",
        checkpoint_count=5,heads_per_checkpoint=3,directional_tests=15,
        numerical_scalar_precision="float64",
        source_parameters_modified=False,new_training_steps=0,
        protected_2016_2017_opened=False,
        lean_full_model_IEEE_and_probabilistic_source_refinement_complete=False,
        records=records)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    if OUT.exists():
        old=json.loads(OUT.read_text())
        if old!=report:
            raise AssertionError("recorded original source result differs; do not overwrite")
        print("ME108_ORIGINAL_JOINT_AD_REPLAY=PASS")
    else:
        OUT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
        print("ME108_ORIGINAL_JOINT_AD=PASS")
    print("FIVE_SHA_VERIFIED_MODELS="+str(report["checkpoint_count"]))
    print("FIFTEEN_FULL_JOINT_VJPS="+str(report["directional_tests"]))
    print("NO_TRAINED_WEIGHTS_OR_PROTECTED_OUTCOMES_CHANGED=PASS")

if __name__=="__main__":
    check()
