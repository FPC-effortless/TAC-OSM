"""ME028 zero-head algebra, nonzero causal prefixes and native recurrence integrity."""
import torch
from tac_osm.state_value import StateValueModel
from tac_osm.me028_nonlinear_readout import (
    NativeNonlinearObservableReadout,
    original_parameter_count,
    native_nonlinear_math_preflight
)


def test_me028_full_analytic_native_nonlinear_causality():
    result=native_nonlinear_math_preflight()
    assert result["status"]=="ME028_MATH_AND_PREFIX_FALSIFIER_PASS"
    assert result["readout_extra"]==310
    assert result["recurrence_unchanged"]
    assert not result["reserved_labels_read"]


def test_me028_exact_fitted_parameter_budget():
    pop=torch.full((16000,),.325)
    base=StateValueModel("tac",0.4,0.2,pop)
    candidate=NativeNonlinearObservableReadout(0.4,0.2,pop)
    mlp=StateValueModel("mlp",0.4,0.2,pop)
    assert tuple(original_parameter_count(m) for m in (base,candidate,mlp))==(
        135067,135377,135057)
    assert abs(original_parameter_count(candidate)-original_parameter_count(base))==310


def test_me028_no_new_recurrent_state_matrices():
    pop=torch.full((16000,),.3)
    base=StateValueModel("tac",0.3,0.1,pop)
    modified=NativeNonlinearObservableReadout(0.3,0.1,pop)
    recurrent=("enc.","transition.","gate.","slow.","slow_gate.")
    a={n:tuple(p.shape) for n,p in base.named_parameters()
       if n.startswith(recurrent)}
    b={n:tuple(p.shape) for n,p in modified.named_parameters()
       if n.startswith(recurrent)}
    assert a==b
    assert len(a)>0


def test_me028_nonlinear_extra_head_receives_trainable_gradients():
    pop=torch.full((16000,),.325)
    model=NativeNonlinearObservableReadout(0.3,0.1,pop)
    assert model.observable_hidden.weight.requires_grad
    assert model.observable_output.weight.requires_grad
    assert model.head.weight.requires_grad
    assert model.observable_output.bias.shape==(1,)
