"""LANGRECOMP054 fixtures: scope-only checkpoint intercept and gradient preservation."""
import pytest,torch
from torch import nn
from scripts.langrecomp054_warmed_scoped import (
    checkpoint_original_upper_only,execute,dist,TIMED)
from scripts.langrecomp053_training_parity import compare_outputs
class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=nn.Embedding(256,12)
        self.layernorm=nn.Sequential(nn.LayerNorm(12),nn.Linear(12,12),nn.Tanh())
        self.head=nn.Linear(12,256)
    def forward(self,x):
        return self.head(self.layernorm(self.embedding(x)))
def test_scoped_upper_reproduces_all_parameter_gradients():
    torch.manual_seed(12);m=Toy().train()
    x=torch.tensor([[1,2,3,4,5]],dtype=torch.long)
    y=torch.tensor([[2,3,4,5,6]],dtype=torch.long)
    a=execute(m,x,y,"native")
    b=execute(m,x,y,"scoped_upper")
    c=execute(m,x,y,"checkpoint_entire_graph")
    assert compare_outputs(a,b)["max_gradient_abs"]==0
    assert compare_outputs(a,c)["max_gradient_abs"]==0
def test_orig_layernorm_forward_is_restored():
    m=Toy().train()
    original=m.layernorm.forward
    with checkpoint_original_upper_only(m):
        assert m.layernorm.forward!=original
    assert m.layernorm.forward==original
def test_wrong_arm_never_silently_uses_native():
    m=Toy().train()
    x=torch.tensor([[0,1,2]],dtype=torch.long)
    with pytest.raises(ValueError,match="unsupported"):
        execute(m,x,x,"surprise")
def test_warm_timing_summary_needs_exact_repeats():
    with pytest.raises(ValueError,match="matched"):
        dist([1.,2.])
    assert dist([1.,2.,3.,4.,5.])["median"]==3.
def test_scope_does_not_reallocate_or_rename_registered_parameters():
    m=Toy().train()
    before={n:id(p) for n,p in m.named_parameters()}
    with checkpoint_original_upper_only(m):
        assert {n:id(p) for n,p in m.named_parameters()}==before
    assert {n:id(p) for n,p in m.named_parameters()}==before
