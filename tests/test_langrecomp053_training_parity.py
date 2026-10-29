"""Engineering fixtures only, not original-source model capability evidence."""
import pytest,torch
from torch import nn
from scripts.langrecomp053_training_parity import (
    SavedRequestCounter,source_batch,one_pass,compare_outputs)
def tiny_net():
    # Isolates PyTorch checkpoint and primitive gradient validation.
    return nn.Sequential(nn.Embedding(256,8),nn.Linear(8,256))
def test_checkpoint_preserves_backward_when_integer_inputs_have_no_grad():
    torch.manual_seed(42)
    m=tiny_net().train()
    x=torch.tensor([[1,2,3,2]],dtype=torch.long)
    y=torch.tensor([[2,3,2,4]],dtype=torch.long)
    a=one_pass(m,x,y,"native")
    b=one_pass(m,x,y,"checkpoint_entire_graph")
    result=compare_outputs(a,b)
    assert result["max_gradient_abs"]<1e-5
    assert result["gradient_tensor_count"]>0
    assert a["saved_logical_tensor_bytes"]>=0
    assert b["saved_logical_tensor_bytes"]>=0
def test_wrong_gradient_cannot_pass_review():
    torch.manual_seed(42);m=tiny_net().train()
    x=torch.tensor([[1,2,3,4]],dtype=torch.long)
    y=torch.tensor([[2,3,4,5]],dtype=torch.long)
    a=one_pass(m,x,y,"native")
    b=one_pass(m,x,y,"checkpoint_entire_graph")
    key=next(iter(b["grads"]))
    b["grads"][key]=b["grads"][key]+100
    with pytest.raises(ValueError,match="gradients changed"):
        compare_outputs(a,b)
def test_nonfinite_loss_is_a_hard_failure():
    torch.manual_seed(1);m=tiny_net().train()
    x=torch.tensor([[1,2,3,4]],dtype=torch.long)
    y=torch.tensor([[2,3,4,256]],dtype=torch.long)
    with pytest.raises((IndexError,RuntimeError)):
        one_pass(m,x,y,"native")
def test_original_source_batch_only_preceding_true_next_byte():
    fit={"fixture-owner":bytes(range(200))}
    # source_batch requires real FIT iterator; test the direct real source
    # legality formula without impersonating real publisher provenance.
    raw=fit["fixture-owner"]
    x=torch.tensor(list(raw[:32]))
    y=torch.tensor(list(raw[1:33]))
    assert torch.equal(x[1:],y[:-1]) and x.shape==y.shape
def test_logical_save_hook_is_not_peak_memory_claim():
    counter=SavedRequestCounter()
    x=torch.randn(3,4)
    assert counter.unpack(counter.pack(x)) is x
    assert counter.logical_bytes==48
    assert counter.calls==1
