from dataclasses import replace

import pytest
import torch

from .test_model import tiny
from .steady import SteadyTACSession


def test_frozen_steady_matches_eager_across_pages_and_large_clock():
    model,features,boxes,refs=tiny("tac")
    model.eval()
    with torch.inference_mode():
        _,state=model(torch.tensor([[256,1,2,3],[256,4,5,6]]),refs,owners=("a","b"))
        steady=SteadyTACSession(model,refs,state)
        # Change absolute clock after tracing: a traced constant would fail.
        state=replace(state,clock=2**24+17)
        eager,fast=state,state
        for i in range(12):
            if i==5:
                refs=model.prepare_visual(features+1,boxes,owners=("a","b"),page_ids=("p2","p2"))
            token=torch.tensor([[i],[i+2]])
            expected,eager=model.step(token,refs,owners=("a","b"),state=eager)
            actual,fast=steady.step(token,refs,owners=("a","b"),state=fast)
            torch.testing.assert_close(actual,expected,atol=1e-9,rtol=1e-8)
            for name,value in fast.tensors().items():
                torch.testing.assert_close(value,eager.tensors()[name],atol=1e-9,rtol=1e-8)
                assert value.untyped_storage().nbytes()==value.numel()*value.element_size()
        with pytest.raises(ValueError,match="owner"):
            steady.step(token,refs,owners=("other","b"),state=fast)
        next(model.parameters()).add_(.01)
        with pytest.raises(ValueError,match="weights"):
            steady.step(token,refs,owners=("a","b"),state=fast)


def test_frozen_path_rejects_training_and_unfilled_cache():
    model,_,_,refs=tiny("tac")
    with pytest.raises(ValueError,match="eval"):
        SteadyTACSession(model,refs,model.start(("a","b")))
    model.eval()
    with pytest.raises(ValueError,match="prefill"):
        SteadyTACSession(model,refs,model.start(("a","b")))
