"""ME060 pure software gradient reachability tests, NOT real model performance."""
import torch
from torch import nn

from scripts.me060_audit_real_gradients import (
    gradient_reachability, exact_source_loss)
from scripts.me058_profile_original_fit import construct

torch.set_num_threads(1)


def test_gradient_reachability_counts_unused_parameters_separately():
    class Demo(nn.Module):
        def __init__(self):
            super().__init__()
            self.used=nn.Linear(2,2)
            self.unused=nn.Linear(2,2)
    m=Demo()
    m.used(torch.ones(2,2)).square().mean().backward()
    rows,count=gradient_reachability(m)
    assert count["declared_trainable"]==12
    assert count["gradient_present"]==6
    assert count["gradient_absent"]==6
    assert any(x["parameter"]=="unused.weight" and
               x["gradient_present"] is False for x in rows)


def test_actual_training_loss_graph_does_not_update_no_grad_feedback_paths():
    prior={"delta_scale":[.03]*14,"state_mean":[0.]*14,
           "state_scale":[1.]*14,"action_mean":[0.]*14,
           "action_scale":[1.]*14,"servo_gain":[.10]*14,
           "servo_step_limit":[.03]*14}
    torch.manual_seed(3)
    states=torch.randn(2,273,14)*.05
    actions=torch.randn(2,272,14)*.05
    human=torch.zeros(2,273,1,dtype=torch.bool)
    batch=(states,actions,human,((0,0),(0,0)))
    for name in ("native_tac","gru_world"):
        model=construct(name,prior)
        loss=exact_source_loss(name,model,batch)
        assert torch.isfinite(loss)
        loss.backward()
        rows,count=gradient_reachability(model)
        assert count["gradient_absent"]>0
        assert count["gradient_present"]>0
        parameters=dict(model.named_parameters())
        if name=="native_tac":
            assert parameters["feedback_map.weight"].grad is None
            assert parameters["feedback_key.weight"].grad is None
            assert parameters["native.retention"].grad is None
        else:
            assert parameters["feedback_cell.weight_ih"].grad is None
