"""Whole ES-WAM002 input-to-output executable graph trace, original weights unchanged.

Wraps ORIGINAL model operators rather than implementing an unrelated toy network.
Records each module's tensor input/output and checks complete model forward,
role writes, shape contracts, differentiable losses and backward propagation.
No optimizer update; this is NOT a training/real-world capability result.
"""
from __future__ import annotations
import argparse
import json
import torch
from tac_osm.variants.exchange_world_action import ExchangeWorldActionModel

def trace_whole_model(batch: int=3, dimension: int=14, action_dim: int=14):
    torch.manual_seed(10010)
    model=ExchangeWorldActionModel(dimension=dimension,action_dim=action_dim)
    stages=[]
    handles=[]
    for name,module in model.named_modules():
        if isinstance(module,torch.nn.Linear):
            def record(mod, inputs, output, stage=name):
                stages.append({"stage":stage,
                    "input":list(inputs[0].shape),
                    "output":list(output.shape),
                    "parameters":sum(p.numel() for p in mod.parameters())})
            handles.append(module.register_forward_hook(record))
    state=model.initial(batch)
    x=torch.randn(batch,dimension,requires_grad=True)
    action=torch.randn(batch,action_dim,requires_grad=True)
    elapsed=torch.full((batch,),.1)
    ids=torch.arange(batch,dtype=torch.long)
    mask=torch.ones(batch,dimension,dtype=torch.bool)
    goal=torch.randn(batch,action_dim)
    try:
        first=model(x,action,elapsed,ids,mask,goal,state)
        second=model(x+.01,action,elapsed,ids,mask,goal,first.state)
        target=torch.randn_like(second.mean_next)
        physical_mae=(second.mean_next-target).abs().mean()
        physical_nll=((second.mean_next-target).abs()/second.scale_next
                      +torch.log(2*second.scale_next)).mean()
        command_imitation=(second.proposed_action-action).abs().mean()
        total=physical_nll+.5*physical_mae+.05*command_imitation
        total.backward()
        gradient_modules={
            name:{"nonzero_gradient":bool(p.grad is not None and
                         torch.isfinite(p.grad).all() and (p.grad.abs().sum()>0)),
                  "finite_gradient":bool(p.grad is not None and torch.isfinite(p.grad).all())}
            for name,p in model.named_parameters()}
        result={
            "source_model":"tac_osm.variants.exchange_world_action.ExchangeWorldActionModel",
            "batch":batch,"state_dimension":dimension,"action_dimension":action_dim,
            "trainable_parameters":sum(p.numel() for p in model.parameters()),
            "forward_stages":stages,
            "output_shapes":{
                "mean_next":list(second.mean_next.shape),
                "scale_next":list(second.scale_next.shape),
                "proposed_action":list(second.proposed_action.shape),
                "fast":list(second.state.fast.shape),
                "slow":list(second.state.slow.shape),
                "memory_ids":list(second.state.memory.ids.shape),
                "memory_values":list(second.state.memory.values.shape)},
            "first_observation_written":first.observation_written.tolist(),
            "first_prediction_written":first.prediction_written.tolist(),
            "second_previous_prediction_present":second.previous_prediction_present.tolist(),
            "scale_strictly_positive":bool((second.scale_next>0).all()),
            "all_outputs_finite":bool(all(torch.isfinite(t).all() for t in
                      (second.mean_next,second.scale_next,second.proposed_action))),
            "input_gradient_finite":bool(x.grad is not None and torch.isfinite(x.grad).all()),
            "parameter_gradients":gradient_modules,
            "note":"Executable original full-model graph with example numerical data; not real-data validation or formal verification of nonlinear semantics."
        }
        assert result["all_outputs_finite"] and result["scale_strictly_positive"]
        assert result["input_gradient_finite"]
        assert all(v["finite_gradient"] for v in gradient_modules.values())
        return result
    finally:
        for h in handles:h.remove()

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--batch",type=int,default=3)
    p.add_argument("--output",default="")
    args=p.parse_args()
    result=trace_whole_model(args.batch)
    rendered=json.dumps(result,indent=2,sort_keys=True)
    if args.output:
        from pathlib import Path
        path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(rendered+"\n")
    print(rendered)
