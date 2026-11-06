"""CSL-125-R4 model integration no-fit tests; none measure tool-use accuracy."""
import pytest
import torch

from tac_osm.csl125_r3 import CSL125R3, CSL125R3Config
from tac_osm.csl125_tool_model import CSL125R4
from tac_osm.csl125_tool_protocol import (
    ByteFallbackCodec, ToolContractError, catalogue, prepare_tool_supervision,
)


def cfg():
    return CSL125R3Config(vocab_size=256, width=32, ff_width=64, layers=4,
        role_count=4, state_width=8, rungs=(2,4), attention_layers=(2,4),
        query_heads=4, kv_heads=2, attention_window=16, correction_rank=4)


def tools():
    return catalogue([{
        "name":"weather",
        "description":"Look up the weather by city",
        "parameters":{"type":"object","properties":{
            "city":{"type":"string","maxLength":80}},
            "required":["city"],"additionalProperties":False},
    }])


def training_example():
    item=prepare_tool_supervision("Weather in Jos",tools(),[
        {"name":"weather","arguments":{"city":"Jos"}}])
    ids=torch.tensor([item.token_ids],dtype=torch.long)
    mask=torch.tensor([item.target_mask],dtype=torch.bool)
    return ids,mask,item


def test_full_default_source_has_identical_123873263_budget():
    with torch.device('meta'):
        old=CSL125R3(device='meta')
        new=CSL125R4(device='meta')
    assert old.parameter_count()==new.parameter_count()==123_873_263
    assert new.parameter_count()<=125_000_000
    assert tuple(old.state_dict())==tuple(new.state_dict())
    assert all(p.requires_grad for p in new.parameters())


def test_r4_sft_loss_is_finite_and_model_gradients_exist():
    torch.manual_seed(6)
    model=CSL125R4(cfg()).train()
    ids,mask,_=training_example()
    loss=model.tool_sft_loss(ids,mask)
    assert torch.isfinite(loss)
    loss.backward()
    grads=dict(model.named_parameters())
    for name in ("embedding.weight","blocks.0.write_gate.weight",
                 "blocks.1.up.weight"):
        assert grads[name].grad is not None, name
        assert torch.isfinite(grads[name].grad).all(), name
    assert not torch.allclose(grads["embedding.weight"].grad,
                              torch.zeros_like(grads["embedding.weight"].grad))
    assert grads["accept_head.weight"].grad is None
    assert grads["error_head.weight"].grad is None


def test_r3_r4_parity_before_any_tool_supervision_or_fit():
    torch.manual_seed(9)
    parent=CSL125R3(cfg()).eval()
    child=CSL125R4(cfg()).eval()
    child.load_state_dict(parent.state_dict(),strict=True)
    x=torch.randint(0,256,(1,19))
    with torch.no_grad():
        old,sold=parent(x,depth=4,stream_ids=("case",))
        new,snew=child(x,depth=4,stream_ids=("case",))
    for k in old[4]:
        assert torch.equal(old[4][k],new[4][k])
    for a,b in zip(sold.layers,snew.layers):
        for k in ("fast","fast_mass","slow","slow_mass","conv_tail"):
            assert torch.equal(getattr(a,k),getattr(b,k))


def test_loss_mask_refuses_missing_prompt_and_packed_document_islands():
    torch.manual_seed(11)
    model=CSL125R4(cfg())
    ids,mask,_=training_example()
    with pytest.raises(ToolContractError,match="suffix"):
        model.tool_sft_loss(ids,torch.zeros_like(mask))
    with pytest.raises(ToolContractError,match="suffix"):
        model.tool_sft_loss(ids,torch.cat((mask[:,:-2],
            torch.tensor([[False,True]])),dim=1))
    with pytest.raises(ToolContractError,match="nonempty prompt"):
        model.tool_sft_loss(ids,torch.ones_like(mask))
    with pytest.raises(ToolContractError,match="correction_weight"):
        model.tool_sft_loss(ids,mask,correction_weight=float("nan"))
    with pytest.raises(ToolContractError,match="exit weights"):
        model.tool_sft_loss(ids,mask,weights={2:1.,4:-1.})


def test_unguided_random_weights_do_not_claim_valid_tool_use():
    torch.manual_seed(15)
    model=CSL125R4(cfg()).eval()
    result=model.propose_tools("Weather in Jos",tools(),depth=2,max_new_bytes=2)
    assert result.status in ("incomplete","invalid","abstain")
    assert not hasattr(model,"execute_tool")
    assert not hasattr(model,"call_tool")
    assert not hasattr(result,"confidence")
    assert result.raw_json.startswith('{"function_calls":[')


def test_proposal_requires_eval_and_registered_exit():
    torch.manual_seed(4)
    model=CSL125R4(cfg()).train()
    with pytest.raises(ToolContractError,match=r"model.eval"):
        model.propose_tools("Weather in Jos",tools(),depth=2,max_new_bytes=2)
    model.eval()
    with pytest.raises(ToolContractError,match="inference exit"):
        model.propose_tools("Weather in Jos",tools(),depth=3,max_new_bytes=2)
    with pytest.raises(ToolContractError,match="1..512"):
        model.propose_tools("Weather in Jos",tools(),depth=2,max_new_bytes=513)
