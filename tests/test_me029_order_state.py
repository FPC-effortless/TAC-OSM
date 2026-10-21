"""ME029 numeric-only memory order/parameter/causality tests."""
import torch
from tac_osm.me029_order_state import (
    preflight, block_order_sort, OrderFreeNonlinearControl,
    BlockSortedNative, active_params
)
from tac_osm.me028_nonlinear_readout import NativeNonlinearObservableReadout


def test_me029_full_mathematical_causal_falsifiers():
    evidence=preflight()
    assert evidence["status"]=="ME029_NUMERICAL_ONLY_FALSIFIER_PASS"
    assert evidence["all_active"]
    assert evidence["causal_prefix_noninterference"]
    assert not evidence["reserved_data_read"]


def test_me029_exact_near_equal_active_parameter_counts():
    pop=torch.full((16000,),.325)
    a=NativeNonlinearObservableReadout(.5,.2,pop)
    b=BlockSortedNative(.5,.2,pop)
    c=OrderFreeNonlinearControl(.5,.2,pop)
    assert (active_params(a),active_params(b),active_params(c))==(
        135377,135377,135913)
    assert (active_params(c)-active_params(a))/active_params(a)<.01


def test_me029_preserve_item_metadata_pair_and_complete32_cut():
    x=torch.arange(0,128,dtype=torch.long).flip(0).repeat(2,1)
    meta=torch.arange(128,dtype=torch.float32).reshape(1,128,1)
    meta=torch.cat((meta,meta+200.,meta+400.),-1).repeat(2,1,1)
    sorted_items,sorted_meta=block_order_sort(x,meta)
    for r in range(2):
        for k in (32,64,96,128):
            assert torch.equal(torch.sort(sorted_items[r,:k]).values,
                               torch.sort(x[r,:k]).values)
            assert torch.equal(
                sorted_meta[r,:k,0],127.-sorted_items[r,:k].float())
