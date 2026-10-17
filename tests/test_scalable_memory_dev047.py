import pytest
torch = pytest.importorskip("torch")
from tac_osm.development.scalable_maintenance import make_pair, public_reference, catalog_tensors
from tac_osm.development.local_memory_variants import build_variant


@pytest.mark.parametrize("entities",[8,16,32])
@pytest.mark.parametrize("multiple",[False,True])
def test_independent_public_queue_matches_simulator_for_unseen_catalogs(entities,multiple):
    for role in ("pending_cancel","reissued_cancel"):
        for deadline in ("before","at","after"):
            for case in make_pair(47,128,32,role,deadline,entities=entities,multiple_pending=multiple,unseen_ids=True):
                assert public_reference(case.observations,case.entity_ids,case.tick,case.horizon)==case.wait_targets


@pytest.mark.parametrize("variant",["multiscale_gru","fast_gru","selective_diagonal","temporal_order"])
def test_shared_local_weights_transfer_to_32_new_entity_addresses(variant):
    a = build_variant(variant,47,8)
    b = build_variant(variant,47,32)
    b.load_state_dict(a.state_dict())
    case = make_pair(47,64,32,"reissued_cancel","at",entities=32,unseen_ids=True)[0]
    state = b.encode(*catalog_tensors([case.observations],[case.entity_ids]))
    assert state.fast.shape==(1,32,8)
    assert torch.isfinite(state.fast).all()
    unchanged = b.memory.step(state,torch.randn(1,8),torch.tensor([3]),torch.tensor([65.]),torch.tensor([False]))
    assert torch.equal(state.fast,unchanged.fast) and torch.equal(state.event,unchanged.event)
