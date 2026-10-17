import pytest
torch = pytest.importorskip("torch")
from tac_osm.development.bounded_mtsk import build_bounded
from tac_osm.development.writer_reader_factorial import configure,is_reader,group_hash


def test_frozen_reservoir_keeps_internal_query_reader_trainable():
    m = configure(build_bounded(4411),"writer_frozen_reader_train")
    assert all(p.requires_grad==is_reader(n) for n,p in m.memory.named_parameters())
    assert m.event_encoder[0].weight.requires_grad and m.head[0].weight.requires_grad


def test_writer_update_does_not_depend_on_internal_query_reader_weights():
    m = build_bounded(4411)
    x = torch.randn(2,12)
    state = m.memory.initial(x)
    a = m.memory.update(state,x,torch.tensor([0,2]))
    before = group_hash(m,False)
    with torch.no_grad():
        for n,p in m.memory.named_parameters():
            if is_reader(n):
                p.add_(10.)
    b = m.memory.update(state,x,torch.tensor([0,2]))
    assert torch.equal(a,b) and group_hash(m,False)==before
