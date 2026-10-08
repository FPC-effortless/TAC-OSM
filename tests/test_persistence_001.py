import torch

from tac_osm.persistence_001_benchmark import make_collision_pair
from tac_osm.persistence_001 import PersistencePLM001


def test_collision_pair_has_identical_current_observation_and_opposite_targets():
    pair = make_collision_pair(__import__("random").Random(7), current_bit=0)
    a, b = pair
    assert torch.equal(a.current_text, b.current_text)
    assert torch.equal(a.current_image, b.current_image)
    assert torch.equal(a.current_audio, b.current_audio)
    assert a.target != b.target


def test_no_memory_ceiling_is_half_on_balanced_collision():
    pair0 = make_collision_pair(__import__("random").Random(8), current_bit=0)
    pair1 = make_collision_pair(__import__("random").Random(9), current_bit=1)
    assert {e.memory_bit for e in pair0} == {0,1}
    assert {e.memory_bit for e in pair1} == {0,1}


def test_state_write_receives_gradient():
    torch.manual_seed(0)
    model = PersistencePLM001()
    b = 4
    t0 = (
        torch.tensor([[2,11,3,7],[2,12,3,7],[2,11,3,7],[2,12,3,7]]),
        torch.randn(b,1,8,8).sigmoid(),
        torch.randn(b,32),
    )
    current = t0
    logits, _ = model(t0, tuple(), current)
    target = torch.tensor([0,1,1,0])
    torch.nn.functional.cross_entropy(logits, target).backward()
    g = model.write_delta[0].weight.grad
    assert g is not None
    assert torch.isfinite(g).all()
    assert float(g.abs().sum()) > 0.0
