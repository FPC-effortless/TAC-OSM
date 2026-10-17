from dataclasses import replace
import pytest
torch = pytest.importorskip("torch")
from tac_osm.development.schema_maintenance import make_pair
from tac_osm.development.public_state_training import public_tensors, student_rollout, prediction_from_state
from tac_osm.development.unified_state_prototype import UnifiedPredictiveStatePrototype


def test_public_encoding_is_equivariant_to_unseen_entity_permutations():
    torch.manual_seed(46)
    case = make_pair(46,32,32,"reissued_cancel","at")[0]
    perm = torch.tensor([3,7,5,0,1,6,2,4])
    renamed = tuple(replace(o,entity=int(perm[o.entity])) for o in case.observations)
    a = public_tensors([case.observations])
    b = public_tensors([renamed])
    assert torch.equal(a[0],b[0]) and torch.equal(perm[a[1]],b[1])
    model = UnifiedPredictiveStatePrototype(dim=8,budget=8)
    pa = prediction_from_state(model,model.encode(*a),torch.tensor([32.]))
    pb = prediction_from_state(model,model.encode(*b),torch.tensor([32.]))
    assert torch.equal(pa,pb[:,perm])


def test_student_rollout_only_applies_chosen_services_and_preserves_source_worlds():
    torch.manual_seed(46)
    cases = make_pair(46,32,32,"reissued_cancel","at")
    model = UnifiedPredictiveStatePrototype(dim=8,budget=8)
    state = model.encode(*public_tensors([c.observations for c in cases]))
    p = torch.zeros(2,8,2)
    p[:,0,0] = 1.
    next_state,y,counts = student_rollout(model,state,cases,p)
    assert counts=={"services":2,"inspections":2}
    assert y.shape==(2,8,2) and next_state is not state
    from tac_osm.development.schema_maintenance import future_bits
    assert all(future_bits(c.world,c.horizon)==c.wait_targets for c in cases)
