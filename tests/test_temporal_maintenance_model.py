"""DEV-016 gradient and public-interface gates, CPU PyTorch required."""
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_temporal_maintenance_dev import matching_models, tensors  # noqa: E402
from tac_osm.development.casm_readout_alignment import CASMAlignedModel  # noqa: E402
from tac_osm.development.delayed_maintenance import make_pair  # noqa: E402
from tac_osm.development.temporal_maintenance_model import TemporalMaintenanceModel  # noqa: E402


def test_matching_initialization_and_sole_768_scalar_state():
    models, hashes = matching_models(CASMAlignedModel(), 43)
    assert len(set(hashes.values())) == 1
    assert len({sum(p.numel() for p in m.parameters()) for m in models.values()}) == 1
    cases = make_pair(43, 32, 4, role="pending_cancel")
    inputs = tensors(cases)
    for m in models.values():
        action, future, auxiliary = m(*inputs)
        assert action.shape == (2, 2)
        assert future.shape == auxiliary.shape == (2, 2, 2)
        assert m.last_state_scalars == 768


def test_actual_action_gradient_reaches_joint_writer_and_sparse_casm():
    m = TemporalMaintenanceModel(CASMAlignedModel(), "joint_task")
    cases = make_pair(59, 32, 4)
    action, _, _ = m(*tensors(cases))
    F.cross_entropy(action, torch.tensor([x.expected for x in cases])).backward()
    for p in (m.reader.memory.write.weight, m.reader.compute.selector[-1].weight):
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0


def test_frozen_writer_and_event_adapter_have_no_gradient():
    m = TemporalMaintenanceModel(CASMAlignedModel(), "frozen_writer")
    cases = make_pair(71, 32, 4)
    action, _, _ = m(*tensors(cases))
    F.cross_entropy(action, torch.tensor([x.expected for x in cases])).backward()
    assert m.reader.memory.write.weight.grad is None
    assert m.event_map.weight.grad is None
    assert m.query_map.weight.grad is not None


def test_common_future_readout_training_cannot_update_writer_or_casm():
    m = TemporalMaintenanceModel(CASMAlignedModel(), "joint_task")
    cases = make_pair(59, 32, 4)
    _, future, _ = m(*tensors(cases))
    target = torch.tensor([[x.expected, x.service_expected] for x in cases])
    F.cross_entropy(future.reshape(-1, 2), target.reshape(-1)).backward()
    assert m.future_head.weight.grad is not None
    assert m.reader.memory.write.weight.grad is None
    assert m.reader.compute.selector[-1].weight.grad is None


def test_predictive_auxiliary_reaches_actual_writer():
    m = TemporalMaintenanceModel(CASMAlignedModel(), "joint_predictive")
    cases = make_pair(43, 32, 4, role="pending_cancel")
    _, _, auxiliary = m(*tensors(cases))
    target = torch.tensor([[x.expected, x.service_expected] for x in cases])
    F.cross_entropy(auxiliary.reshape(-1, 2), target.reshape(-1)).backward()
    grad = m.reader.memory.write.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_reset_state_has_identical_predictions_for_opposite_pasts():
    m = TemporalMaintenanceModel(CASMAlignedModel(), "joint_task").eval()
    cases = make_pair(71, 32, 4, role="pending_cancel")
    inputs = tensors(cases)
    with torch.no_grad():
        state = m.encode(*inputs[:3])
        action, future, _ = m.decode(torch.zeros_like(state), inputs[3])
    assert torch.equal(inputs[3][0], inputs[3][1])
    torch.testing.assert_close(action[0], action[1], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(future[0], future[1], rtol=1e-6, atol=1e-6)
    assert torch.equal(action.argmax(-1)[0], action.argmax(-1)[1])
    assert torch.equal(future.argmax(-1)[0], future.argmax(-1)[1])
