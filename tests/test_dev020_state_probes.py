"""DEV-020: synthetic integrity checks; no observed outcome is a capability claim."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("dev020", ROOT / "scripts/run_dev020_state_probes.py")
dev020 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dev020)
from tac_osm.development.memory_baselines import MemoryBaseline  # noqa: E402


def test_pair_metrics_reject_unbalanced_and_count_disagreements():
    y = torch.tensor([0., 1., 1., 0.])
    scores = torch.tensor([.1, .9, .8, .8])
    result = dev020.pair_metrics(y, scores)
    assert result["accuracy"] == .75
    assert result["paired_class_disagreement"] == .5
    with pytest.raises(AssertionError):
        dev020.pair_metrics(torch.tensor([0., 0.]), torch.tensor([.1, .9]))


def test_archived_accuracy_uses_exact_count_ratio():
    # DEV-018 short seed43 schema41: 91/96 is not exact in float32.
    y = torch.tensor([0., 1.] * 48)
    scores = y.clone()
    scores[:5] = 1 - scores[:5]
    result = dev020.pair_metrics(y, scores)
    assert result["accuracy"] == 91 / 96
    assert result["paired_class_disagreement"] == 47 / 48


def test_disjoint_synthetic_domains_and_matched_recent_observation():
    train = dev020.episodes(43, "dev020-probe-train", 1)
    frozen = dev020.episodes(43, "dev018-eval", 1)
    assert len(train) == len(frozen) == 24
    assert not {e.fingerprint for e in train} & {e.fingerprint for e in frozen}
    assert len({e.fingerprint for e in train}) == len(train)
    assert float(dev020.labels(train).mean()) == .5
    assert all(train[i].observations[-1] == train[i + 1].observations[-1]
               for i in range(0, len(train), 2))


def test_probe_entity_conditioning_and_frozen_read_boundary():
    queries = torch.zeros(2, 8, 3, 14)
    state = torch.zeros(2, 41)
    state[:, :40] = torch.arange(40).float().reshape(1, 40)
    targets = torch.tensor([1, 7])
    schema = MemoryBaseline("schema41", 2).eval()
    projected = dev020.target_state_features(schema, state, queries, targets)
    assert projected.shape == (2, 20)
    torch.testing.assert_close(projected[0, :5], state[0, 5:10])
    torch.testing.assert_close(projected[1, :5], state[1, 35:40])
    memoryless = MemoryBaseline("memoryless", 2).eval()
    assert dev020.target_state_features(memoryless, torch.zeros(2, 12),
                                       queries, targets).shape == (2, 26)
