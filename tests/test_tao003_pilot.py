"""TAOPILOT003: no-fit research preflight and checkpoint-denial tests."""
import sys
from pathlib import Path
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT/"src", ROOT/"research/mathematical-engineering/langmem086"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from scripts import tao003_pilot as pilot
from tac_osm.optimizer_tao import TAO


def test_registered_original_source_and_all_five_arms():
    manifest, original = pilot.frozen()
    assert manifest["study_id"] == "TAOPILOT003"
    assert manifest["budget"]["steps"] == 32
    assert manifest["no_architecture_superiority_claim"] is True
    assert manifest["execution_level"] == "internal_research"
    assert sorted(m["name"] for m in manifest["models"]) == sorted(pilot.ARMS)
    assert all(m["trainable_parameters"] == 100521 for m in manifest["models"])
    assert original["targets"] == 163840


def test_real_original_fit_schedule_prefix_from_publisher_bytes():
    docs, sha = pilot.original.fit_sources086()
    assert sha == pilot.SOURCE_SHA
    _, original = pilot.frozen()
    rows, schedule = pilot.original_schedule(docs, original)
    assert len(rows) == 32 and schedule["targets"] == 32768
    assert schedule["sample_order_sha256"] != schedule["sample_bytes_sha256"]


def test_all_optimizers_route_only_ffn_up_and_preserve_initial_weights():
    initials = set()
    for arm in pilot.ARMS:
        torch.manual_seed(1701)
        model = pilot.CompactDualClippedTAC086()
        before = pilot.model_hash(model)
        opt = pilot.optimizer_for(arm, model)
        initials.add(before)
        assert pilot.model_hash(model) == before
        if arm != "adamw":
            assert isinstance(opt, TAO)
            route = next(g for g in opt.param_groups if g["kind"] == "orthogonal")
            assert len(route["params"]) == 1 and route["params"][0] is model.ffn_up.weight
            assert sum(p.numel() for p in route["params"]) == 15040
        else:
            assert isinstance(opt, torch.optim.AdamW)
    assert len(initials) == 1


def test_no_dev_scoring_without_all_five_git_backed_checkpoints(tmp_path, monkeypatch):
    def forbidden(*a, **kw):
        raise AssertionError("should not access DEV before frozen checkpoint matrix")
    monkeypatch.setattr(pilot, "read_sha_source", forbidden)
    with pytest.raises(FileNotFoundError):
        pilot.evaluate(tmp_path)


def test_numeric_cross_host_replay_bounded_and_top1_stable():
    original = torch.tensor([[[0.1, 0.2, 0.3], [0.001, 0.00101, 0.01]]])
    exact = pilot.independent_numeric_replay(original, original)
    assert exact["bitwise"] and exact["numerically_equivalent"]
    tiny = original.clone()
    tiny[0, 0, 0] += 4.76837158203125e-7
    measured = pilot.independent_numeric_replay(tiny, original)
    assert not measured["bitwise"] and measured["numerically_equivalent"]
    broken = original.clone()
    broken[0, 0, 0] += 0.01
    with pytest.raises(ValueError):
        pilot.independent_numeric_replay(broken, original)
    # A tiny perturbation inside the numerical bound must still fail when
    # it changes the predicted class (not permit scientific rank drift).
    close = torch.tensor([[[0.0, 1.e-5]]])
    reversed_rank = torch.tensor([[[2.e-5, 1.e-5]]])
    with pytest.raises(ValueError):
        pilot.independent_numeric_replay(reversed_rank, close)
