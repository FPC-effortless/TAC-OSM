"""Bounded constructed optimizer/replay fixtures, not corpus capability scores."""
from dataclasses import asdict
import random

import pytest
import torch

from research.langcache104.model import Config104, DeepCacheTAC104
from research.langdepth106.train import (
    SCIENTIFIC_STATUS, OffsetSampler, checkpoint_payload, learning_rate,
    load_checkpoint, load_protocol, make_batch, make_optimizer, optimization_step,
    parameter_digest, restore_runtime, save_checkpoint, verify_readback,
)


def tiny_runtime(seed=10601):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    random.seed(seed)
    config = Config104(d_model=8, n_layers=2, n_heads=1, key_dim=2, value_dim=2,
                       write_rank=2, ffn_dim=12, window=4, chunk_size=2)
    model = DeepCacheTAC104(config).float()
    fit = {"steps": 8, "batch_size": 2, "sequence_length": 5,
           "optimizer": "AdamW", "lr": .0003, "min_lr": .00003, "warmup_steps": 2,
           "betas": [.9, .95], "eps": 1e-8, "weight_decay": .1, "clip_grad_norm": 1.}
    optimizer = make_optimizer(model, fit)
    sampler = OffsetSampler(64, fit["sequence_length"], seed)
    train = torch.arange(64, dtype=torch.long)
    return model, optimizer, sampler, train, fit


def metadata(model, fit, seed=10601):
    return {"arm": "four", "seed": seed, "config": asdict(model.config), "smoke": False,
            "parameter_count": model.counts()["trainable"],
            "protocol_sha256": "constructed-unit-protocol-not-real-registration",
            "source_sha256": "constructed-unit-bytes-not-real-corpus",
            "train_sha256": "constructed-unit-train-not-real-source",
            "scientific_status": SCIENTIFIC_STATUS,
            "initialized_parameters_sha256": parameter_digest(model),
            "optimizer_metadata": {key: fit[key] for key in
                ("optimizer", "lr", "betas", "eps", "weight_decay", "clip_grad_norm")},
            "scheduler_metadata": {key: fit[key] for key in
                ("steps", "lr", "min_lr", "warmup_steps")}}


def unit_updates(runtime, begin, end):
    model, optimizer, sampler, train, fit = runtime
    random_observations = []
    for step in range(begin, end+1):
        # Exercise both global RNGs independently of the dedicated data RNG.
        random_observations.append((torch.rand(3), random.random()))
        offsets = sampler.draw(fit["batch_size"])
        x, y = make_batch(train, offsets, fit["sequence_length"])
        result = optimization_step(model, optimizer, x, y, step=step, fit=fit,
                                   owners=("constructed-A", "constructed-B"))
        assert result["step"] == step
        assert result["loss"] > 0
    return random_observations


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for name in left:
            assert_nested_equal(left[name], right[name])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_equal(a, b)
    else:
        assert left == right


def test_offsets_are_legal_paired_and_independent_of_initialization_rng():
    a = OffsetSampler(100, 7, 10601)
    b = OffsetSampler(100, 7, 10601)
    first = a.draw(40)
    torch.manual_seed(123)
    torch.randn(257)
    random.seed(999)
    assert b.draw(40) == first
    assert all(0 <= value < 93 for value in first)
    saved = a.state_dict()
    suffix = a.draw(40)
    restored = OffsetSampler(100, 7, 10601)
    restored.load_state_dict(saved)
    assert restored.draw(40) == suffix
    assert restored.fingerprint() == a.fingerprint()


def test_source_and_targets_stay_within_one_training_partition():
    train = torch.arange(64, dtype=torch.long)
    x, y = make_batch(train, [0, 58], 5)
    assert torch.equal(x[:, 1:], y[:, :-1])
    assert y[-1, -1] == 63
    for offsets in ([-1], [59]):
        with pytest.raises(ValueError):
            make_batch(train, offsets, 5)


def test_learning_rate_is_update_indexed_and_reaches_declared_endpoints():
    fit = {"steps": 4096, "warmup_steps": 128, "lr": .0003, "min_lr": .00003}
    assert learning_rate(1, fit) == fit["lr"]/128
    assert learning_rate(128, fit) == fit["lr"]
    assert learning_rate(129, fit) < fit["lr"]
    assert learning_rate(4096, fit) == fit["min_lr"]
    with pytest.raises(ValueError): learning_rate(0, fit)
    with pytest.raises(ValueError): learning_rate(4097, fit)


def test_actual_optimizer_resume_matches_uninterrupted_weights_state_and_rng(tmp_path):
    uninterrupted = tiny_runtime()
    original = parameter_digest(uninterrupted[0])
    all_random = unit_updates(uninterrupted, 1, 8)
    reference_rng, reference_python = torch.get_rng_state(), random.getstate()
    reference_model = {k: v.detach().clone() for k, v in uninterrupted[0].state_dict().items()}
    reference_optimizer = uninterrupted[1].state_dict()
    interrupted = tiny_runtime()
    common = metadata(interrupted[0], interrupted[4])
    first_random = unit_updates(interrupted, 1, 4)
    payload = checkpoint_payload(interrupted[0], interrupted[1], interrupted[2],
                                 step=4, metadata=common)
    path = tmp_path/"checkpoint-4.pt"
    save_checkpoint(path, payload)
    # A fresh process can perform arbitrary initialization before restoration.
    torch.rand(40); random.random()
    continued = tiny_runtime()
    loaded = load_checkpoint(path, protocol_sha256=common["protocol_sha256"],
        source_sha256=common["source_sha256"], arm="four", seed=10601,
        config=asdict(continued[0].config))
    restore_runtime(loaded, continued[0], continued[1], continued[2])
    latter_random = unit_updates(continued, 5, 8)
    assert_nested_equal(all_random, first_random+latter_random)
    assert_nested_equal(reference_model, continued[0].state_dict())
    assert_nested_equal(reference_optimizer, continued[1].state_dict())
    assert torch.equal(reference_rng, torch.get_rng_state())
    assert reference_python == random.getstate()
    assert uninterrupted[2].offsets == continued[2].offsets
    assert uninterrupted[2].fingerprint() == continued[2].fingerprint()
    assert continued[2].draws == 16
    assert parameter_digest(continued[0]) != original


def test_checkpoint_readback_does_not_change_live_parameters_or_rng(tmp_path):
    runtime = tiny_runtime()
    model, optimizer, sampler, train, fit = runtime
    common = metadata(model, fit)
    unit_updates(runtime, 1, 2)
    path = tmp_path/"checkpoint-2.pt"
    save_checkpoint(path, checkpoint_payload(model, optimizer, sampler, step=2, metadata=common))
    torch_before, python_before, parameters_before = torch.get_rng_state(), random.getstate(), parameter_digest(model)
    result = verify_readback(path, model, fit, train[:5][None])
    assert result["fixed_TRAIN_probe_logits_exact"]
    assert result["optimizer_reloaded"] and result["sampler_rng_reloaded"]
    assert torch.equal(torch_before, torch.get_rng_state())
    assert python_before == random.getstate()
    assert parameters_before == parameter_digest(model)


def test_checkpoint_identity_corruption_and_unsafe_resume_are_rejected(tmp_path):
    model, optimizer, sampler, _train, fit = tiny_runtime()
    common = metadata(model, fit)
    payload = checkpoint_payload(model, optimizer, sampler, step=0, metadata=common)
    path = tmp_path/"checkpoint-0.pt"
    save_checkpoint(path, payload)
    with pytest.raises(ValueError): load_checkpoint(path, arm="eight")
    with pytest.raises(ValueError): load_checkpoint(path, seed=10602)
    with pytest.raises(ValueError): load_checkpoint(path, protocol_sha256="wrong-protocol")
    with pytest.raises(ValueError): load_checkpoint(path, source_sha256="wrong-source")
    payload["resume_safe"] = False
    with pytest.raises(ValueError): restore_runtime(payload, model, optimizer, sampler)
    payload["resume_safe"] = True
    payload["trained_parameters_sha256"] = "wrong-tensor-hash"
    with pytest.raises(ValueError): restore_runtime(payload, model, optimizer, sampler)


def test_frozen_protocol_counts_match_live_unmodified_graph():
    protocol, checksum = load_protocol()
    assert len(checksum) == 64
    for arm, row in protocol["models"].items():
        model = DeepCacheTAC104(Config104(**row["config"]))
        assert model.counts()["trainable"] == row["parameters"], arm
