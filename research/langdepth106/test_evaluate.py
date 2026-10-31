"""Constructed no-fit fixtures for frozen evaluation accounting and integrity.

The hand-built tiny checkpoint is a corruption/replay-format fixture, not a fit
or a capability result. It cannot satisfy the six-fit frozen-cohort audit.
"""
from dataclasses import asdict
import math

import pytest
import torch

from research.langcache104.model import Config104, DeepCacheTAC104
from research.langdepth106 import evaluate
from research.langdepth106.train import OffsetSampler, checkpoint_payload, parameter_digest


def tiny_model(seed=10601):
    torch.manual_seed(seed)
    return DeepCacheTAC104(Config104(d_model=8, n_layers=1, n_heads=1, key_dim=2,
        value_dim=2, write_rank=1, ffn_dim=8, window=4, chunk_size=2)).eval()


def test_metrics_weight_by_actual_targets_including_short_chunk():
    logits = torch.zeros(1, 5, 256)
    logits[:, :4, 0] = 5
    logits[:, 4, 7] = 9
    targets = torch.tensor([[0, 0, 0, 0, 1]])
    split = evaluate.ByteMetrics()
    split.update(logits[:, :4], targets[:, :4])
    split.update(logits[:, 4:], targets[:, 4:])
    complete = evaluate.ByteMetrics()
    complete.update(logits, targets)
    a, b = split.result(), complete.result()
    assert a["target_count"] == 5 and a["correct_count"] == 4
    assert a["nll_sum_nats"] == pytest.approx(b["nll_sum_nats"], abs=1e-12)
    assert a["next_byte_nll_nats"] == pytest.approx(b["next_byte_nll_nats"], abs=1e-12)
    assert a["bits_per_byte"] == pytest.approx(a["next_byte_nll_nats"]/math.log(2))


def test_window_tail_and_source_target_alignment():
    parts = list(evaluate.windows(bytes(range(11)), 4))
    assert [start for start, _, _ in parts] == [0, 4, 8]
    assert [inputs.shape[1] for _, inputs, _ in parts] == [4, 4, 2]
    assert torch.cat([x for _, x, _ in parts], 1).tolist() == [list(range(10))]
    assert torch.cat([y for _, _, y in parts], 1).tolist() == [list(range(1, 11))]


def test_partition_modes_count_every_target_and_carry_without_refeeding():
    model = tiny_model()
    calls = []
    original = model.forward
    def recording(tokens, **kwargs):
        state = kwargs.get("state")
        calls.append((tokens.clone(), None if state is None else state.clock))
        return original(tokens, **kwargs)
    model.forward = recording
    data = bytes(range(11))
    streaming = evaluate.partition_metrics(model, data, sequence_length=4, streaming=True)
    assert streaming["target_count"] == 10 and streaming["final_carried_clock"] == 10
    assert [clock for _, clock in calls] == [None, 4, 8]
    assert torch.cat([x for x, _ in calls], 1).tolist() == [list(range(10))]
    calls.clear()
    reset = evaluate.partition_metrics(model, data, sequence_length=4, streaming=False)
    assert reset["target_count"] == 10 and reset["final_carried_clock"] is None
    assert [clock for _, clock in calls] == [None, None, None]


def test_generation_prefills_once_consumes_sample_once_and_preserves_global_rng():
    model = tiny_model()
    calls = []
    original = model.forward
    def recording(tokens, **kwargs):
        calls.append(tokens.clone())
        return original(tokens, **kwargs)
    model.forward = recording
    before = torch.get_rng_state().clone()
    prompt = b"ROMEO:\n"
    sample, metadata = evaluate.generate_bytes(model, prompt=prompt, length=5,
        temperature=.8, generator_seed=10699)
    assert torch.equal(before, torch.get_rng_state())
    assert sample.startswith(prompt) and len(sample) == len(prompt)+5
    assert calls[0].tolist() == [list(prompt)]
    assert [int(value[0, 0]) for value in calls[1:6]] == list(sample[len(prompt):])
    # Two extra calls check identical serialization replay; neither samples bytes.
    assert len(calls) == 8
    assert metadata["final_clock"] == len(sample)
    assert metadata["serialization_replay"]["status"] == "PASS_EXACT_CPU_REPLAY"
    assert metadata["vocabulary"] == "all 256 byte IDs"


def tiny_checkpoint(path):
    seed = 10601
    model = tiny_model(seed)
    initial = parameter_digest(model)
    # A deliberately constructed changed-weight/optimizer-format fixture, not an
    # optimizer update. Full frozen-cohort validation still requires real receipts.
    with torch.no_grad():
        model.embedding.weight.add_(.001)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    for parameter in model.parameters():
        optimizer.state[parameter] = {"step": torch.tensor(1.),
            "exp_avg": torch.zeros_like(parameter), "exp_avg_sq": torch.zeros_like(parameter)}
    sampler = OffsetSampler(100, 4, seed)
    sampler.draw(2)
    metadata = {"arm": "four", "seed": seed, "config": asdict(model.config),
        "parameter_count": model.counts()["trainable"], "protocol_sha256": "1"*64,
        "source_sha256": "2"*64, "train_sha256": "3"*64,
        "scientific_status": evaluate.STATUS, "initialized_parameters_sha256": initial,
        "optimizer_metadata": {}, "scheduler_metadata": {}, "status": "FIT_COMPLETED_EXPLORATORY", "smoke": False}
    payload = checkpoint_payload(model, optimizer, sampler, step=1, metadata=metadata)
    torch.save(payload, path)
    expected = dict(config=asdict(model.config), parameter_count=model.counts()["trainable"],
        expected_step=1, protocol_sha256="1"*64, source_sha256="2"*64,
        train_sha256="3"*64, arm="four", seed=seed, expected_sample_draws=2)
    return payload, expected


def test_checkpoint_audit_rejects_changed_weights_missing_rng_and_bad_optimizer(tmp_path):
    path = tmp_path/"fixture.pt"
    payload, expected = tiny_checkpoint(path)
    _, evidence = evaluate.audit_checkpoint(path, **expected)
    assert evidence["actual_steps"] == 1
    payload["trained_parameters_sha256"] = "0"*64
    torch.save(payload, path)
    with pytest.raises(ValueError, match="weights"):
        evaluate.audit_checkpoint(path, **expected)
    payload, expected = tiny_checkpoint(path)
    payload.pop("rng_torch")
    torch.save(payload, path)
    with pytest.raises(ValueError, match="incomplete checkpoint"):
        evaluate.audit_checkpoint(path, **expected)
    payload, expected = tiny_checkpoint(path)
    first = next(iter(payload["optimizer"]["state"].values()))
    first["step"] = torch.tensor(0.)
    torch.save(payload, path)
    with pytest.raises(ValueError, match="optimizer update counter"):
        evaluate.audit_checkpoint(path, **expected)


def test_absent_six_final_fits_blocks_before_heldout_open(tmp_path, monkeypatch):
    protocol = {"seeds": [10601, 10602, 10603], "fit": {"cpu_threads": 1,
        "deterministic_algorithms": True}}
    monkeypatch.setattr(evaluate, "load_protocol", lambda path: (protocol, "a"*64))
    def forbidden(*args, **kwargs):
        raise AssertionError("held-out data was opened before six final fits passed")
    monkeypatch.setattr(evaluate, "verify_evaluation_partition", forbidden)
    with pytest.raises(ValueError, match="all six"):
        evaluate.run_evaluation(protocol_path=tmp_path/"protocol.json", source_dir=tmp_path,
            run_dir=tmp_path, output_dir=tmp_path/"output", authorize_final_test=True)


def test_actual_tiny_resource_fixture_accounts_physical_state():
    result = evaluate.timing_profile(tiny_model(), prefill_length=5, decode_steps=2,
                                     warmups=0, repetitions=1)
    assert result["persistent_logical_tensor_bytes"] == result["persistent_physical_storage_bytes"]
    assert result["gpu_measured"] is False and result["full_flops_measured"] is False
