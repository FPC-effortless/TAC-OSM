"""No-fit software checks; never issue a real/fixture training attestation."""
import copy
import json
from pathlib import Path

import pytest
import torch

from research.tac_ocr001.model import GroundedOCRDecoder, OCRConfig
from research.tac_ocr001.tokenizer import BOS, EOS, PAD
from . import core, train
from .stage import notebook_driver


def rows(n=141):
    return [{"text": "x" * (1 + i % 38), "sample_id": f"s-{i}"} for i in range(n)]


def test_every_region_once_and_replayable_bucket_schedule():
    source = rows()
    batches = core.epoch_batches(source, 1101, 0, 16, pool_size=64)
    assert sorted(i for batch in batches for i in batch) == list(range(len(source)))
    assert max(map(len, batches)) == 16
    assert core.epoch_batches(source, 1101, 0, 16, 64) == batches
    assert core.epoch_batches(source, 1102, 0, 16, 64) != batches
    assert core.epoch_batches(source, 1101, 1, 16, 64) != batches


def test_teacher_forcing_includes_last_character_and_eos():
    inputs, labels = core.targets([{"text": "₦a"}, {"text": "x"}], "cpu")
    encoded = list("₦a".encode("utf-8"))
    assert inputs[0].tolist() == [BOS, *encoded]
    assert labels[0].tolist() == [*encoded, EOS]
    assert inputs[1].tolist() == [BOS, ord("x"), PAD, PAD, PAD]
    assert labels[1].tolist() == [ord("x"), EOS, -100, -100, -100]


def test_real_unsigned_registration_blocks_before_optimizer_or_cuda(tmp_path, monkeypatch):
    monkeypatch.delenv("TAC_OSM_REWARD_VERIFIER_PUBLIC_KEY_HEX", raising=False)
    monkeypatch.delenv("TAC_OSM_PREFIT_ATTESTATION_JSON", raising=False)
    def forbidden(*a, **kw):
        pytest.fail("unsigned fit reached optimizer/GPU/feature code")
    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    monkeypatch.setattr(train, "verify_two_t4s", forbidden)
    monkeypatch.setattr(train, "feature_worker", forbidden)
    with pytest.raises(ValueError, match="independent"):
        train.main(["--data", str(tmp_path / "unopened"), "--output", str(tmp_path / "out")])
    receipt = json.loads((tmp_path / "out/prefit-blocked.json").read_text())
    assert receipt["optimizer_updates"] == 0
    assert receipt["test_opened"] is False


def test_inference_uses_no_gold_and_owner_renaming_cannot_change_logits():
    torch.manual_seed(88)
    torch.set_num_threads(1)
    model = GroundedOCRDecoder(OCRConfig(family="tac", d_model=16, n_layers=1,
        n_heads=2, ffn_dim=24, feature_dim=8, reference_budget=3,
        window=4, chunk_size=2, key_dim=4, value_dim=4)).eval()
    features = torch.randn(2, 3, 8)
    boxes = torch.zeros(2, 3, 4)
    baseline = core.predict(model, features, boxes, ("first", "second"), 4)
    renamed = core.predict(model, features, boxes, ("arbitrary-a", "arbitrary-b"), 4)
    assert baseline == renamed
    repeated = core.predict(model, features, boxes, ("first", "second"), 4)
    assert baseline == repeated
    with pytest.raises(TypeError):
        core.predict(model, features, boxes, ("first", "second"), gold=["denied", "target"])


def test_metrics_count_unicode_characters_and_words_without_normalization():
    report = core.transcription_metrics(["₦1", "De tin"], ["N1", "de tin"])
    assert report["gold_characters"] == 8
    assert report["character_error_rate"] == pytest.approx(2 / 8)
    assert report["word_error_rate"] == pytest.approx(2 / 3)
    assert report["exact_region_accuracy"] == 0


def test_splits_fail_closed_after_byte_mutation(tmp_path):
    path = tmp_path / "train_filtered.jsonl"
    path.write_text(json.dumps({"text": "gold", "sample_id": "1", "split": "train"}) + "\n")
    expected = core.sha256(path)
    assert len(core.load_rows(tmp_path, "train", expected)) == 1
    path.write_text(path.read_text().replace("gold", "changed"))
    with pytest.raises(ValueError, match="differs"):
        core.load_rows(tmp_path, "train", expected)


def test_checkpoint_replay_restores_weights_optimizer_and_rejects_changed_identity(tmp_path):
    torch.manual_seed(3)
    model = torch.nn.Linear(2, 3)
    optimizer = torch.optim.AdamW(model.parameters())
    original = copy.deepcopy(model.state_dict())
    path = tmp_path / "last.pt"
    identity = {"source_sha256": "a" * 64, "seed": 1101}
    core.atomic_checkpoint(path, model, optimizer, cursor={"epoch": 1, "batch": 0},
                           identity=identity, counters={"steps": 0})
    with torch.no_grad():
        model.weight.add_(5)
    cursor, counters = core.restore_checkpoint(path, model, optimizer, identity)
    assert cursor == {"epoch": 1, "batch": 0}
    assert counters == {"steps": 0}
    for key, value in model.state_dict().items():
        assert torch.equal(value, original[key])
    with pytest.raises(ValueError, match="identity"):
        core.restore_checkpoint(path, model, optimizer, {**identity, "seed": 1102})


def test_no_gpu_substitution(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i: "Tesla T4")
    with pytest.raises(RuntimeError, match="exactly two"):
        train.verify_two_t4s()
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i: "NVIDIA L4")
    with pytest.raises(RuntimeError, match="exactly two"):
        train.verify_two_t4s()


def test_learning_rate_warmup_and_floor():
    assert core.learning_rate(0, 100, 10, .001) == pytest.approx(.0001)
    assert core.learning_rate(9, 100, 10, .001) == pytest.approx(.001)
    assert core.learning_rate(99, 100, 10, .001) == pytest.approx(.0001)


def test_notebook_checks_exact_hash_and_preflight_before_images_or_training():
    driver = notebook_driver("1" * 64)
    compile(driver, "notebook", "exec")
    assert "1" * 64 in driver
    assert driver.index("require_prefit; require_prefit()") < driver.index("extract(inputs / 'data.bin'")
    assert driver.index("require_prefit; require_prefit()") < driver.index("'research.tac_ocr003.train'")
    assert "HF_HUB_OFFLINE='1'" in driver
    assert "--no-index" in driver
    assert "TAC_OSM_POSTFIT_ATTESTATION_JSON" in driver
    assert "research.tac_ocr003.assess" in driver
