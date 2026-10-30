"""TAOTUNE004 prospectively registered treatment and no-DEV gates."""
from pathlib import Path
import sys
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/"src",ROOT/"research/mathematical-engineering/langmem086"):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
from scripts import tao004_tune as tune
from tac_osm.optimizer_tao import TAO


def test_tao004_manifest_and_trial_parity():
    m=tune.registered()
    assert m["execution_level"]=="internal_research"
    assert m["no_architecture_superiority_claim"] is True
    assert len(m["models"])==15
    assert {x["name"] for x in m["models"]}=={f"{a}_lr{i}" for a in tune.ARMS for i in range(3)}
    assert all(x["trainable_parameters"]==100521 for x in m["models"])
    assert m["budget"]["targets_per_trial"]==32768
    assert set(m["tuning_cohort"]).isdisjoint(set(m["reporting_cohort"]))


def test_tao004_pinned_real_publisher_fit_order():
    docs,rows,sched=tune.fit_original_batches()
    assert len(docs)==14 and len(rows)==32 and sched["targets"]==32768
    assert rows[0][0][0].startswith("Doc/")
    assert len(sched["sample_order_sha256"])==64


def test_tao004_only_ffn_group_lr_changed():
    m=tune.registered()
    for mode in tune.ARMS:
        for lr in m["learning_rate_grid"][mode]:
            torch.manual_seed(1701)
            model=tune.CompactDualClippedTAC086()
            before=tune.weight_hash(model)
            o=tune.optimizer_for(mode,model,lr)
            assert tune.weight_hash(model)==before
            if mode=="adamw":
                assert isinstance(o,torch.optim.AdamW)
                assert len(o.param_groups)==1 and o.param_groups[0]["lr"]==lr
            else:
                assert isinstance(o,TAO)
                orth=next(g for g in o.param_groups if g["kind"]=="orthogonal")
                fallback=next(g for g in o.param_groups if g["kind"]=="adamw")
                assert orth["lr"]==lr and fallback["lr"]==.001
                assert len(orth["params"])==1 and orth["params"][0] is model.ffn_up.weight


def test_tao004_tune_fails_before_dev_if_originals_absent(tmp_path,monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError("DEV must not be fetched before every original checkpoint")
    monkeypatch.setattr(tune,"read_sha_source",forbidden)
    with pytest.raises(FileNotFoundError):
        tune.tune(tmp_path)
