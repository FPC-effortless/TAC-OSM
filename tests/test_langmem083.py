"""LANGMEM083 engineering-only tests: no artificial success claim."""
from __future__ import annotations
from pathlib import Path
import hashlib,json,sys,torch
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"research/mathematical-engineering/langmem083"))
for name in ("langmem082","langmem081","langmem080","langmem079","langmem078",
             "langscale075","langscale070","context026","langmemory042","langbase024"):
    p=str(ROOT/"research/mathematical-engineering"/name)
    if p not in sys.path:sys.path.insert(0,p)
from frozen_probe import FrozenReadout083,wrong_writer_key,AGES,ARMS,INPUT,FEATURE_COUNT
from scripts.langmem083_fit_eval import read_unseen_publisher_docs

def test_exact_9120_useful_parameters_and_all_gradients():
    torch.manual_seed(83001)
    m=FrozenReadout083()
    assert sum(p.numel() for p in m.parameters())==9120
    X=torch.randn(32,20)
    y=torch.randint(0,256,(32,))
    loss=torch.nn.functional.cross_entropy(m(X),y)
    loss.backward()
    assert all(w.grad is not None and torch.isfinite(w.grad).all()
               and w.grad.abs().sum().item()>0 for w in m.parameters())

def test_wrong_past_source_key_uses_only_preceding_observed_context():
    rng=bytes((i%31 for i in range(130)))
    keys={a:torch.nn.functional.normalize(torch.arange(1,9,dtype=torch.float32)+a,dim=0)
          for a in range(4,128)}
    a=80;t=128
    selected=wrong_writer_key(keys,rng[:t+1],a,t,keys[a])
    assert selected.shape==(8,)
    assert torch.isfinite(selected).all()
    assert not torch.equal(selected,keys[a])
    assert all(4<=b<t for b in keys)
    # Future true target byte at t+1 cannot enter the selector.
    assert torch.equal(selected,wrong_writer_key(keys,rng[:t+1],a,t,keys[a]))

def test_original_new_publisher_doc_receipt_is_frozen_disjoint_and_typed():
    d=json.loads((ROOT/"research/mathematical-engineering/langmem083/NEW_DEV_SOURCE.json").read_text())
    old=json.loads((ROOT/"research/mathematical-engineering/language_source023/SOURCE_RECEIPT.json").read_text())
    known={x["path"] for x in old["files"]}
    assert len(d["documents"])==2
    assert all(x["path"] not in known and x["role"]=="UNSEEN_DEV_FROZEN_BEFORE_PROBE_FIT"
               and len(x["sha256"])==64 and len(x["git_blob"])==40
               and x["bytes"]>=640 for x in d["documents"])
    assert d["commit"]==old["commit"]

def test_eval_rejects_missing_or_changed_dev_without_fallback(tmp_path):
    with pytest.raises(ValueError):
        read_unseen_publisher_docs(tmp_path)
    (tmp_path/"regex.rst").write_bytes(b"unapproved replacement")
    (tmp_path/"argparse.rst").write_bytes(b"unapproved replacement")
    with pytest.raises(ValueError):
        read_unseen_publisher_docs(tmp_path)

def test_frozen_source_horizon_and_no_future_target_in_feature_api():
    assert AGES==(8,32,128,256) and ARMS==("correct","address_only","wrong_key")
    assert INPUT==20 and FEATURE_COUNT==9120
