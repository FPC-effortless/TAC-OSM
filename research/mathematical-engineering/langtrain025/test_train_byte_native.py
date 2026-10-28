"""LANGTRAIN025 constructed software tests; authentic source metric is separate."""
import importlib.util
import math
import numpy as np
import torch
from torch.nn import functional as F
from byte_waveced_500k import ByteWaveCED500K
from train_byte_native import PARAMS,SEEDS,UPDATES,source_sample,windows,numerical_metrics

def test_registered_learned_parameter_count_and_byte_output():
    torch.set_num_threads(2)
    model=ByteWaveCED500K()
    assert model.parameter_count()==499978==PARAMS
    a=torch.tensor([[0,1,255,128,3,3,5,0]],dtype=torch.long)
    out=model(a)
    assert out.shape==(1,8,256)
    law=out.softmax(-1)
    assert torch.allclose(law.sum(-1),torch.ones(1,8),atol=1e-6)
    assert torch.all(law>0)

def test_graph_q_k_are_predictively_active():
    torch.manual_seed(271)
    model=ByteWaveCED500K()
    src=torch.randint(0,256,(2,8))
    y=torch.randint(0,256,(2,8))
    logits=model(src)
    F.cross_entropy(logits.reshape(-1,256),y.reshape(-1)).backward()
    assert model.concept_q.weight.grad is not None
    assert model.concept_k.weight.grad is not None
    assert torch.any(model.concept_q.weight.grad !=0)
    assert torch.any(model.concept_k.weight.grad !=0)

def test_future_label_poison_cannot_change_causal_prefix():
    torch.manual_seed(99)
    model=ByteWaveCED500K().eval()
    good=torch.randint(0,256,(2,12))
    poison=good.clone()
    poison[:,7:]=torch.randint(0,256,poison[:,7:].shape)
    with torch.no_grad():
        a=model(good)[:,:7]
        b=model(poison)[:,:7]
    assert torch.allclose(a,b,rtol=0,atol=2e-5)

def test_no_owner_crossing_when_sampling():
    docs={"a":b"x"*260,"b":b"z"*290}
    rng=np.random.default_rng(1701)
    import hashlib
    digest=hashlib.sha256()
    sample=source_sample(rng,docs,tuple(sorted(docs)),digest)
    assert sample.shape==(8,129)
    for row in sample.tolist():
        assert len(set(row))==1 and row[0] in (ord("x"),ord("z"))
    assert len(digest.hexdigest())==64

def test_fixed_eight_window_source_rule():
    docs={str(i):bytes([i])*400 for i in range(4)}
    w=windows(docs)
    assert len(w)==4 and all(len(block)==8 for block in w.values())
    assert all(len(win)==129 for blocks in w.values() for _,win in blocks)

def test_raw_256_byte_accuracy_metrics_are_finite():
    logits=torch.zeros(1,3,256)
    labels=torch.tensor([[0,127,255]])
    r=numerical_metrics(logits,labels)
    assert r["n"]==3 and r["rare_n"]==1
    assert math.isclose(r["nll_sum"]/3,math.log(256),rel_tol=1e-6)
    assert 0<=r["top1_sum"]<=3
