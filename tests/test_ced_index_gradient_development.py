"""DEV-009 regression controls for the excluded-gold top-k gradient problem."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from tac_osm.development.ced_hardened_router import QueryOnlyCED


def test_first_stage_gold_gradient_even_if_excluded_from_sparse_topk():
    torch.manual_seed(13)
    model=QueryOnlyCED(mode="sparse",topk=2)
    reference=torch.tensor([1.,0.,1.,0.,1.,0.,1.,0.])
    mask=torch.tensor([1.,1.,1.,0.,0.,0.,0.,0.])
    x=torch.randint(0,2,(8,8),dtype=torch.int64).float()
    logits,_=model(reference,mask,x)
    retained=(logits>-1e5).nonzero(as_tuple=True)[0]
    excluded=next(i for i in range(8) if i not in retained.tolist())
    assert logits[excluded].item()==-1e6
    scores=model.last_first_index_scores
    assert scores is not None and scores.shape==(8,) and scores.requires_grad
    ce=F.cross_entropy(scores[None],torch.tensor([excluded]))
    assert torch.isfinite(ce)
    ce.backward()
    assert model.decoder[0].query.weight.grad is not None
    assert model.decoder[0].query.weight.grad.abs().sum()>0


def test_shared_and_sparse_index_scores_start_identical():
    torch.manual_seed(43)
    shared=QueryOnlyCED(mode="shared",topk=4)
    torch.manual_seed(43)
    sparse=QueryOnlyCED(mode="sparse",topk=4)
    ref=torch.rand(8);mask=torch.tensor([1.,1.,0.,0.,0.,0.,0.,0.])
    x=torch.randint(0,2,(8,8)).float()
    _,a=shared(ref,mask,x)
    _,b=sparse(ref,mask,x)
    torch.testing.assert_close(shared.last_first_index_scores,
                               sparse.last_first_index_scores,atol=0,rtol=0)
    assert a.first_global_score_pairs==b.first_global_score_pairs
