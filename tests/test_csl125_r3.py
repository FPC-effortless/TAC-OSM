"""R3 no-fit tests: math/causality/parameter budget, NOT learned correction."""

import pytest
import torch

from tac_osm.csl125_r2 import CSL125R2
from tac_osm.csl125_r3 import CSL125R3, CSL125R3Config


def small_cfg():
    return CSL125R3Config(vocab_size=64, width=32, ff_width=64, layers=4,
        role_count=4, state_width=8, rungs=(2, 4), attention_layers=(2, 4),
        query_heads=4, kv_heads=2, attention_window=16, correction_rank=4)


def test_meta_all_trainable_weights_included_in_125m_limit():
    with torch.device('meta'):
        model = CSL125R3(device='meta')
    assert model.parameter_count() == 123_873_263
    assert model.parameter_count() <= 125_000_000
    assert sum(p.numel() for n,p in model.named_parameters()
               if n.startswith(('error_head.', 'revise_up.', 'revise_down.'))) == 50_689
    assert all(p.requires_grad for p in model.parameters())


def test_original_r2_outputs_and_memory_unchanged_on_identical_weights():
    torch.manual_seed(7)
    cfg = small_cfg()
    base = CSL125R2(cfg).eval()
    revised = CSL125R3(cfg).eval()
    missing, unexpected = revised.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected
    assert set(missing) == {'error_head.weight', 'error_head.bias',
                            'revise_up.weight', 'revise_down.weight', 'revise_down.bias'}
    x = torch.randint(0, cfg.vocab_size, (1, 15))
    with torch.no_grad():
        a, sa = base(x, all_rungs=True, stream_ids=('trial',))
        b, sb = revised(x, all_rungs=True, stream_ids=('trial',), emit_repair_logits=True)
    for depth in cfg.rungs:
        for key in a[depth]:
            assert torch.equal(a[depth][key], b[depth][key]), (depth, key)
        assert torch.isfinite(b[depth]['repair_logits']).all()
        assert b[depth]['error_logit'].shape == (1, 15)
    for st1, st2 in zip(sa.layers, sb.layers):
        for key in ('fast', 'slow', 'fast_mass', 'slow_mass', 'conv_tail'):
            assert torch.equal(getattr(st1,key), getattr(st2,key))


def test_closed_correction_equals_base_even_with_enabled_repair():
    torch.manual_seed(2)
    m = CSL125R3(small_cfg()).eval()
    with torch.no_grad():
        m.revise_down.weight.zero_()
        m.revise_down.bias.zero_()
        x = torch.randint(0,64,(1,13))
        y,_ = m(x, depth=4, emit_repair_logits=True)
    assert torch.equal(y[4]['token_logits'],y[4]['repair_logits'])


def test_repair_bound_applies_to_each_hidden_feature():
    torch.manual_seed(4)
    model = CSL125R3(small_cfg())
    z = torch.randn(2,7,32) * 50
    risk = torch.randn(2,7,1) * 50
    revised = model._revised_features(z, risk)
    assert torch.isfinite(revised).all()
    assert (revised-z).abs().max() <= model.config.max_correction_gain + 1e-5


def test_no_future_poison_and_stream_chunks():
    torch.manual_seed(6)
    model = CSL125R3(small_cfg()).eval()
    a = torch.randint(0,64,(1,23))
    b = a.clone(); b[:,16:] = torch.randint(0,64,(1,7))
    with torch.no_grad():
        y,_ = model(a, depth=4, emit_repair_logits=True, stream_ids=('run',))
        poisoned,_ = model(b, depth=4, emit_repair_logits=True, stream_ids=('run',))
        for key in ('token_logits','repair_logits','error_logit'):
            assert torch.allclose(y[4][key][:,:16], poisoned[4][key][:,:16], atol=3e-5)
        state=None; chunks=[]; scores=[]
        for lo,hi in ((0,5),(5,14),(14,23)):
            o,state = model(a[:,lo:hi],depth=4,emit_repair_logits=True,
                            state=state,stream_ids=('run',))
            chunks.append(o[4]['repair_logits']); scores.append(o[4]['error_logit'])
        assert torch.allclose(torch.cat(chunks,1), y[4]['repair_logits'],atol=5e-5,rtol=5e-5)
        assert torch.allclose(torch.cat(scores,1), y[4]['error_logit'],atol=5e-5,rtol=5e-5)
        with pytest.raises(ValueError,match='stream'):
            model(a[:,:1],depth=4,state=state,stream_ids=('wrong',))
        with pytest.raises(ValueError,match='reset/replay'):
            model(a[:,:1],depth=2,state=state,stream_ids=('run',))


def test_training_loss_reaches_risk_reviser_and_base_without_target_leakage():
    torch.manual_seed(13)
    model = CSL125R3(small_cfg())
    tokens = torch.randint(0,64,(2,12))
    mask = torch.ones((2,11),dtype=torch.bool); mask[:,0]=False
    loss = model.nested_repair_loss(tokens, loss_mask=mask)
    assert torch.isfinite(loss)
    loss.backward()
    for name in ('error_head.weight','revise_up.weight','revise_down.weight',
                 'embedding.weight','blocks.0.write_gate.weight'):
        grad = dict(model.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all(), name
    with pytest.raises(TypeError):
        model(tokens, depth=2, target_labels=tokens)
    with pytest.raises(ValueError,match='loss_mask'):
        model.nested_repair_loss(tokens, loss_mask=torch.zeros_like(mask))
    with pytest.raises(ValueError,match='weights'):
        model.nested_repair_loss(tokens, weights={2: 1., 4: float('nan')})


def test_invalid_config_and_unregistered_depth_fail_closed():
    with pytest.raises(ValueError,match='correction_rank'):
        CSL125R3Config(correction_rank=0)
    with pytest.raises(ValueError,match='max_correction_gain'):
        CSL125R3Config(max_correction_gain=float('nan'))
    with pytest.raises(ValueError,match='initial_error_bias'):
        CSL125R3Config(initial_error_bias=100)
    m=CSL125R3(small_cfg())
    with pytest.raises(ValueError,match='depth'):
        m(torch.randint(0,64,(1,5)),depth=0)


def test_candidate_topk_matches_full_revision_on_selected_candidates():
    torch.manual_seed(17)
    model = CSL125R3(small_cfg()).eval()
    x = torch.randint(0,64,(1,9))
    with torch.no_grad():
        outs,_ = model(x, depth=4, emit_repair_logits=True,
                       candidate_repair_topk=5)
    y=outs[4]
    ids=y['repair_candidate_ids']
    scores=y['repair_candidate_scores']
    assert ids.shape == scores.shape == (1,9,5)
    assert torch.equal(y['token_logits'].topk(5,dim=-1).indices,ids)
    assert torch.allclose(scores,y['repair_logits'].gather(-1,ids),atol=2e-5,rtol=2e-5)
    assert torch.unique(ids).numel() <= 45


def test_candidate_topk_fails_closed_and_never_adds_non_candidates():
    torch.manual_seed(29)
    model = CSL125R3(small_cfg()).eval()
    x = torch.randint(0,64,(1,5))
    with pytest.raises(ValueError,match='candidate_repair_topk'):
        model(x,depth=2,candidate_repair_topk=0)
    with pytest.raises(ValueError,match='candidate_repair_topk'):
        model(x,depth=2,candidate_repair_topk=65)
    with pytest.raises(ValueError,match='requires base token logits'):
        model(x,depth=2,candidate_repair_topk=5,emit_token_logits=False)
    with torch.no_grad():
        y,_ = model(x,depth=2,candidate_repair_topk=5)
    assert 'repair_logits' not in y[2]
    assert set(y[2]['repair_candidate_ids'].flatten().tolist()).issubset(
        set(y[2]['token_logits'].topk(5,dim=-1).indices.flatten().tolist()))
