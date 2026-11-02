"""No-fit state/cache lifetime regressions; constructed inputs, no data/checkpoints."""
import copy
import pickle
from dataclasses import replace

import pytest
import torch
from torch import nn

from .model import DecoderState
from .steady import SteadyTACSession
from .test_model import tiny


@pytest.mark.parametrize('family', ['gru', 'transformer', 'tac'])
def test_state_owner_detach_and_legacy_rejection(family):
    model, features, boxes, refs = tiny(family)
    other, _, _, _ = tiny(family)
    tokens = torch.tensor([[256, 1, 2, 3], [256, 4, 5, 6]])
    with torch.no_grad():
        logits, state = model(tokens, refs, owners=('a', 'b'))
        detached = state.detach()
        assert detached.signature == model.signature == other.signature
        assert detached.model_tag == model.model_tag != other.model_tag
        assert all(not t.requires_grad for t in detached.tensors().values())
        fresh = other.prepare_visual(features, boxes, owners=('a', 'b'), page_ids=('p1', 'p1'))
        other_logits, _ = other(tokens, fresh, owners=('a', 'b'))
        torch.testing.assert_close(logits, other_logits, rtol=0, atol=0)
        for foreign in (state, detached, DecoderState(state.owners, state.signature, state.clock, state.layers)):
            with pytest.raises(ValueError, match='another'):
                other(tokens, fresh, owners=('a', 'b'), state=foreign)


@pytest.mark.parametrize('family', ['gru', 'transformer', 'tac'])
@pytest.mark.parametrize('load_kind', ['direct', 'parent', 'failed'])
def test_load_invalidates_before_partial_or_parent_load(family, load_kind):
    model, features, boxes, refs = tiny(family)
    tokens = torch.tensor([[256], [256]])
    with torch.no_grad():
        _, state = model(tokens, refs, owners=('a', 'b'))
        old_tag = model.model_tag
        # Registered pre-hook observes generation before any tensor copy.
        observed = []
        handle = model.register_load_state_dict_pre_hook(lambda *args: observed.append(model.model_tag))
        weights = {k: v.clone() for k, v in model.state_dict().items()}
        weights['visual_projection.weight'].add_(.1)
        if load_kind == 'parent':
            nn.ModuleDict({'decoder': model}).load_state_dict({'decoder.'+k: v for k, v in weights.items()})
        elif load_kind == 'failed':
            weights['unexpected'] = torch.zeros(1)
            with pytest.raises(RuntimeError):
                model.load_state_dict(weights)
        else:
            model.load_state_dict(weights)
        handle.remove()
        assert observed == [model.model_tag] and model.model_tag != old_tag
        with pytest.raises(ValueError, match='another'):
            model(tokens, refs, owners=('a', 'b'))
        fresh = model.prepare_visual(features, boxes, owners=('a', 'b'), page_ids=('p1', 'p1'))
        with pytest.raises(ValueError, match='another'):
            model(tokens, fresh, owners=('a', 'b'), state=state)
        output, new_state = model(tokens, fresh, owners=('a', 'b'))
        assert torch.isfinite(output).all() and new_state.model_tag == model.model_tag


def test_steady_preserves_tag_and_rejects_cross_instance_or_generation():
    model, features, boxes, refs = tiny('tac')
    model.eval()
    tokens = torch.tensor([[256, 1, 2, 3], [256, 4, 5, 6]])
    with torch.inference_mode():
        _, state = model(tokens, refs, owners=('a', 'b'))
        session = SteadyTACSession(model, refs, state)
        _, next_state = session.step(tokens[:, :1], refs, owners=('a', 'b'), state=state)
        assert next_state.model_tag == state.model_tag == model.model_tag
        # A state accepted by the frozen path must also resume in eager mode.
        model(tokens[:, :1], refs, owners=('a', 'b'), state=next_state)
        foreign = replace(state, model_tag='foreign-instance')
        with pytest.raises(ValueError):
            SteadyTACSession(model, refs, foreign)
        with pytest.raises(ValueError):
            session.step(tokens[:, :1], refs, owners=('a', 'b'), state=foreign)
        # assign=True can replace parameters without changing their version counters.
        model.load_state_dict({k: v.clone() for k, v in model.state_dict().items()}, assign=True)
        fresh = model.prepare_visual(features, boxes, owners=('a', 'b'), page_ids=('p1', 'p1'))
        _, current = model(tokens, fresh, owners=('a', 'b'))
        with pytest.raises(ValueError):
            session.step(tokens[:, :1], fresh, owners=('a', 'b'), state=current)


@pytest.mark.parametrize('family', ['gru', 'transformer', 'tac'])
@pytest.mark.parametrize('clone_kind', ['shallow', 'deep', 'pickle'])
def test_copied_or_unpickled_model_has_distinct_state_ownership(family, clone_kind):
    model, features, boxes, refs = tiny(family)
    tokens = torch.tensor([[256, 1], [256, 2]])
    with torch.no_grad():
        expected, state = model(tokens, refs, owners=('a', 'b'))
        if clone_kind == 'shallow':
            clone = copy.copy(model)
        elif clone_kind == 'deep':
            clone = copy.deepcopy(model)
        else:
            # Trusted in-memory module roundtrip, no dataset/checkpoint files.
            clone = pickle.loads(pickle.dumps(model))
        assert clone is not model and clone.model_tag != model.model_tag
        assert clone.signature == model.signature
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, clone.state_dict()[key], rtol=0, atol=0)
        with pytest.raises(ValueError, match='another'):
            clone(tokens, refs, owners=('a', 'b'))
        own_refs = clone.prepare_visual(features, boxes, owners=('a', 'b'), page_ids=('p1', 'p1'))
        with pytest.raises(ValueError, match='another'):
            clone(tokens, own_refs, owners=('a', 'b'), state=state)
        actual, own_state = clone(tokens, own_refs, owners=('a', 'b'))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert own_state.model_tag == clone.model_tag
