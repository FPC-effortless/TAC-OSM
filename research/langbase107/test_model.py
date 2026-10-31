from __future__ import annotations

import torch

from research.langbase107.model import CausalTransformer107, TransformerConfig107, parameter_count


def model() -> CausalTransformer107:
    torch.manual_seed(107)
    return CausalTransformer107().float().eval()


def test_registered_parameter_count_is_within_one_tenth_percent_of_tac() -> None:
    transformer = model()
    assert parameter_count(transformer) == 998_816
    assert abs(parameter_count(transformer) - 998_024) / 998_024 < 0.001
    assert transformer.lm_head.weight is transformer.token_embedding.weight


def test_future_bytes_cannot_change_earlier_logits() -> None:
    transformer = model()
    tokens = torch.arange(32, dtype=torch.long)[None]
    changed = tokens.clone()
    changed[:, 17:] = 255
    with torch.no_grad():
        original, _ = transformer(tokens)
        perturbed, _ = transformer(changed)
    torch.testing.assert_close(original[:, :17], perturbed[:, :17], atol=2e-6, rtol=2e-5)


def test_full_and_incremental_kv_cache_are_numerically_equivalent() -> None:
    transformer = model()
    tokens = (torch.arange(65, dtype=torch.long) * 17 % 256)[None]
    with torch.no_grad():
        parallel, _ = transformer(tokens)
        cache = None
        pieces = []
        for token in tokens.unbind(dim=1):
            logits, cache = transformer(token[:, None], cache=cache, use_cache=True)
            pieces.append(logits)
    recurrent = torch.cat(pieces, dim=1)
    torch.testing.assert_close(parallel, recurrent, atol=2e-5, rtol=2e-4)
    assert cache.position == tokens.shape[1]
    assert cache.tensor_bytes() == 2 * 4 * 8 * 65 * 16 * 4


def test_cache_is_bounded_and_all_parameters_receive_finite_gradients() -> None:
    transformer = model().train()
    tokens = torch.arange(128, dtype=torch.long)[None].repeat(2, 1)
    targets = (tokens + 1) % 256
    logits, _ = transformer(tokens)
    torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten()).backward()
    assert all(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
               for parameter in transformer.parameters())
    transformer.eval()
    cache = None
    with torch.no_grad():
        for _ in range(3):
            _, cache = transformer(tokens, cache=cache, use_cache=True)
    assert all(key.shape[2] == TransformerConfig107().context for key in cache.keys)
    assert cache.tensor_bytes() == 2 * 2 * 4 * 8 * 128 * 16 * 4

