"""Tests for factorized product-key persistent-state addressing."""

from __future__ import annotations

from tac_osm.product_key_index import ProductKeyConfig, ProductKeyStateIndex


def _items():
    return [(f"s{i}", tuple(float(1 if d == i % 4 else 0) for d in range(4))) for i in range(16)]


def test_product_key_rejects_odd_embedding_dimensions():
    index = ProductKeyStateIndex()
    try:
        index.build([("a", (1.0, 0.0, 0.0))])
    except ValueError as exc:
        assert "even dimension" in str(exc)
    else:
        raise AssertionError("odd product-key embeddings must fail")


def test_product_key_builds_two_factor_cells():
    index = ProductKeyStateIndex(ProductKeyConfig(factor_size=4, factor_beam=2, iterations=2, max_shortlist=8))
    diag = index.build(_items())
    assert diag.state_items == 16
    assert diag.factor_size == 4
    assert diag.factor_dim == 2
    assert diag.nonempty_cells >= 1


def test_product_key_lookup_is_bounded():
    index = ProductKeyStateIndex(ProductKeyConfig(factor_size=4, factor_beam=2, iterations=2, max_shortlist=4))
    index.build(_items())
    hit = index.lookup((1.0, 0.0, 0.0, 0.0))
    assert len(hit.candidate_addresses) <= 4
    assert hit.state_rerank_macs <= 16
    assert hit.factor_score_macs == 16


def test_product_key_requires_even_uniform_embeddings():
    index = ProductKeyStateIndex()
    try:
        index.build([("a", (1.0, 0.0)), ("b", (1.0, 0.0, 0.0, 0.0))])
    except ValueError as exc:
        assert "equal dimension" in str(exc)
    else:
        raise AssertionError("mixed dimensions must fail")

def test_product_key_codebooks_can_be_fit_on_a_training_subset():
    items = _items()
    train = items[:8]
    index = ProductKeyStateIndex(
        ProductKeyConfig(factor_size=4, factor_beam=2, iterations=2, max_shortlist=4)
    )
    diag = index.build(items, codebook_items=train)
    assert diag.state_items == 16
    assert diag.codebook_training_items == 8


def test_product_key_rejects_training_items_outside_runtime_pool():
    index = ProductKeyStateIndex(ProductKeyConfig(factor_size=4, factor_beam=2, iterations=2))
    try:
        index.build(_items(), codebook_items=[("missing", (1.0, 0.0, 0.0, 0.0))])
    except ValueError as exc:
        assert "subset" in str(exc)
    else:
        raise AssertionError("external codebook item must fail")
