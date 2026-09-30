"""Tests for the logical packed persistent-state layout."""

from __future__ import annotations

from tac_osm.packed_state import PackedStateLayout


def test_packed_layout_is_row_major_and_stable():
    layout = PackedStateLayout.from_items(
        [
            ("b", (2.0, 0.0)),
            ("a", (1.0, 0.0)),
            ("c", (0.0, 1.0)),
        ]
    )
    assert layout.addresses == ("a", "b", "c")
    assert layout.row("b") == 1


def test_packed_layout_reports_contiguous_runs():
    items = [(str(i), (float(i), 1.0)) for i in range(8)]
    layout = PackedStateLayout.from_items(items)
    assert layout.layout_diagnostics(["1", "2", "3"])["contiguous_runs"] == 1
    assert layout.layout_diagnostics(["1", "3", "5"])["contiguous_runs"] == 3
    blocked = PackedStateLayout.from_blocks(items, (("0", "1", "2", "3"), ("4", "5", "6", "7")))
    assert blocked.addresses[:4] == ("0", "1", "2", "3")
    assert blocked.layout_diagnostics(["0", "1", "2", "3"])["contiguous_runs"] == 1


def test_packed_layout_rejects_empty_or_mixed_dimensions():
    try:
        PackedStateLayout.from_items([])
    except ValueError as exc:
        assert "at least one" in str(exc)
    else:
        raise AssertionError("empty packed layout must fail")

    try:
        PackedStateLayout.from_items([("a", (1.0,)), ("b", (1.0, 2.0))])
    except ValueError as exc:
        assert "uniform dimension" in str(exc)
    else:
        raise AssertionError("mixed embedding dimensions must fail")
