"""Do not weaken existing graph or historical parity when adding a candidate."""
from copy import deepcopy
import pytest

from research.scripts.check_pnds_remote_parity import compare_existing_snapshot


def test_existing_snapshot_preserved_and_only_candidate_array_is_separate():
    old = {'variants': [{'id': 'original'}], 'prospective_graph_index': {'ids': ['103']}}
    current = {**old, 'prospective_candidates': [{'id': '104'}]}
    assert compare_existing_snapshot(current, old) == [{'id': '104'}]
    for key in old:
        changed = deepcopy(current)
        changed[key] = []
        with pytest.raises(ValueError):
            compare_existing_snapshot(changed, old)


def test_unexpected_registry_field_is_not_ignored():
    with pytest.raises(ValueError):
        compare_existing_snapshot({'variants': [], 'unreviewed': True}, {'variants': []})


def test_invalid_candidate_array_fails_closed():
    for candidates in (None, '104', [None], [3]):
        with pytest.raises(ValueError):
            compare_existing_snapshot({'variants': [], 'prospective_candidates': candidates}, {'variants': []})
