import pytest
from unittest.mock import patch

from scripts import me106_priority_panel as panel
from scripts import diaggrad001_replay as replay


def test_only_global_rounding_counter_is_exempt():
    replay.compare(47, 48, 'result.summary.all.opposing_aux_after')
    with pytest.raises(AssertionError):
        replay.compare(47, 48, 'result.summary.slow.opposing_aux_after')
    with pytest.raises(AssertionError):
        replay.compare(1., 2., 'result.rows[0].measurement.blocks.all.priority_aux_after.dot')


def test_nested_original_gradient_comparison_remains_strict():
    with patch.object(panel, '_compare_numeric', replay.compare):
        with pytest.raises(AssertionError):
            replay.compare({'rows': [{'cosine': .2}]}, {'rows': [{'cosine': .3}]})


def test_scientific_conflict_counts_are_not_exempt():
    with pytest.raises(AssertionError):
        replay.compare(138, 139, 'result.summary.all.pairs.point__interval.negative')
