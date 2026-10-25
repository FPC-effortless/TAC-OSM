"""Replay provenance stays exact while live tensor tolerance checks remain authoritative."""
import pytest
from scripts.me102_fitted_graph_audit import replay_identity


def test_error_maxima_are_diagnostics_not_cross_cpu_bit_contract():
    first = {'checkpoint_sha256': 'original', 'maximum_absolute_errors': {'gradient': 1e-7}}
    second = {'checkpoint_sha256': 'original', 'maximum_absolute_errors': {'gradient': 2e-7}}
    assert replay_identity(first) == replay_identity(second)
    assert first != second  # Original recorded observations are still different.
    second['checkpoint_sha256'] = 'different'
    assert replay_identity(first) != replay_identity(second)
    second = {'checkpoint_sha256': 'original', 'maximum_absolute_errors': {'missing_check': 1e-7}}
    assert replay_identity(first) != replay_identity(second)


@pytest.mark.parametrize('invalid', [float('nan'), float('inf'), -1.0])
def test_invalid_error_diagnostics_are_rejected(invalid):
    with pytest.raises(RuntimeError, match='invalid live error'):
        replay_identity({'maximum_absolute_errors': {'gradient': invalid}})
