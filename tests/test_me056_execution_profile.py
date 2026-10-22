import numpy as np
import pytest

from scripts.me056_profile_paired_execution import percentile_interval


def test_bootstrap_interval_is_deterministic_and_contains_constant_effect():
    assert percentile_interval([.1]*11)==pytest.approx([.1,.1])


def test_bootstrap_rejects_too_few_or_nonfinite_pairs():
    with pytest.raises(ValueError):
        percentile_interval([.1]*4)
    with pytest.raises(ValueError):
        percentile_interval([.1]*10+[np.nan])
