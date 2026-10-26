import pytest
from scripts.diaggrad001_precision import strict_compare


def test_fp64_contract_rejects_wider_float_error():
    strict_compare({'value': 1.}, {'value': 1.+1e-11})
    with pytest.raises(AssertionError):
        strict_compare({'value': 1.}, {'value': 1.+1e-7})


def test_fp64_summary_counts_and_source_identity_are_exact():
    with pytest.raises(AssertionError):
        strict_compare({'count': 139}, {'count': 138})
    with pytest.raises(AssertionError):
        strict_compare({'source': 'a'}, {'source': 'b'})
