"""One representable step is confined to the actual transcendental transform."""
from pathlib import Path
import sys
import pytest
np = pytest.importorskip('numpy')
torch = pytest.importorskip('torch')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from reconstruction_float64_004 import compare_inputs


def test_only_log_transform_allows_one_ulp_and_no_other_change():
    schema = dict(categorical=['status'], vocabulary={'status':['broken']}, numeric=['count'])
    original = np.array([[1., 0., 0., np.log1p(8.), 0.]], dtype=np.float64)
    one = original.copy(); one[0, 3] = np.nextafter(one[0, 3], np.inf)
    stats = dict(values=0, max_abs=0., max_ULP=0)
    compare_inputs(torch.tensor(one), torch.tensor(original), schema, stats)
    assert stats['max_ULP'] == 1 and stats['max_abs'] < 1e-15
    two = one.copy(); two[0, 3] = np.nextafter(two[0, 3], np.inf)
    with pytest.raises(AssertionError): compare_inputs(torch.tensor(two), torch.tensor(original), schema, stats)
    for column in (0, 4):
        wrong = original.copy(); wrong[0, column] = np.nextafter(wrong[0, column], np.inf)
        with pytest.raises(AssertionError): compare_inputs(torch.tensor(wrong), torch.tensor(original), schema, stats)
