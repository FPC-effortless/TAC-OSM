"""Outcome-time units, leakage protection and genuinely held-out entities."""
from pathlib import Path
import sys
import pytest
torch = pytest.importorskip('torch')
np = pytest.importorskip('numpy')
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from tac_osm.development.repair_outcome import RepairOutcomeForecaster
from data_real_maintenance_004 import columns, fit_schema, features, split, labeled


def test_zero_initialized_forecast_is_train_median_and_keeps_seconds():
    model = RepairOutcomeForecaster(3, [0]*3, [1]*3, 123.123456789)
    out = model(torch.tensor([[0., 1., 2.], [8., -3., 1.]], dtype=torch.float64))
    torch.testing.assert_close(out, torch.tensor([123.123456789]*2, dtype=torch.float64), atol=1e-12, rtol=0)


def test_outcomes_and_identity_cannot_enter_features():
    row = dict(functional_status_before='broken', replaced_cylinder_count='2',
        replaced_cylinder_tick='yes', chlorination_done='no', water_point_name='secret',
        time_to_fill_20l_bucket='72', functionality_after='repaired', strokes_to_yield_water='6',
        latitude='-1', submitted_on='2020-03-01', rating_cylinder='1')
    rows = [dict(row), dict(row, functional_status_before='unseen', replaced_cylinder_count='')]
    schema = fit_schema(rows, [0], 'repair')
    before = features(rows, [0, 1], schema)
    for r in rows:
        for name in ('water_point_name', 'time_to_fill_20l_bucket', 'functionality_after',
                     'strokes_to_yield_water', 'latitude', 'submitted_on', 'rating_cylinder'): r[name] = 'UNSEEN_OUTCOME'
    assert np.array_equal(before, features(rows, [0, 1], schema))
    assert 'unseen' not in schema['vocabulary']['functional_status_before']
    assert before[0, -1] == 0 and before[1, -1] == 1
    assert schema['numeric_imputation']['replaced_cylinder_count'] == 2


def test_grouped_time_split_never_reads_targets():
    class Guard(dict):
        def __getitem__(self, name):
            if name not in ('water_point_name', 'chlorination_rehab_date'): raise AssertionError('Outcome read before split')
            return super().__getitem__(name)
    rows = [Guard(water_point_name=f'preview{i}', chlorination_rehab_date='2019-01-01') for i in range(10)]
    for i in range(30):
        rows.append(Guard(water_point_name=f'asset{i}', chlorination_rehab_date=f'2020-01-{i+1:02d}'))
    # Repeated group spanning time boundary must be purged or wholly early.
    rows.append(Guard(water_point_name='asset0', chlorination_rehab_date='2020-01-30'))
    assignment, groups, meta = split(rows)
    sets = {name: {groups[i] for i in ids} for name, ids in assignment.items()}
    assert not sets['train'] & sets['validation'] and not sets['train'] & sets['test']
    assert not sets['validation'] & sets['test']
    assert 10 in assignment['purged'] and 40 in assignment['purged']
    assert meta['excluded_preview_groups'] == 10


def test_malformed_targets_are_not_imputed_or_clipped():
    rows = [dict(time_to_fill_20l_bucket=value) for value in ('60', '', '0', '-1', 'inf', '1e8')]
    ids, target, excluded = labeled(rows, range(6))
    assert ids == [0, 5] and excluded == [1, 2, 3, 4]
    assert target.tolist() == [60., 1e8]
