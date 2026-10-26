"""Constructed-fixture checks; no source loading or fitting."""
import numpy as np
import pytest
from scripts import me110_sampling as sampling


def fixture(general_size=1000):
    gap = [(0, 255 + i) for i in range(300)]
    return dict(phase='fit', gap=gap, general=gap + [(1+i//15000, 255+i%15000) for i in range(general_size)])


def test_monitor_fixed_and_disjoint():
    d = fixture(); m = sampling.make_monitor(d)
    assert m.shape == (320, 2) and np.array_equal(m, sampling.make_monitor(d))
    assert len(set(map(tuple,m))) == 320
    assert set(map(tuple,m[:64])).issubset(d['gap'])
    assert not set(map(tuple,m[64:])) & set(d['gap'])
    assert 'not fresh or independent' in sampling.MONITOR_SCOPE


def test_seeded_plan_cycles_and_excludes_monitor():
    d = fixture(); m = sampling.make_monitor(d)
    p = sampling.make_plan(d,110001,m,nsteps=100)
    assert np.array_equal(p,sampling.make_plan(d,110001,m,nsteps=100))
    assert not np.array_equal(p,sampling.make_plan(d,110002,m,nsteps=100))
    assert not set(map(tuple,p.reshape(-1,2))) & set(map(tuple,m))
    r = sampling.validate_plan(p,d,m,nsteps=100)
    assert r['unique_gap_origins'] == 200 and r['unique_general_origins'] == 744


def test_full_plan_diversity():
    d=fixture(60000); m=sampling.make_monitor(d); p=sampling.make_plan(d,110001,m)
    assert p.shape==(4096,16,2)
    assert sampling.validate_plan(p,d,m)['unique_general_origins']==57344


def test_each_cycle_unique():
    d=fixture();m=sampling.make_monitor(d);p=sampling.make_plan(d,110001,m,nsteps=200)
    sampling.validate_plan(p,d,m,nsteps=200);rows=p[:,2:].reshape(-1,2)
    for start in (0,744): assert len(set(map(tuple,rows[start:start+744])))==744


def test_validator_rejects_shape_overlap_and_repetition():
    d=fixture();m=sampling.make_monitor(d);p=sampling.make_plan(d,110001,m,nsteps=10)
    bads=[p[:,:,0],p.astype(float)]
    for origin in (m[0],p[0,2],p[1,0]):
        bad=p.copy();bad[0,0]=origin;bads.append(bad)
    for bad in bads:
        with pytest.raises(ValueError):sampling.validate_plan(bad,d,m,nsteps=10)


@pytest.mark.parametrize('change',['phase','float','short','duplicate','outside'])
def test_pool_boundaries(change):
    d=fixture()
    if change=='phase':d['phase']='report'
    if change=='float':d['gap']=np.asarray(d['gap'],dtype=float)
    if change=='short':d['gap'][0]=(0,254)
    if change=='duplicate':d['gap'][0]=d['gap'][1]
    if change=='outside':d['gap'][0]=(0,sampling.FIT_STOP-6)
    with pytest.raises(ValueError):sampling.make_monitor(d)


def test_full_protocol_requires_general_diversity():
    d=fixture()
    with pytest.raises(ValueError,match='57344'):sampling.make_plan(d,110001,sampling.make_monitor(d))
