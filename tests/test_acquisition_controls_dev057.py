import pytest
from tac_osm.development.acquisition_controls import (
    AcquisitionStore,Work,execute_xor,benchmark,measure,ARMS)


def test_index_and_scan_match_latest_facts_and_charge_full_build():
    events,queries,gold=benchmark(8571001,256,8)
    for arm in ('full_scan','indexed_acquisition'):
        store=AcquisitionStore(arm,256)
        for event in events:store.write(*event)
        for keys in queries:assert store.acquire(keys)==[gold[k] for k in keys]
        assert store.work.build_input_reads==256
        assert store.work.record_probes==8*(256 if arm=='full_scan' else 8)


def test_equal_bits_stores_own_exactly_the_same_payload_and_no_history():
    for arm in ('equal_bits_scalar','hash_direct','hash_two_way'):
        store=AcquisitionStore(arm,4096)
        assert store.payload_bytes==768
        assert set(store.__dict__)=={'arm','work','records'}


def test_collision_is_missing_not_a_wrong_key_value():
    store=AcquisitionStore('hash_direct',64)
    store.write(1,1,1);store.write(65,2,0)
    facts=store.acquire([1,65,2,3,4,5,6,7])
    assert facts[0] is None and facts[1]==(2,0)
    assert execute_xor(facts,store.work) is None
    assert store.work.structural_operations==0
    two=AcquisitionStore('hash_two_way',64)
    two.write(1,1,1);two.write(33,2,0);two.write(65,3,1)
    facts=two.acquire([1,33,65,2,3,4,5,6])
    assert facts[0] is None and facts[1:3]==[(2,0),(3,1)]


def test_stale_update_and_missing_query_never_invent_current_facts():
    for arm in ARMS:
        store=AcquisitionStore(arm,64)
        store.write(7,10,1);store.write(7,2,0)
        facts=store.acquire([7,8,9,10,11,12,13,14])
        assert facts[0]==(10,1) and facts[1] is None
        assert execute_xor(facts,store.work) is None


def test_actual_circuit_work_and_verification_are_counted():
    work=Work()
    assert execute_xor([(i,1) for i in range(8)],work)==0
    assert work.structural_operations==7
    cell=measure(8571002,64,8,'indexed_acquisition')
    assert cell['success']==1
    assert cell['work']['structural_operations']==56
    assert cell['work']['verification_reads']==64
    assert cell['work']['verification_operations']==64
    assert cell['amortized_declared_operations']>cell['work']['record_probes']/8


def test_query_prefixes_and_budget_failures_are_explicit():
    a=benchmark(8571003,1024,8);b=benchmark(8571003,1024,64)
    assert a[0]==b[0] and a[1]==b[1][:8]
    with pytest.raises(ValueError):AcquisitionStore('hash_direct',64).acquire([1]*8)
    with pytest.raises(ValueError):AcquisitionStore('full_scan',64).write(2,65,1)
