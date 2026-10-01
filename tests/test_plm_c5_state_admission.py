from tac_osm.c5_state_addressing import C5PersistentStateAdmission, make_persistent_query


def test_c5_state_adapter_reuses_existing_lsh_contract():
    adapter = C5PersistentStateAdmission(seed=0, calibration_trials=4)
    records = adapter.target_records(1024)
    descriptor = records[0].key_bits
    query = make_persistent_query(descriptor, seed=0, step=0)
    result = adapter.lookup(
        query=query,
        m=1024,
        target_ids=(records[0].record_id,),
    )
    assert result.requested_tables >= 1
    assert result.actual_tables >= 1
    assert result.raw_candidates >= result.admitted_candidates


def test_c5_state_adapter_reports_routing_fraction():
    adapter = C5PersistentStateAdmission(seed=1, calibration_trials=4)
    records = adapter.target_records(1024)
    q = make_persistent_query(records[0].key_bits, seed=1, step=0)
    result = adapter.lookup(query=q, m=1024, target_ids=(records[0].record_id,))
    assert result.routing_ops >= result.raw_candidates
    assert result.routing_ops / 1024.0 >= 0.0
