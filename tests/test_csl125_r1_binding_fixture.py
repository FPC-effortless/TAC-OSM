"""Source-isolated no-fit counterexamples; no neural optimization."""
import pytest
from tac_osm.csl125_r1_binding_fixture import (
    ABSENT, Event, always_absent, emit_report, exact_oracle,
    fixture_digest, latest_value_without_identity, make_partition,
    score, six_query_history,
)


def test_independent_oracle_six_queries_and_replacement():
    row=six_query_history(('key-a','key-b','missing'),(3,5,7))
    assert exact_oracle(row)==(3,5,ABSENT,7,5,ABSENT)
    assert len(row)==9
    assert all(e.value is None for e in row if e.kind=='query')


def test_absent_frequency_cannot_pass_six_answer_histories():
    rows=make_partition('DEV',count=128)
    truth=score(tuple(exact_oracle(x) for x in rows),rows)
    assert truth['complete_histories']==128
    absent=score(tuple(always_absent(x) for x in rows),rows)
    assert absent['exact_answers']==256
    assert absent['query_count']==768
    assert absent['complete_histories']==0
    latest=score(tuple(latest_value_without_identity(x) for x in rows),rows)
    assert latest['complete_histories']==0
    assert latest['exact_answers']<truth['exact_answers']


def test_train_dev_identity_disjoint_and_hash_replay():
    a=make_partition('TRAIN',count=64,seed=87)
    b=make_partition('DEV',count=64,seed=87)
    keys=lambda rows: {ev.key for row in rows for ev in row}
    assert not(keys(a)&keys(b))
    assert fixture_digest(a)==fixture_digest(make_partition('TRAIN',count=64,seed=87))
    assert fixture_digest(a)!=fixture_digest(b)


def test_future_only_oracle_labels_never_in_prediction_features():
    row=make_partition('TRAIN',count=1)[0]
    poisoned=list(exact_oracle(row));poisoned[-1]=99
    assert poisoned!=list(exact_oracle(row))
    assert all(ev.kind!='query' or ev.value is None for ev in row)
    assert exact_oracle(row).count(ABSENT)==2


def test_reject_untrusted_invalid_oracle_and_scoring_shapes():
    with pytest.raises(ValueError,match='query cannot expose'):
        Event('query','key',value=3)
    with pytest.raises(ValueError,match='distinct'):
        six_query_history(('x','x','z'),(1,2,3))
    row=six_query_history(('x','y','z'),(1,2,3))
    with pytest.raises(ValueError,match='six-query'):
        score(((1,2),),(row,))
    with pytest.raises(ValueError,match='nonempty'):
        score((),())


def test_oracle_and_negative_control_receipt_scope():
    report=emit_report()
    assert report['model_calls']==report['optimizer_updates']==0
    assert not report['source_qualified']
    assert report['train_dev_key_overlap']==0
    assert report['scores']['symbolic_exact_oracle']['complete_histories']==128
    assert report['scores']['always_absent']['complete_histories']==0


def test_query_order_randomized_across_histories_prevent_position_shortcut():
    rows=make_partition('TRAIN',count=128,seed=918)
    def orders(events):
        first=events[0].key;second=events[1].key
        absent=next(e.key for e in events if e.kind=='query' and e.key not in (first,second))
        keys=[first,second,absent]
        return (tuple(keys.index(e.key) for e in events[2:5]),
                tuple(keys.index(e.key) for e in events[6:9]))
    pairs={orders(row) for row in rows}
    assert len(pairs)>=20, 'fixed positions could leak answer roles'
    assert all(sorted(a)==[0,1,2] and sorted(b)==[0,1,2] for a,b in pairs)
