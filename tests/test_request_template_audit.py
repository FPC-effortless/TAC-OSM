"""Constructed software falsifiers, never tool-calling capability evidence."""
import copy
import pytest
from tac_osm.csl125_training_data import DataContractError,query_group,split_for
from tac_osm.request_template_audit import request_keys,audit_requests


def lineage(requests):
    return [{'record_index':i,'disposition':'accepted','partition':split_for(query_group(q))} for i,q in requests]

@pytest.mark.parametrize('left,right',[
 ('Call 123-456-7890','Call 900-800-7000'),
 ('set alarm for 09:30','set alarm for 12:15'),
 ('schedule 2026/11/01','schedule 2026/12/02'),
 ('increase by -2.5','increase by +7.0'),
])
def test_numbers(left,right):
    assert request_keys(left)['exact'] != request_keys(right)['exact']
    assert request_keys(left)['numbers'] == request_keys(right)['numbers']

@pytest.mark.parametrize('left,right',[
 ('send "hello" to "Ada"','send "world" to "Bob"'),
 ("send 'hello' to 'Ada'", "send 'world' to 'Bob'"),
 ('send "quote \\" here"','send "other"'),
])
def test_quotes(left,right):
    assert request_keys(left)['quotes_numbers'] == request_keys(right)['quotes_numbers']


def test_no_intent_or_marker_collision():
    for key in ('numbers','quotes_numbers'):
        assert request_keys('call 12')[key] != request_keys('delete 12')[key]
        assert request_keys('call number')[key] != request_keys('call 12')[key]
    assert request_keys('send string')['quotes_numbers'] != request_keys('send "x"')['quotes_numbers']
    assert request_keys('version12')['numbers'] != request_keys('version13')['numbers']
    assert request_keys("don't call 12")['quotes_numbers'] != request_keys('do call 12')['quotes_numbers']


def test_normalization_exact_matches_legacy():
    assert request_keys('  ＣＡＬＬ\t１２  ')['exact'] == query_group('call 12')


def test_only_requests_enter_grouping_label_mutation_irrelevant():
    source=[{'query':'call 12','answers':[1],'tools':['a']},{'query':'call 13','answers':[2],'tools':['b']}]
    projected=lambda rs:[(i,r['query']) for i,r in enumerate(rs,1)]
    before=audit_requests(projected(source),lineage(projected(source)))
    mutated=copy.deepcopy(source)
    for row in mutated:row.update(answers=['FORBIDDEN'],tools=['LABEL'],future='poison')
    assert audit_requests(projected(mutated),lineage(projected(source))) == before
    with pytest.raises(DataContractError):request_keys(mutated[0])


def test_frozen_join_and_order_invariance():
    # Select request variants with different original exact partitions, not gold labels.
    fit=next('call '+str(i) for i in range(100) if split_for(query_group('call '+str(i)))=='FIT')
    dev=next('call '+str(i) for i in range(100) if split_for(query_group('call '+str(i)))=='DEV')
    rows=[(1,fit),(2,dev)];old=lineage(rows);saved=copy.deepcopy(old)
    result,ledger=audit_requests(rows,old)
    assert result['exact']['accepted_cross_partition_groups']==0
    assert result['numbers']['DEV_records_with_FIT_partner']==1
    assert result['numbers']['DEV_fraction_with_FIT_partner']==1
    assert audit_requests(rows[::-1],old[::-1])==(result,ledger)
    assert old==saved

@pytest.mark.parametrize('rows,old',[
 ([],[]), ([(1,'x'),(1,'y')],[]), ([(True,'x')],[]),
 ([(1,'x')],[]), ([(1,'x')],[{'record_index':1,'disposition':'accepted','partition':'TEST'}]),
 ([(1,'x')],[{'record_index':1,'disposition':'accepted','partition':'FIT'}]*2),
])
def test_invalid_cohorts(rows,old):
    with pytest.raises(DataContractError):audit_requests(rows,old)

@pytest.mark.parametrize('query',[None,'','   ',[], 'x'*16385])
def test_invalid_request(query):
    with pytest.raises(DataContractError):request_keys(query)


def test_exact_partition_corruption_rejected():
    rows=[(1,'same'),(2,'SAME')]
    old=[{'record_index':1,'disposition':'accepted','partition':'FIT'},{'record_index':2,'disposition':'accepted','partition':'DEV'}]
    with pytest.raises(DataContractError,match='original_exact'):audit_requests(rows,old)


def test_quarantine_does_not_remove_full_source_fingerprint():
    rows=[(1,'call 12'),(2,'call 13')]
    old=[{'record_index':1,'disposition':'quarantine','partition':None},{'record_index':2,'disposition':'accepted','partition':'DEV'}]
    result,ledger=audit_requests(rows,old)
    assert len(ledger)==2 and result['numbers']['all_TRAIN_request_groups']==1
    assert result['numbers']['DEV_records_with_FIT_partner']==0
