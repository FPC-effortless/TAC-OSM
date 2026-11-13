"""Request-only lexical overlap warning; never semantic/label qualification.

Grouping precedes frozen disposition join. Accepted membership itself depended
on historic labels/schema. These fingerprints ignore labels conditional on that
fixed cohort; they do not establish semantic equivalence or independence.
"""
from collections import defaultdict
import re
import unicodedata

from tac_osm.csl125_training_data import DataContractError, digest, query_group

METHODS = ('exact', 'numbers', 'quotes_numbers')
NUMBER = r'(?<!\w)[+-]?\d+(?:[.,:/-]\d+)*(?!\w)'
QUOTED = r'"(?:\\.|[^"\\])*"|(?<!\w)\x27(?:\\.|[^\x27\\])*\x27(?!\w)'
SCANNERS = {
    'numbers': re.compile(r'(?P<number>'+NUMBER+r')|(?P<word>\w+)', re.UNICODE),
    'quotes_numbers': re.compile(r'(?P<string>'+QUOTED+r')|(?P<number>'+NUMBER+r')|(?P<word>\w+)', re.UNICODE),
}


def request_keys(query):
    """Accept only one string, never answers, tool lists or label dictionaries."""
    exact = query_group(query)
    if len(query.encode('utf-8')) > 16384:
        raise DataContractError('request_audit_byte_budget')
    normalized = unicodedata.normalize('NFKC', query).casefold()
    result = {'exact': exact}
    for method, scanner in SCANNERS.items():
        # Typed markers cannot collide with words such as 'number' or '<string>'.
        tokens = [(m.lastgroup, m.group() if m.lastgroup == 'word' else None)
                  for m in scanner.finditer(normalized)]
        result[method] = digest([method, tokens])
    return result


def audit_requests(requests, lineage):
    """Pure audit of explicit (index,query) pairs plus immutable dispositions.

    All fingerprints are built before reading/joining any disposition. No split
    creation, answer-dependent redaction, threshold tuning or model execution.
    """
    rows = {}
    for index, query in requests:
        if type(index) is not int or index < 1 or index in rows:
            raise DataContractError('invalid_or_duplicate_request_index')
        rows[index] = request_keys(query)
    if not rows:
        raise DataContractError('empty_request_cohort')
    dispositions = {}
    for entry in lineage:
        index = entry['record_index']
        category, partition = entry['disposition'], entry['partition']
        if type(index) is not int or index not in rows or index in dispositions:
            raise DataContractError('invalid_or_duplicate_lineage_index')
        if category not in ('accepted','quarantine','duplicate') or (
            category == 'accepted' and partition not in ('FIT','DEV')) or (
            category != 'accepted' and partition is not None):
            raise DataContractError('invalid_frozen_disposition')
        dispositions[index] = (category, partition)
    if set(dispositions) != set(rows):
        raise DataContractError('missing_frozen_disposition')
    ledger = [{'record_index': i, 'keys': rows[i],
               'disposition': dispositions[i][0], 'partition': dispositions[i][1]}
              for i in sorted(rows)]
    results = {}
    for method in METHODS:
        groups = defaultdict(list)
        for row in ledger:
            groups[row['keys'][method]].append(row)
        cross = []
        for key, members in sorted(groups.items()):
            fit = [r['record_index'] for r in members if r['partition'] == 'FIT']
            dev = [r['record_index'] for r in members if r['partition'] == 'DEV']
            if fit and dev:
                cross.append({'key':key,'FIT':fit,'DEV':dev})
        dev_count = sum(r['partition'] == 'DEV' for r in ledger)
        affected = sum(len(g['DEV']) for g in cross)
        results[method] = {
            'all_TRAIN_request_groups':len(groups),
            'accepted_FIT_records':sum(r['partition'] == 'FIT' for r in ledger),
            'accepted_DEV_records':dev_count,
            'accepted_cross_partition_groups':len(cross),
            'DEV_records_with_FIT_partner':affected,
            'DEV_fraction_with_FIT_partner':affected/dev_count if dev_count else None,
            'largest_all_TRAIN_group':max(map(len,groups.values())),
            'cross_partition_groups':cross,
        }
    if results['exact']['accepted_cross_partition_groups']:
        raise DataContractError('original_exact_group_partition_invariant_failed')
    return results, ledger
