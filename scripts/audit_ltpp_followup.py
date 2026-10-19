"""Posthoc source eligibility audit: date gaps/cointerventions, never effects.

Public preview requests retrieve rehabilitation dates missing from the first
coverage audit. No condition-value changes, model fitting or data joining across
unrelated assets. Raw pages stay local; only hashes/coverage metadata published.
"""
import argparse
from collections import defaultdict, Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from acquire_ltpp_maintenance import post


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(root, table):
    dictionary = json.loads((root/f'{table}-schema.json').read_text())
    available = [r['FIELD_NAME'] for r in dictionary['tableDictionary']]
    from acquire_ltpp_maintenance import PROFILE
    fields = PROFILE if table == 'MON_HSS_PROFILE_SECTION' else available
    rows = []
    for path in sorted(root.glob(f'{table}-page*.json')):
        rows += [dict(zip(fields, row['cell'], strict=True)) for row in json.loads(path.read_text())['rows']]
    assert rows
    return rows


def run(directory, output):
    root = Path(directory)
    original = json.loads((root/'source-audit.json').read_text())
    for name, expected in original['files_sha256'].items():
        assert sha(root/name) == expected, name
    # One additional public table, normal documented preview path, at most10pages.
    dictionary_path = root/'RHB_IMP-schema.json'
    if not dictionary_path.exists():
        raw, dictionary = post('/Data/GetTableDictionaryInformation', {'tableName': 'RHB_IMP'}, True)
        assert dictionary['OK'] is True
        dictionary_path.write_bytes(raw)
    fields = [r['FIELD_NAME'] for r in json.loads(dictionary_path.read_text())['tableDictionary']]
    assert {'STATE_CODE', 'SHRP_ID', 'IMP_DATE', 'IMP_COST'} <= set(fields)
    page, count, total = 1, 0, None
    while total is None or count < total:
        path = root/f'RHB_IMP-page{page:03d}.json'
        if path.exists():
            response = json.loads(path.read_text())
        else:
            raw, response = post('/Data/GetPreviewData', dict(
                page=page, rows=10000, sidx='STATE_CODE', sord='asc', totalRecords=0,
                selectedFilter='{}', schemaName='LTPPDBA', tableName='RHB_IMP',
                fields=','.join(fields), sqlExportQuery='', moduleName='tableexport'))
            assert response['Message'] == 'success'
            path.write_bytes(raw)
        total = int(response['records']) if total is None else total
        assert int(response['records']) == total and response['rows']
        count += len(response['rows']); page += 1
        assert page <= 11
    assert count == total
    tables = {name: load(root, name) for name in ('MNT_COST', 'MNT_IMP', 'RHB_IMP', 'MON_HSS_PROFILE_SECTION')}
    date = lambda value: datetime.strptime(value, '%m/%d/%Y')
    key = lambda row: (row['STATE_CODE'], row['SHRP_ID'])
    visits, maintenance, rehabilitation = defaultdict(set), defaultdict(list), defaultdict(list)
    missing_dates = Counter()
    for row in tables['MON_HSS_PROFILE_SECTION']:
        if row['MRI'].strip() and row['VISIT_DATE'].strip():
            visits[key(row)].add(date(row['VISIT_DATE']))
    for table, destination in (('MNT_IMP', maintenance), ('RHB_IMP', rehabilitation)):
        for row in tables[table]:
            if row['IMP_DATE'].strip():
                destination[key(row)].append(date(row['IMP_DATE']))
            else:
                missing_dates[table] += 1
    events = []
    for row in tables['MNT_COST']:
        if not any(v.strip() and float(v) > 0 for k, v in row.items() if k.endswith('_TOT_COST')):
            continue
        if not row['DATE_COMPLETE'].strip():
            missing_dates['MNT_COST'] += 1; continue
        target = date(row['DATE_COMPLETE']); before = [d for d in visits[key(row)] if d < target]
        after = [d for d in visits[key(row)] if d > target]
        if not before or not after:
            continue
        left, right = max(before), min(after)
        events.append(dict(state=key(row)[0], section=key(row)[1], treatment=target.date().isoformat(),
            before_date=left.date().isoformat(), after_date=right.date().isoformat(),
            before_days=(target-left).days, after_days=(right-target).days,
            other_maintenance_rows=sum(left <= d <= right and d != target for d in maintenance[key(row)]),
            same_day_maintenance_rows=sum(d == target for d in maintenance[key(row)]),
            rehabilitation_rows_between_measurements=sum(left <= d <= right for d in rehabilitation[key(row)])))
    assert len(events) == original['positive_cost_events_with_measured_MRI_before_and_after']
    def clean(event):
        return event['other_maintenance_rows'] == 0 and event['rehabilitation_rows_between_measurements'] == 0
    result = dict(status='SOURCE_ELIGIBILITY_ONLY_NO_MODEL', retrieved_utc=datetime.now(timezone.utc).isoformat(),
        original_source_audit_sha256=sha(root/'source-audit.json'),
        original_table_files_sha256=original['files_sha256'],
        additional_files_sha256={p.name: sha(p) for p in sorted(root.glob('RHB_IMP-*.json'))},
        rehabilitation_rows=len(tables['RHB_IMP']),
        rehabilitation_record_status_counts=dict(Counter(r['RECORD_STATUS'] for r in tables['RHB_IMP'])),
        missing_date_rows=dict(missing_dates), events=len(events),
        both_measurements_within_days={str(n): sum(max(e['before_days'], e['after_days']) <= n for e in events) for n in (30, 90, 180, 365)},
        events_with_other_recorded_maintenance=sum(e['other_maintenance_rows'] > 0 for e in events),
        events_with_recorded_rehabilitation=sum(e['rehabilitation_rows_between_measurements'] > 0 for e in events),
        events_without_these_recorded_cointerventions=sum(clean(e) for e in events),
        within_90days_without_these_recorded_cointerventions=sum(clean(e) and max(e['before_days'], e['after_days']) <= 90 for e in events),
        event_metadata=events,
        limits='Exploratory coverage, not preregistered sample selection or a treatment-effect estimate. Positive component total is not complete job cost. No condition values compared. Missing/other interventions, same-day bundles, traffic/weather and inspection prices remain unresolved. Recorded absence is not proof of no cointervention. Pavement is not factory equipment, and roughness is not independent binary repair verification.')
    Path(output).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('event_metadata', 'additional_files_sha256')}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--directory', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); run(a.directory, a.output)
