"""Read public FHWA maintenance/cost/condition tables; audit coverage, not effects.

Uses the normal InfoPave table-preview API without credentials or SQL queries.
No fitting, response-based sample selection, outcome scoring or causal claims.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE = 'https://infopave.fhwa.dot.gov'
PROFILE = ['STATE_CODE', 'SHRP_ID', 'CONSTRUCTION_NO', 'VISIT_DATE',
           'VISIT_NO', 'RUN_NUMBER', 'MRI', 'IRI_LEFT_WHEEL_PATH',
           'IRI_RIGHT_WHEEL_PATH', 'RECORD_STATUS']


def post(endpoint, payload, json_body=False):
    body = json.dumps(payload).encode() if json_body else urlencode(payload).encode()
    request = Request(BASE + endpoint, data=body, headers={
        'Content-Type': 'application/json; charset=UTF-8' if json_body else
                        'application/x-www-form-urlencoded; charset=UTF-8'})
    with urlopen(request, timeout=45) as response:
        raw = response.read()
    return raw, json.loads(raw)


def acquire(directory):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {}
    tables = {}
    for table, schema in [('MNT_COST', 'LTPPDBA'), ('MNT_IMP', 'LTPPDBA'),
                          ('MON_HSS_PROFILE_SECTION', 'INFOPAVE')]:
        raw, dictionary = post('/Data/GetTableDictionaryInformation', {'tableName': table}, True)
        assert dictionary['OK'] is True
        schema_path = root / (table + '-schema.json')
        schema_path.write_bytes(raw)
        manifest[schema_path.name] = hashlib.sha256(raw).hexdigest()
        available = [v['FIELD_NAME'] for v in dictionary['tableDictionary']]
        fields = PROFILE if table == 'MON_HSS_PROFILE_SECTION' else available
        assert set(fields) <= set(available)
        rows, page, total = [], 1, None
        while total is None or len(rows) < total:
            args = dict(page=page, rows=10000, sidx='STATE_CODE', sord='asc',
                        totalRecords=0, selectedFilter='{}', schemaName=schema,
                        tableName=table, fields=','.join(fields),
                        sqlExportQuery='', moduleName='tableexport')
            raw, response = post('/Data/GetPreviewData', args)
            assert response['Message'] == 'success', response.get('Message')
            if total is None:
                total = int(response['records'])
            assert int(response['records']) == total, 'Source changed during paging'
            assert response['rows']
            filename = f'{table}-page{page:03d}.json'
            (root / filename).write_bytes(raw)
            manifest[filename] = hashlib.sha256(raw).hexdigest()
            rows.extend(dict(zip(fields, r['cell'], strict=True)) for r in response['rows'])
            page += 1
            assert page <= 30, 'Bounded public-source acquisition, not unlimited crawling'
        assert len(rows) == total
        tables[table] = rows
        print(table, total, flush=True)
    cost = tables['MNT_COST']
    positive = [r for r in cost if any(float(v) > 0 for k, v in r.items()
                                     if k.endswith('_TOT_COST') and v.strip())]
    profiles = defaultdict(list)
    for row in tables['MON_HSS_PROFILE_SECTION']:
        if row['MRI'].strip():
            profiles[(row['STATE_CODE'], row['SHRP_ID'])].append(row)
    matched = []
    for row in positive:
        date = datetime.strptime(row['DATE_COMPLETE'], '%m/%d/%Y')
        visits = profiles.get((row['STATE_CODE'], row['SHRP_ID']), [])
        before = {v['VISIT_DATE'] for v in visits
                  if datetime.strptime(v['VISIT_DATE'], '%m/%d/%Y') < date}
        after = {v['VISIT_DATE'] for v in visits
                 if datetime.strptime(v['VISIT_DATE'], '%m/%d/%Y') > date}
        if before and after:
            matched.append({'state': row['STATE_CODE'], 'section': row['SHRP_ID'],
                            'treatment_date': row['DATE_COMPLETE'],
                            'before_visit_dates': len(before), 'after_visit_dates': len(after)})
    result = {
        'status': 'SOURCE_COVERAGE_ONLY', 'retrieved_utc': datetime.now(timezone.utc).isoformat(),
        'source': BASE + '/Data/TableExport/', 'files_sha256': manifest,
        'counts': {k: len(v) for k, v in tables.items()},
        'duplicate_selected_field_row_counts': {k: len(rows) - len({json.dumps(row, sort_keys=True) for row in rows}) for k, rows in tables.items()},
        'positive_reported_total_cost_rows': len(positive),
        'positive_total_cost_sections': len({(r['STATE_CODE'], r['SHRP_ID']) for r in positive}),
        'positive_cost_events_with_measured_MRI_before_and_after': len(matched),
        'matched_sections': len({(r['state'], r['section']) for r in matched}),
        'matched_event_coverage': matched,
        'record_status_counts': {k: dict(Counter(r['RECORD_STATUS'] for r in rows))
                                 for k, rows in tables.items()},
        'limits': 'Road pavement, not factory equipment. MRI is measured condition, not binary repair success. Source reports costs in historical dollars; missing costs are not zero. No cost imputation, inflation conversion, inspection price, treatment-effect estimate, trained model or sealed performance evaluation. Table quality-code and reuse-terms review remain required before a registered modeling study. Whole table acquired for coverage only; no condition-value improvements calculated.',
    }
    (root / 'source-audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print({k: result[k] for k in ('positive_reported_total_cost_rows',
                                 'positive_cost_events_with_measured_MRI_before_and_after',
                                 'matched_sections')}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', required=True)
    acquire(parser.parse_args().directory)
