#!/usr/bin/env python3
"""Independent stdlib-only source/cohort and saved-summary verification; no inference."""
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ('past_byte_nll','past_byte_correct','next_byte_nll','original_minus_arm_nll')


def verify(folder):
    receipt = json.loads((folder/'ARTIFACT_RECEIPT.json').read_text())
    payload = (folder/'ORIGINAL_ARTIFACT.zip').read_bytes()
    assert hashlib.sha256(payload).hexdigest() == receipt['zip_sha256'], 'original ZIP'
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        result = json.loads(archive.read('RESULTS.json'))
    assert result == json.loads((folder/'RESULTS.json').read_text()), 'original result altered'
    source_receipt = json.loads((ROOT/'research/mathematical-engineering/lacuna_retrieval003/SOURCE_RECEIPT.json').read_text())
    docs = {}
    for source in source_receipt['documents']:
        raw = (ROOT/'research/source-bundles/LACUNA-RETRIEVAL003'/source['path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == source['sha256']
        if source['path'] != 'LICENSE':
            docs[source['path']] = raw
    rows = [row for row in result['rows'] if row['status'] == 'MEASURED']
    predictions = {}
    for row in rows:
        raw = docs[row['owner']]
        t,j,b = row['clock'],row['historical_clock'],row['wrong_clock']
        candidates = range(max(4,t+1-248+4),t)
        matches = [i for i in candidates if row['bin_begin'] <= t-i < row['bin_end_exclusive'] and raw[i-4:i] == raw[t-3:t+1]]
        assert j == max(matches), 'prefix-only most recent match'
        assert b in candidates and abs(b-j)>=5 and not b-4<=j<b and raw[b-4:b]!=raw[j-4:j], 'wrong-address exclusion'
        assert row['target_byte'] == raw[j] and row['next_byte'] == raw[t+1]
        assert row['historical_continuation_matches_next_byte'] == int(raw[j] == raw[t+1])
        for arm in row['arms'].values():
            assert abs(row['original_nll']-arm['next_byte_nll']-arm['original_minus_arm_nll']) < 1e-12
        # Same current context gives the same supplied key across age bins.
        # All four main predictions must therefore be independent of selected j.
        key = row['seed'],row['owner'],t
        main = [row['original_nll']]+[row['arms'][a]['next_byte_nll'] for a in ('A','B','C','D')]
        if key in predictions:
            assert main == predictions[key], 'historical-target selection changed current prediction'
        predictions[key] = main
    for summary in result['summary']:
        rr = [r for r in rows if r['bin_begin']==summary['bin_begin']]
        cells = {(r['owner'],r['clock']) for r in rr}
        assert len(cells)==summary['distinct_source_query_cells']
        assert len(rr)==summary['measurements']
        for arm,metrics in summary['arms'].items():
            for field in FIELDS:
                mean = sum(r['arms'][arm][field] for r in rr)/len(rr) if rr else None
                assert mean == metrics[field], (arm,field)
        positive = []
        for arm in ('A','B','C','D'):
            wrong = 'matrix_wrong' if arm in ('A','B') else 'raw_wrong'
            means = [sum(r['arms'][arm]['original_minus_arm_nll'] for r in rr if r[field]==value)/sum(r[field]==value for r in rr)
                     for field,values in (('seed',(1701,1702,1703)),('owner',sorted(docs)))
                     for value in values if any(r[field]==value for r in rr)]
            gain = summary['arms'][arm]['original_minus_arm_nll']
            missing = any(r['status']!='MEASURED' for r in result['rows'] if r['bin_begin']==summary['bin_begin'])
            if not missing and len(cells)>=20 and len(means)==5 and min(means)>0 and gain>=.001 and all(gain>summary['arms'][control]['original_minus_arm_nll'] for control in (wrong,arm+'_shuffle')):
                positive.append(arm)
        assert positive==summary['provisional_diagnostic_signals'], 'gate recomputation'
    print(json.dumps({'status':'PASS_ORIGINAL_ZIP_SOURCE_COHORT_AND_SUMMARY_RECOMPUTATION',
                      'measurements':len(rows),'model_executions':0,'independently_qualified':False}))


if __name__ == '__main__':
    verify(Path(sys.argv[1]))
