#!/usr/bin/env python3
"""Independent original-ZIP, paired cohort, norm and gate audit; zero inference."""
import hashlib,io,json,math,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def norm(v):return math.sqrt(sum(x*x for x in v))
def add(a,b):return [x+y for x,y in zip(a,b)]

def verify(folder):
    receipt=json.loads((folder/'ARTIFACT_RECEIPT.json').read_text());raw=(folder/'ORIGINAL_ARTIFACT.zip').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==receipt['zip_sha256'],'original ZIP'
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:result=json.loads(archive.read('RESULTS.json'))
    if (folder/'RESULTS.json').exists():
        assert result==json.loads((folder/'RESULTS.json').read_text()),'original output modified'
    summary={k:v for k,v in result.items() if k!='rows'}
    assert summary==json.loads((folder/'SUMMARY.json').read_text()),'saved summary modified'
    previous=json.loads((ROOT/'research/evidence/lacuna_retrieval003/RESULTS.json').read_text())
    old={(r['seed'],r['owner'],r['clock'],r['bin_begin']):r for r in previous['rows']}
    rows=result['rows'];seen=set()
    for row in rows:
        key=row['seed'],row['owner'],row['clock'],row['bin_begin'];seen.add(key);prior=old[key]
        assert all(row[f]==prior[f] for f in ('historical_clock','wrong_clock','target_byte','next_byte')),'paired source drift'
        assert abs(row['original_nll']-prior['original_nll'])<=3e-5
        for arm,methods in row['arms'].items():
            assert abs(methods['historical_only']['next_byte_nll']-prior['arms'][arm]['next_byte_nll'])<=3e-5
            for method,item in methods.items():
                assert abs(row['original_nll']-item['next_byte_nll']-item['original_minus_arm_nll'])<1e-12
                if method!='dual_norm':continue
                vec=item['vectors'];h,c,out=vec['history'],vec['current'],vec['transformed_history']
                assert abs(norm(h)-norm(out))<=1e-10,'historical norm'
                assert abs(norm(add(h,c))-norm(add(out,c)))<=1e-10,'total slow norm'
                assert abs(sum(a*b for a,b in zip(h,c))-sum(a*b for a,b in zip(out,c)))<=1e-10,'projection'
                effective=[a-b for a,b in zip(vec['native_total'],c)]
                assert abs(norm(effective)-norm(h))<=3e-6,'native historical norm'
                assert abs(norm(vec['native_total'])-norm(add(h,c)))<=3e-6,'native total norm'
                if not item['status'].startswith('REGULAR'):assert out==h,'fallback must preserve original direction'
    assert seen==set(old),'original cohort not preserved'
    for summary in result['summary']:
        rr=[r for r in rows if r['bin_begin']==summary['bin_begin']]
        assert len(rr)==summary['measurements']
        cells={(r['owner'],r['clock']) for r in rr};assert len(cells)==summary['distinct_source_query_cells']
        for arm,methods in summary['arms'].items():
            for method in ('historical_only','dual_norm'):
                for field,value in methods[method].items():
                    if field=='fallback_count':
                        actual=sum(not r['arms'][arm][method]['status'].startswith('REGULAR') for r in rr) if method=='dual_norm' else 0
                    else:actual=sum(r['arms'][arm][method][field] for r in rr)/len(rr)
                    assert actual==value,(arm,method,field)
        positives=[]
        for arm in ('A','B','C','D'):
            wrong='matrix_wrong' if arm in ('A','B') else 'raw_wrong'
            fallback=any(summary['arms'][a]['dual_norm']['fallback_count'] for a in (arm,wrong,arm+'_shuffle'))
            means=[sum(r['arms'][arm]['dual_norm']['original_minus_arm_nll'] for r in rr if r[f]==v)/sum(r[f]==v for r in rr)
                   for f,vs in [('seed',(1701,1702,1703)),('owner',sorted({r['owner'] for r in rows}))]
                   for v in vs if any(r[f]==v for r in rr)]
            gain=summary['arms'][arm]['dual_norm']['original_minus_arm_nll']
            if not fallback and len(cells)>=20 and len(means)==5 and min(means)>0 and gain>=.001 and all(gain>summary['arms'][a]['dual_norm']['original_minus_arm_nll'] for a in (wrong,arm+'_shuffle')):positives.append(arm)
        assert positives==summary['provisional_diagnostic_signals'],'gate recomputation'
    print(json.dumps({'status':'PASS_ORIGINAL_PAIRED_COHORT_NORMS_SCORES_GATES','rows':len(rows),'model_executions':0,'fresh_evaluation':False}))

if __name__=='__main__':verify(Path(sys.argv[1]))
