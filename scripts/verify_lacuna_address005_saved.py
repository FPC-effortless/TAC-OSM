#!/usr/bin/env python3
"""Independent stdlib-only source/cohort/norm/decision audit. Zero model calls."""
import hashlib,io,json,math,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def norm(v):return math.sqrt(sum(x*x for x in v))
def add(a,b):return [x+y for x,y in zip(a,b)]
def balanced(rows,values,owners):
    docs={o:sum(v for r,v in zip(rows,values) if r['owner']==o)/sum(r['owner']==o for r in rows) for o in owners if any(r['owner']==o for r in rows)}
    pubs={p:sum(v for o,v in docs.items() if o.split(':')[0]==p)/sum(o.split(':')[0]==p for o in docs) for p in ('cpython','rust-book') if any(o.split(':')[0]==p for o in docs)}
    seeds={}
    for s in (1701,1702,1703):
        means=[sum(v for r,v in zip(rows,values) if r['owner']==o and r['seed']==s)/sum(r['owner']==o and r['seed']==s for r in rows) for o in owners if any(r['owner']==o and r['seed']==s for r in rows)]
        if means:seeds[str(s)]=sum(means)/len(means)
    return {'macro':sum(docs.values())/len(docs) if docs else None,'owners':docs,'publishers':pubs,'seeds':seeds}
def same(actual,expected):
    if isinstance(actual,dict):
        assert set(actual)==set(expected)
        for k in actual:same(actual[k],expected[k])
    elif isinstance(actual,(int,float)) and isinstance(expected,(int,float)):
        assert math.isfinite(actual) and abs(actual-expected)<1e-12
    else:assert actual==expected

def verify(folder):
    receipt=json.loads((folder/'ARTIFACT_RECEIPT.json').read_text());raw=(folder/'ORIGINAL_ARTIFACT.zip').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==receipt['zip_sha256'],'original ZIP SHA'
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:result=json.loads(archive.read('RESULTS.json'))
    summary={k:v for k,v in result.items() if k not in ('rows','all_query_rows')}
    same(summary,json.loads((folder/'SUMMARY.json').read_text()))
    source=json.loads((ROOT/'research/mathematical-engineering/lacuna_address005/SOURCE_RECEIPT.json').read_text());docs={}
    for d in source['sources']:
        data=(ROOT/d['local_path']).read_bytes();assert hashlib.sha256(data).hexdigest()==d['sha256']
        assert hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()==d['git_blob']
        if d['role']=='NEW_DEV_DOCUMENT':docs[d['publisher']+':'+d['path']]=data
    owners=sorted(docs);rows=result['rows'];allrows=result['all_query_rows']
    expected=set()
    for o,data in docs.items():
        for t in range(256,4096,4):
            for lo,hi in ((8,32),(32,128),(128,240)):
                matches=[j for j in range(t-243,t) if lo<=t-j<hi and data[j-4:j]==data[t-3:t+1]]
                if matches:
                    for s in (1701,1702,1703):expected.add((s,o,t,lo,max(matches)))
    observed={(r['seed'],r['owner'],r['clock'],r['bin_begin'],r['historical_clock']) for r in rows}
    assert observed==expected and len(observed)==len(rows),'prefix-only complete cohort'
    for r in rows:
        data=docs[r['owner']];t,j=r['clock'],r['historical_clock'];context=data[t-3:t+1]
        assert r['target_byte']==data[j] and r['next_byte']==data[t+1]
        for label in ('wrong_clock','irrelevant_clock'):
            b=r[label]
            if b is None:continue
            assert t-243<=b<t and abs(b-j)>=5 and not b-4<=j<b and data[b-4:b]!=context
            if label=='irrelevant_clock':assert r['bin_begin']<=t-b<r['bin_end_exclusive']
        g=r['geometry'];h,c,f=g['history'],g['current'],g['fast'];a,b=g['mix']
        target_mix=[a*x+b*(y+z) for x,y,z in zip(f,h,c)]
        for arm,v in r['arms'].items():
            out=v['history_out'];total=v['slow_native'];mixed=v['mixed_native']
            assert abs(norm(out)-norm(h))<=1e-10
            assert abs(norm(add(out,c))-norm(add(h,c)))<=1e-10
            assert abs(norm([a*x+b*(y+z) for x,y,z in zip(f,out,c)])-norm(target_mix))<=1e-10
            assert abs(norm([x-y for x,y in zip(total,c)])-norm(h))<=3e-6
            assert abs(norm(total)-norm(add(h,c)))<=3e-6
            assert abs(norm(mixed)-norm(g['original_mixed_native']))<=3e-6
            assert math.isfinite(v['nll']) and v['nll']>=0
    qkeys={(r['seed'],r['owner'],r['clock']) for r in allrows}
    assert len(qkeys)==len(allrows) and qkeys=={(s,o,t) for s in (1701,1702,1703) for o in owners for t in range(256,4096,4)}
    for r in allrows:
        data=docs[r['owner']];t=r['clock'];assert r['next_byte']==data[t+1]
        for index,hit in [('query_index','query_context_hit'),('random_index','random_context_hit')]:
            j=r[index];assert t-243<=j<t;assert r[hit]==int(data[j-4:j]==data[t-3:t+1])
    for b in result['summary']['bins']:
        rr=[r for r in rows if r['bin_begin']==b['bin_begin']]
        counts={o:len({r['clock'] for r in rr if r['owner']==o}) for o in owners};assert counts==b['source_support'] and len(rr)==b['measurement_count']
        for arm,reported in b['arms'].items():
            same(balanced(rr,[r['original_nll']-r['arms'][arm]['nll'] for r in rr],owners),reported['original_minus_arm'])
            assert reported['fallback_or_missing']==sum(r['arms'][arm]['status']!='REGULAR' for r in rr)
        for arm,wrong,irrelevant in [('B','matrix_wrong','matrix_irrelevant'),('D','raw_wrong','raw_irrelevant')]:
            comparisons={}
            for label,control in [('wrong',wrong),('shuffled',arm+'_shuffle'),('irrelevant',irrelevant),('random_mean',None)]:
                values=[(r['arms'][control]['nll'] if control else .5*(r['arms']['random0']['nll']+r['arms']['random1']['nll']))-r['arms'][arm]['nll'] for r in rr]
                comparisons[label]=balanced(rr,values,owners)
            same(comparisons,b['gates'][arm]['comparisons'])
            d=comparisons['wrong'];valid=all(n>=20 for n in counts.values()) and not any(b['arms'][v]['fallback_or_missing'] for v in (arm,wrong,irrelevant,arm+'_shuffle','random0','random1'))
            meaningful=d['macro'] is not None and d['macro']>=.001 and len(d['owners'])==6 and min(d['owners'].values())>0 and len(d['seeds'])==3 and min(d['seeds'].values())>0 and len(d['publishers'])==2 and min(d['publishers'].values())>=.001
            additional=all(comparisons[k]['macro'] is not None and comparisons[k]['macro']>=.001 and len(comparisons[k]['publishers'])==2 and min(comparisons[k]['publishers'].values())>0 for k in ('shuffled','irrelevant','random_mean'))
            gate='INCONCLUSIVE_SUPPORT_OR_CONTROL' if not valid else ('PASS_FRESH_CONDITIONAL_SPECIFICITY' if meaningful and additional else 'NOT_PASSED')
            assert gate==b['gates'][arm]['specificity']
            u=b['arms'][arm]['original_minus_arm'];practical=u['macro'] is not None and u['macro']>=.001 and len(u['owners'])==6 and min(u['owners'].values())>0 and len(u['seeds'])==3 and min(u['seeds'].values())>0
            assert bool(practical)==b['gates'][arm]['practical_prediction_utility']
    hits=balanced(allrows,[r['query_context_hit']-r['random_context_hit'] for r in allrows],owners);same(hits,result['summary']['query_selection']['hit_advantage'])
    select=hits['macro'] is not None and hits['macro']>=.01 and len(hits['owners'])==6 and min(hits['owners'].values())>0 and len(hits['seeds'])==3 and min(hits['seeds'].values())>0
    assert bool(select)==result['summary']['query_selection']['passed']
    native=[]
    for label in ('original','random_mean'):
        delta=[(r['original_nll'] if label=='original' else .5*(r['scores']['random0']+r['scores']['random1']))-r['scores']['C'] for r in allrows]
        d=balanced(allrows,delta,owners);same(d,result['summary']['all_query_native']['comparisons'][label])
        native.append(d['macro'] is not None and d['macro']>=.001 and len(d['owners'])==6 and min(d['owners'].values())>0 and len(d['publishers'])==2 and min(d['publishers'].values())>0 and len(d['seeds'])==3 and min(d['seeds'].values())>0)
    passed=all(native) and all(all(r['statuses'][a]=='REGULAR' for a in ('C','random0','random1')) for r in allrows)
    assert passed==result['summary']['all_query_native']['passed']
    assert bool(select and passed)==result['summary']['autonomous_conditional_gate']
    print(json.dumps({'status':'PASS_ORIGINAL_SOURCE_COHORT_THREE_NORMS_BALANCED_GATES','measurements':len(rows),'all_query_measurements':len(allrows),'model_executions':0}))

if __name__=='__main__':verify(Path(sys.argv[1]))
