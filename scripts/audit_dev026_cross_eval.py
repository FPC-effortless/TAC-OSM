"""DEV-026 stdlib world/cost audit and complete-diagonal gate."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

import audit_dev025_cross_eval as prior
from audit_dev024_writer_attribution import validate_rows, metrics, spec_and_sources
from dev026_replay_numerics import compare_diagonal, REL_TOL, ABS_TOL, FLOAT32_FIELDS

ROOT, digest, inputs = prior.ROOT, prior.digest, prior.inputs
PROTOCOL = 'development_protocols/DEV-026-FROZEN-CROSS-EVAL.json'
SOURCES = (PROTOCOL, 'docs/DEV-026-FROZEN-CROSS-EVAL-PROTOCOL.md',
           'scripts/run_dev026_cross_eval.py', 'scripts/audit_dev026_cross_eval.py',
           'scripts/dev026_replay_numerics.py')


def config():
    c=json.loads((ROOT/PROTOCOL).read_text())
    old=prior.config()
    for key in ('seeds','regimes','cohorts','evaluation_offsets','history_levels','eval_steps','training_steps','expected_cells','decision'):
        assert c[key]==old[key], 'scientific protocol drift'
    assert c['numerical_replay']['relative_tolerance']==REL_TOL
    assert c['numerical_replay']['absolute_tolerance']==ABS_TOL
    assert set(c['numerical_replay']['allowed_fields'])==FLOAT32_FIELDS
    return c


def common_sources():
    return {p:digest(ROOT/p) for p in (*SOURCES,*prior.SOURCES,'scripts/audit_dev024_writer_attribution.py','contracts/DEV-024.json')}


def source_hashes(parent):
    result=prior.source_hashes(parent)
    result.update(common_sources())
    return result


def diagonal_evidence(rows, archived):
    stats={'differing_float_leaves':0,'max_absolute_float_difference':0.,'max_difference_path':None}
    def scan(a,b,path=''):
        assert type(a) is type(b), 'type changed'
        if isinstance(a,dict):
            assert a.keys()==b.keys(), 'fields changed'
            for k in a:scan(a[k],b[k],path+'/'+str(k))
        elif isinstance(a,list):
            assert len(a)==len(b), 'length changed'
            for k,(x,y) in enumerate(zip(a,b)):scan(x,y,path+'/'+str(k))
        elif isinstance(a,float):
            assert math.isfinite(a) and math.isfinite(b), 'nonfinite value'
            if a!=b:
                stats['differing_float_leaves']+=1
                if abs(a-b)>stats['max_absolute_float_difference']:
                    stats.update(max_absolute_float_difference=abs(a-b),max_difference_path=path)
    try:
        scan(rows,archived)
        fresh,old=metrics(rows),metrics(archived)
        stats['metric_differences']={k:[fresh[k],old[k]] for k in fresh if fresh[k]!=old[k]}
        compare_diagonal(rows,archived)
        stats.update(validated=True,error=None)
    except AssertionError as e:
        stats.update(validated=False,error=str(e))
    return stats


def expected_models():
    c=config()
    return {f'{a}-{g}-{s}' for a in c['cohorts'] for g in c['regimes'] for s in c['seeds']}


def validate_certificate(cert):
    assert cert['study']=='DEV-026-DIAGONAL-AUDIT' and cert['passed'] is True
    assert cert['cells']==config()['expected_diagonals']==20
    assert set(cert['diagonal_report_sha256'])==expected_models()
    assert cert['common_source_sha256']==common_sources()
    assert cert['input_manifest_sha256']==digest(ROOT/prior.MANIFEST)


def audit(folder, certificate=None):
    path=folder/'report.json'
    r=json.loads(path.read_text())
    c=config()
    assert r['study']=='DEV-026' and r['training_steps']==0
    assert r['cohort'] in c['cohorts'] and r['regime'] in c['regimes'] and r['seed'] in c['seeds'] and r['grid'] in c['cohorts']
    key=f"{r['cohort']}-{r['regime']}-{r['seed']}"
    parent,frozen=inputs(folder.parent/'original',key)
    assert r['frozen_input']==frozen and r['source_sha256']==source_hashes(parent)
    assert r['evaluation_offset']==c['evaluation_offsets'][r['grid']]
    assert json.loads((folder/'replay.json').read_text())==dict(verified=True,report_sha256=digest(path),checkpoint_sha256=frozen['checkpoint_sha256'],source_sha256=r['source_sha256'])
    spec=spec_and_sources()
    spec.update(h_levels=[512],eval_steps=4)
    validate_rows(r['rows'],r['seed']+r['evaluation_offset']-24000,'joint',spec)
    assert metrics(r['rows'])==r['summary'] and r['summary']['primary_n']==96 and r['summary']['reset_primary']==.5
    if r['grid']==r['cohort']:
        archived=[x for x in parent['rows'] if x['condition']['history']==512]
        evidence=diagonal_evidence(r['rows'],archived)
        assert evidence['validated'] and r['diagonal_evidence']==evidence
        assert r['diagonal_certificate_sha256'] is None
    else:
        assert certificate is not None
        cert=json.loads(certificate.read_text())
        validate_certificate(cert)
        assert r['diagonal_evidence'] is None
        assert r['diagonal_certificate_sha256']==digest(certificate)
    return r


def certify_diagonals(records,folders):
    assert len(records)==len(folders)==20
    by={f"{r['cohort']}-{r['regime']}-{r['seed']}":(r,p) for r,p in zip(records,folders)}
    assert set(by)==expected_models()
    assert all(r['cohort']==r['grid'] and r['diagonal_evidence']['validated'] for r in records)
    result=dict(study='DEV-026-DIAGONAL-AUDIT',passed=True,cells=20,
                common_source_sha256=common_sources(),input_manifest_sha256=digest(ROOT/prior.MANIFEST),
                diagonal_report_sha256={k:digest(p/'report.json') for k,(r,p) in sorted(by.items())},
                per_diagonal={k:r['diagonal_evidence'] for k,(r,p) in sorted(by.items())})
    validate_certificate(result)
    return result


def summarize(records):
    # Scientific contrasts and physical gates remain the frozen DEV-025 rules.
    result=prior.summarize(records)
    result['study']='DEV-026'
    c=config()
    by={(r['cohort'],r['regime'],r['seed'],r['grid']):r for r in records}
    for regime in c['regimes']:
        delta=[(by['DEV024',regime,s,'DEV024']['summary']['primary']-by['DEV023',regime,s,'DEV024']['summary']['primary'])-
               (by['DEV024',regime,s,'DEV023']['summary']['primary']-by['DEV023',regime,s,'DEV023']['summary']['primary']) for s in c['seeds']]
        result['by_regime'][regime]['cohort_by_grid_interaction']=prior.interval(delta)
    result['numerically_bounded_archived_diagonals']=True
    result['diagonal_evidence']={f"{r['cohort']}-{r['regime']}-{r['seed']}":r['diagonal_evidence'] for r in records if r['cohort']==r['grid']}
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('folders',nargs='+',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--diagonals-only',action='store_true')
    p.add_argument('--certificate',type=Path)
    a=p.parse_args()
    rs=[audit(folder,a.certificate) for folder in a.folders]
    if a.diagonals_only:
        result=certify_diagonals(rs,a.folders)
    else:
        assert a.certificate is not None
        result=summarize(rs)
        cert=json.loads(a.certificate.read_text())
        for r,folder in zip(rs,a.folders):
            if r['cohort']==r['grid']:
                key=f"{r['cohort']}-{r['regime']}-{r['seed']}"
                assert cert['diagonal_report_sha256'][key]==digest(folder/'report.json')
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print('DEV026_DIAGONALS_AUDITED' if a.diagonals_only else 'DEV026_COMPLETE_AUDIT_PASSED',len(rs),flush=True)
