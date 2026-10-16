"""Reject substituted frozen checkpoints and incomplete evaluation grids."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
s = importlib.util.spec_from_file_location('dev025audit', ROOT/'scripts/audit_dev025_cross_eval.py')
a = importlib.util.module_from_spec(s)
s.loader.exec_module(a)


@pytest.mark.parametrize('tamper', ['checkpoint','report','cohort'])
def test_bound_inputs_reject_substitution(tmp_path,monkeypatch,tamper):
    original=tmp_path/'original'
    original.mkdir()
    checkpoint=original/'model.pt'
    checkpoint.write_bytes(b'bound original checkpoint')
    record={'study':'DEV-024','regime':'long','seed':43,'arm':'joint','smoke':False,
            'training':{'final_hash':'original-parameters'},'model':{'sha256':a.digest(checkpoint)},'source_sha256':{}}
    report=original/'report.json'
    report.write_text(json.dumps(record))
    (tmp_path/'research/autoresearch').mkdir(parents=True)
    manifest={'DEV024-long-43':{'report_sha256':a.digest(report),'checkpoint_sha256':a.digest(checkpoint),'parameter_hash':'original-parameters'}}
    (tmp_path/a.MANIFEST).write_text(json.dumps(manifest))
    monkeypatch.setattr(a,'ROOT',tmp_path)
    a.inputs(original,'DEV024-long-43')
    if tamper=='checkpoint':checkpoint.write_bytes(b'different checkpoint')
    elif tamper=='report':report.write_text(report.read_text()+' ')
    else:
        record['study']='DEV-023'
        report.write_text(json.dumps(record))
        manifest['DEV024-long-43']['report_sha256']=a.digest(report)
        (tmp_path/a.MANIFEST).write_text(json.dumps(manifest))
    with pytest.raises(AssertionError):a.inputs(original,'DEV024-long-43')


def test_audit_requires_all_forty_crossed_cells():
    with pytest.raises(AssertionError):a.summarize([])
