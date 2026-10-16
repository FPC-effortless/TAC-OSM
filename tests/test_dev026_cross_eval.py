"""The complete diagonal gate and numerical records cannot rescue bad outcomes."""
import copy
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_dev026_cross_eval as a  # noqa: E402


def certificate():
    return dict(study='DEV-026-DIAGONAL-AUDIT',passed=True,cells=20,
                common_source_sha256=a.common_sources(),input_manifest_sha256=a.digest(a.ROOT/a.prior.MANIFEST),
                diagonal_report_sha256={k:'0'*64 for k in a.expected_models()})


@pytest.mark.parametrize('field',['missing_model','manifest','source','count','passed'])
def test_cross_gate_rejects_incomplete_or_substituted_diagonals(field):
    c=certificate()
    a.validate_certificate(c)
    if field=='missing_model':c['diagonal_report_sha256'].pop(next(iter(c['diagonal_report_sha256'])))
    elif field=='manifest':c['input_manifest_sha256']='different'
    elif field=='source':c['common_source_sha256']['scripts/run_dev026_cross_eval.py']='different'
    elif field=='count':c['cells']=19
    else:c['passed']=False
    with pytest.raises(AssertionError):a.validate_certificate(c)


def test_all_twenty_diagonals_are_required():
    with pytest.raises(AssertionError):a.certify_diagonals([],[])


def test_numerical_evidence_counts_rounding_but_rejects_cost_change(monkeypatch):
    # Isolate the numerical comparison; raw-world and metric auditing uses the
    # inherited independent case/physical validator in audit().
    monkeypatch.setattr(a,'metrics',lambda rows:{'primary':1.,'primary_brier':rows[0]['logits'][0][0]})
    old=[dict(logits=[[1.,-1.]],predictions=[0],controls={'selected':{'total_cost':.5}})]
    new=copy.deepcopy(old)
    new[0]['logits'][0][0]+=1e-7
    result=a.diagonal_evidence(new,old)
    assert result['validated'] and result['differing_float_leaves']==1
    assert result['max_difference_path']=='/0/logits/0/0'
    assert 'primary_brier' in result['metric_differences']
    new[0]['controls']['selected']['total_cost']+=1e-10
    assert not a.diagonal_evidence(new,old)['validated']


def test_scientific_decision_rule_cannot_drift(tmp_path,monkeypatch):
    c=a.config()
    c['decision']='accept favorable mean without all seeds'
    p=tmp_path/a.PROTOCOL
    p.parent.mkdir(parents=True)
    import json
    p.write_text(json.dumps(c))
    monkeypatch.setattr(a,'ROOT',tmp_path)
    with pytest.raises(AssertionError):a.config()
