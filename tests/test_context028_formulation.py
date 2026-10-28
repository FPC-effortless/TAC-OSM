"""Deliberate shape/coverage failures; constructed software controls only."""
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('context028',Path(__file__).resolve().parents[1]/'scripts/check_context028_formulation.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def test_lean_shape_parser_and_wrong_tensor_rejection():
    text='structure Weights where\n  upW : Mat 501 280\n  angle : Vec 8\n  rotation : Fin 2 → Mat 8 8\n\nstructure State where'
    fields=audit.lean_shapes(text)
    assert fields=={'upW':[501,280],'angle':[8],'rotation':[2,8,8]}
    mapping={'upW':'ffn.weight','angle':'angle','rotation':'rotation'}
    actual={'ffn.weight':[501,280],'angle':[8],'rotation':[2,8,8]}
    audit.check_shapes(fields,mapping,actual)
    with pytest.raises(ValueError,match='shape differs'):
        audit.check_shapes({**fields,'upW':[500,280]},mapping,actual)


def test_missing_duplicate_or_unknown_tensor_rejected():
    with pytest.raises(ValueError,match='coverage differs'):
        audit.check_shapes({'a':[8]},{'a':'a'},{'a':[8],'hidden':[9]})
    with pytest.raises(ValueError,match='duplicate Lean weight'):
        audit.lean_shapes('structure Weights where\n  a : Vec 8\n  a : Vec 9\n\nstructure State where')
    with pytest.raises(ValueError,match='unknown Lean learned tensor'):
        audit.lean_shapes('structure Weights where\n  a : HiddenEncoder\n\nstructure State where')
