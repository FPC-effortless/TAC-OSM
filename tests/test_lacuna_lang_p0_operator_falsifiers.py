"""Exercise no-fit numerical contracts in a normal pytest discovery pass.

Constructed reference tests only: no source-trained TAC, real evaluation or
proof of full neural graph correctness. See research issue #633.
"""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def test_lacuna_constructed_numeric_operators():
    path = Path(__file__).resolve().parents[1] / "research/scripts/lacuna_lang_p0_operator_falsifiers.py"
    spec = spec_from_file_location("lacuna_lang_p0_operator_falsifiers", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()
