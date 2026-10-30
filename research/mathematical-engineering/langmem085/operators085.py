"""Isolated085 import of the unchanged original operator source.

084 and085 both historically used the bare module name `operators`; Python's
process-wide cache ignores sys.path changes after the first import. This
bridge executes the original085 file under a study-specific name, preserving
all equations/constants and source bytes while allowing both test suites.
"""
import importlib.util
from pathlib import Path
import sys

_NAME='tac_research_langmem085_original_operators'
if _NAME not in sys.modules:
    _spec=importlib.util.spec_from_file_location(_NAME,Path(__file__).with_name('operators.py'))
    _module=importlib.util.module_from_spec(_spec)
    sys.modules[_NAME]=_module
    _spec.loader.exec_module(_module)
_module=sys.modules[_NAME]
ARMS=_module.ARMS
AGES=_module.AGES
write=_module.write
prospective_wrong_keys=_module.prospective_wrong_keys
load_frozen_new_rust=_module.load_frozen_new_rust
fixed_fit_prior=_module.fixed_fit_prior
context_key=_module.context_key
byte_book=_module.byte_book
decode=_module.decode
