"""Scoped import of original038 adapter, avoiding unrelated cached probe_adapter."""
import importlib.util
from pathlib import Path
_spec=importlib.util.spec_from_file_location('tac_langlayer038_original_probe_adapter',Path(__file__).with_name('probe_adapter.py'))
_module=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
UPPER_PROBE=_module.UPPER_PROBE
