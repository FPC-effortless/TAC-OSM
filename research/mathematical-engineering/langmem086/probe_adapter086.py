"""Scoped loader for unchanged086 adapter; avoids other studies' cached bare name."""
import importlib.util
from pathlib import Path
_spec=importlib.util.spec_from_file_location('tac_langmem086_original_probe_adapter',Path(__file__).with_name('probe_adapter.py'))
_module=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
adapter086=_module.adapter086
