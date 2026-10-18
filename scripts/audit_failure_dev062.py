"""Keep strict replay failures reviewable without changing their acceptance rule."""
import json
from pathlib import Path
import sys
import traceback
import torch
from run_learned_stack_dev055 import write


def preserve(output):
    path=Path(output)
    try:data=json.loads(path.read_text()) if path.exists() else {}
    except (ValueError,OSError):data={}
    data.update(status='FAILED',failure=traceback.format_exc(),host={'cpu_models':sorted({line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')}),'python':sys.version,'torch':str(torch.__version__)})
    write(path,data)
