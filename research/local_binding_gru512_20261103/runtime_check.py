"""Package and framework identity check only; no task model or optimizer."""
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
pin=json.loads((ROOT/'RUNTIME_PIN.json').read_bytes())
actual=[[name,importlib.metadata.version(name)] for name,_ in pin['packages']]
if actual!=pin['packages']:raise RuntimeError('Pinned package versions differ')
if sys.version_info[:3]!=(3,12,14):raise RuntimeError('Pinned Python patch differs')
import torch
if torch.__version__!='2.13.0+cpu' or torch.version.cuda is not None:
    raise RuntimeError('Pinned CPU framework differs')
receipt={'status':'PINNED_VERSIONS_PASS_NO_TASK_MODEL_EXECUTION','python':sys.version,
         'executable':sys.executable,'packages':actual,'torch':torch.__version__,
         'cuda':torch.version.cuda,'explicit_task_models_created':0,'optimizer_updates':0,
         'platform_reconstruction':'Source/reference checks are required; exact package versions do not prove identical arithmetic on every host.'}
(ROOT/'PREFLIGHT_RUNTIME.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
