"""Hardware-recorded trained reference timing; no fitting or target scoring."""
import argparse
import importlib.util
import json
from pathlib import Path
import platform
import sys
import torch

p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
ROOT=Path(__file__).resolve().parents[3]
s=importlib.util.spec_from_file_location('lm100_reference_runner',ROOT/'research/mathematical-engineering/langmath100/run.py')
R=importlib.util.module_from_spec(s);sys.modules[s.name]=R;s.loader.exec_module(R)
assert R.sha(a.checkpoint)==R.CHECKPOINTS[9901]
M=R.load_source(ROOT/'research/mathematical-engineering/langtrain099/model.py')
saved=torch.load(a.checkpoint,weights_only=True);m=M.ScaledContextualAssociativeTAC099().eval();m.load_state_dict(saved['model']);before=R.weight_hash(m)
assert before==saved['metadata']['model_sha256']
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
a.output.mkdir(parents=True,exist_ok=True)
with torch.inference_mode():R.timing(M,m,R.read(a.data/'DEV_ACTOR.json'),a.output)
assert R.weight_hash(m)==before
cpu=next((s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name')),'UNKNOWN')
R.save(a.output/'HARDWARE.json',{'cpu':cpu,'platform':platform.platform(),'torch':torch.__version__,'threads':torch.get_num_threads(),'dtype':'FP32','batch':1,'checkpoint_sha256':R.sha(a.checkpoint),'model_source_sha256':R.SOURCE_SHA,'optimizer_steps':0,'energy':'NOT_MEASURED','source_workload':'constructed timing fixture; not learned capability score'})
print(json.dumps({'status':'PASS_TRAINED_REFERENCE_BENCHMARK','cpu':cpu,'optimizer_steps':0}))
