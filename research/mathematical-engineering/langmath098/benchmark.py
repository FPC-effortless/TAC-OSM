"""Registered constructed CPU timings; no dataset, fits or learned scores."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m

def run():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError('retain original timing receipt')
    torch.set_num_threads(1);torch.manual_seed(98002)
    old=load('langmath098_baseline086',ROOT/'research/mathematical-engineering/langmem086/compact_tac.py').CompactDualClippedTAC086().eval()
    new=load('langmath098_benchmark_model',Path(__file__).with_name('model.py')).ContextualAssociativeTAC098().eval()
    rows=[]
    with torch.inference_mode():
        for name,m in [('original086',old),('candidate098',new)]:
            for kind in ('prefill','stream_step'):
                if kind=='prefill':
                    tokens=(torch.arange(4*128).reshape(4,128)%256);owners=('a','b','c','d');state=None
                else:
                    owners=('a',);_,state=m(torch.arange(64)[None],owners=owners);tokens=torch.tensor([[64]])
                for _ in range(5):m(tokens,state=state,owners=owners)
                values=[]
                for _ in range(30):
                    started=time.perf_counter_ns();m(tokens,state=state,owners=owners)
                    values.append((time.perf_counter_ns()-started)/1e6)
                rows.append({'model':name,'workload':kind,'batch':tokens.shape[0],'tokens_per_stream':tokens.shape[1],
                    'counts':m.counts(),'persistent_state_bytes_per_stream':m.start(('x',)).tensor_bytes(),
                    'p50_ms':float(np.percentile(values,50)),'p95_ms':float(np.percentile(values,95)),
                    'all_samples_ms':values})
    cpu='unknown'
    info=Path('/proc/cpuinfo')
    if info.is_file():
        cpu=next((s.split(':',1)[1].strip() for s in info.read_text().splitlines() if s.startswith('model name')),cpu)
    receipt={'schema':'langmath098.constructed-no-fit-timing.v1','cpu':cpu,'platform':platform.platform(),
        'torch':torch.__version__,'threads':1,'dtype':'FP32','warmups':5,'samples':30,'rows':rows,
        'model_source_sha256':hashlib.sha256(Path(__file__).with_name('model.py').read_bytes()).hexdigest(),
        'optimizer_steps':0,'actual_dataset_access':False,'trained_weights_loaded':False,
        'scope':'Constructed engineering timings with untrained models; no equal-FLOP, architecture-superiority, useful prediction, deployment cost or global-optimality qualification.'}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'output':str(args.output),'cpu':cpu,'rows':[{k:r[k] for k in ('model','workload','p50_ms','p95_ms','persistent_state_bytes_per_stream')} for r in rows]}))

if __name__=='__main__':run()
