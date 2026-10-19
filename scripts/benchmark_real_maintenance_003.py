"""Registered first-seed forward CPU limit; no DRAM/FLOP or production-SLA claim."""
import argparse
import json
from pathlib import Path
import time
import torch
from run_real_maintenance_003 import make
from data_real_maintenance_003 import write


@torch.no_grad()
def run(directory,output):
    torch.set_num_threads(1);root=Path(directory)
    assert json.loads((root/'summary.json').read_text())['status']=='COMPLETE'
    data=torch.load(root/'_datasets/validation.pt',weights_only=True)
    normalization=torch.load(root/'_datasets/normalization.pt',weights_only=True)
    measurements=[]
    for path in sorted((root/'group-97301').glob('*/report.json')):
        report=json.loads(path.read_text());model=make(report['spec'],97301,normalization).double().eval()
        model.load_state_dict(torch.load(path.parent/'agent.pt',weights_only=True))
        for batch in (1,16):
            x=data['x'][:batch];minutes=data['minutes'][:batch]
            for _ in range(5):model(x,minutes)
            cpu=[];wall=[]
            for _ in range(100):
                c,w=time.process_time_ns(),time.perf_counter_ns();model(x,minutes)
                cpu.append((time.process_time_ns()-c)/1e6);wall.append((time.perf_counter_ns()-w)/1e6)
            measurements.append(dict(arm=report['arm'],seed=97301,batch=batch,dtype='float64',
                median_cpu_ms=float(torch.tensor(cpu).median()),median_wall_ms=float(torch.tensor(wall).median()),
                parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
                buffer_bytes=sum(p.numel()*p.element_size() for p in model.buffers()),
                context_tensor_bytes=x.numel()*x.element_size()+minutes.numel()*minutes.element_size()))
    selection=json.loads((root/'selection.json').read_text())
    candidate=next(v for v in measurements if v['arm']==selection['candidate'] and v['batch']==1)
    write(output,dict(measurements=measurements,candidate_batch1_cpu10ms_pass=candidate['median_cpu_ms']<=10,
        source='scripts/benchmark_real_maintenance_003.py',raw_temperature_unit='degF',
        scope='One host, first predeclared initialization,5warmups100forwardrepetitions,preconstructed public contexts. Batch times perbatch. CPU/wall exclude JSON/startup/dataacquisition. Tensor bytes exclude allocator/temporaries and are not physical DRAM traffic. Not an energy/FLOP/production or universal-host claim.'))
    print(json.dumps(candidate),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--directory',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.directory,args.output)
