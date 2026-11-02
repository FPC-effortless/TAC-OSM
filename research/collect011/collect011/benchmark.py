"""Paired real-TRAIN-byte no-update graph measurements. CPU is not a speed claim."""
import argparse,hashlib,json,platform,time
from pathlib import Path
import torch
from torch.nn import functional as F
from .model import CollectiveTAC,FFN_DIM as FFN

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--lengths',type=int,nargs='+',default=[32,128,256])
    parser.add_argument('--arms',nargs='+',choices=CollectiveTAC.ARMS,default=list(CollectiveTAC.ARMS))
    parser.add_argument('--output',default='GRAPH_BENCHMARK');args=parser.parse_args()
    torch.set_num_threads(1)
    root=Path(__file__).resolve().parents[1]
    data=(root/"data/train.bin").read_bytes()
    rows=[]
    for seed in (1,2,3):
        for length in args.lengths:
            offset=10000+seed*1000
            sample=torch.tensor(list(data[offset:offset+2*(length+1)]),dtype=torch.long).reshape(2,length+1)
            x,y=sample[:,:-1],sample[:,1:]
            for arm in args.arms:
                torch.manual_seed(seed);m=CollectiveTAC(arm,ffn_dim=FFN[arm])
                before={n:p.detach().clone() for n,p in m.named_parameters()}
                # Single untimed warmup; no optimizer exists in this executable.
                with torch.no_grad():m(x[:,:8])
                times=[];finite=True;active=True;maxgrad=0.
                for repeat in range(3):
                    m.zero_grad(set_to_none=True)
                    start=time.perf_counter();logits,state=m(x)
                    loss=F.cross_entropy(logits.flatten(0,1),y.flatten());loss.backward()
                    times.append(time.perf_counter()-start)
                    finite &= bool(torch.isfinite(logits).all()) and all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in m.parameters())
                    active &= all(p.grad is not None and bool((p.grad!=0).any()) for p in m.parameters())
                    maxgrad=max(maxgrad,max(float(p.grad.abs().max()) for p in m.parameters()))
                assert all(torch.equal(before[n],p) for n,p in m.named_parameters())
                phase_bytes=sum(s.state_bytes() for s in state.phase)//2
                pbytes=sum(s.precision.numel()*s.precision.element_size() for s in state.phase if s.precision is not None)//2
                mineig=min([float(torch.linalg.eigvalsh(s.precision.detach()).min()) for s in state.phase if s.precision is not None] or [float("nan")])
                row=dict(seed=seed,arm=arm,length=length,batch=2,trainable=sum(p.numel() for p in m.parameters()),
                    ffn_dim=FFN[arm],step_seconds=times,median_seconds=sorted(times)[1],
                    tokens_per_second=2*length/sorted(times)[1],state_bytes_per_stream=state.state_bytes()//2,
                    phase_bytes_per_stream=phase_bytes,precision_bytes_per_stream=pbytes,
                    min_precision_eigenvalue=None if pbytes==0 else mineig,all_finite=finite,all_parameter_tensors_active=active,
                    max_abs_gradient=maxgrad,optimizer_updates=0,weights_unchanged=True,
                    input_sha256=hashlib.sha256(bytes(x.flatten().tolist())).hexdigest())
                rows.append(row);print(json.dumps({k:row[k] for k in ("seed","arm","length","median_seconds","all_finite","all_parameter_tensors_active")}),flush=True)
                source_files=['collect011/model.py','collect011/memory.py','collect011/benchmark.py']
                source_hashes={f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in source_files}
                (root/f"results/{args.output}.json").write_text(json.dumps(dict(status="ENGINEERING_ONLY_NO_FIT",torch=torch.__version__,source_hashes=source_hashes,
                    platform=platform.platform(),threads=torch.get_num_threads(),data_sha256=hashlib.sha256(data).hexdigest(),
                    optimizer_updates=0,test_read=False,rows=rows),indent=2))

if __name__=="__main__":main()
