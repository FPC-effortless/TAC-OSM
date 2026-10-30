#!/usr/bin/env python3
"""Independent PNDS partial split/read/local AD correspondence, no fit or DEV."""
import argparse,importlib.util,json,sys
from dataclasses import replace
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'research/mathematical-engineering/langmem087'))
from diagnostic_tac087 import DiagnosticTAC087

def main():
 p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);a=p.parse_args()
 spec=importlib.util.spec_from_file_location('reference087',a.reference);ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
 torch.set_num_threads(2);torch.manual_seed(87);m=DiagnosticTAC087().double().eval()
 with torch.no_grad():_,state=m(torch.tensor([[31,35,65,42,90]]),owners=('operator',))
 matrix=state.tensors['slow_memory'].clone().requires_grad_(True);state=replace(state,tensors={**state.tensors,'slow_memory':matrix})
 _,nxt,tr=m(torch.tensor([[33]]),state=state,owners=('operator',),return_trace=True);f=tr[0]
 k=f['writer'][0].detach().tolist();v=f['value'][0].detach().tolist();q=f['query'][0].detach().tolist();rho=float(f['slow_rho'].detach());eta=float(f['slow_eta'][0,0].detach());M=matrix[0].detach().tolist()
 old,current=ref.split_slow(M,k,v,rho,eta)
 # Explicit dtype avoids accidentally comparing double source with rounded float32 reference.
 error=max(float((f['old_slow_component'][0]-torch.tensor(old,dtype=torch.float64)).abs().max().detach()),float((f['current_slow_component'][0]-torch.tensor(current,dtype=torch.float64)).abs().max().detach()))
 read_error=0.
 for mode,lesion in [('full',None),('old_zero','old_slow_zero'),('current_zero','current_slow_write_zero'),('old_permuted','old_slow_permuted')]:
  _,_,trace=m(torch.tensor([[33]]),state=state,owners=('operator',),lesion=lesion,return_trace=True)
  r=ref.attributed_read(M,k,v,q,rho,eta,mode);read_error=max(read_error,float((trace[0]['read_slow'][0]-torch.tensor(r,dtype=torch.float64)).abs().max().detach()))
 grad=torch.autograd.grad(f['old_slow_component'].sum(),matrix)[0][0];h=1e-6;derivative_error=0.
 for i,j in ((0,0),(2,5),(5,2),(7,7)):
  plus=[r[:] for r in M];minus=[r[:] for r in M];plus[i][j]+=h;minus[i][j]-=h
  fun=lambda x:sum(map(sum,ref.split_slow(x,k,v,rho,eta)[0]))
  derivative_error=max(derivative_error,abs((fun(plus)-fun(minus))/(2*h)-float(grad[i,j])))
 if max(error,read_error)>1e-12 or derivative_error>1e-6:raise ValueError('actual graph scoped correspondence')
 print(json.dumps({'status':'PASS_PARTIAL_087_SOURCE_SPLIT_READ_LOCAL_AD','split_max_error':error,'read_max_error':read_error,'local_derivative_max_error':derivative_error,'whole_graph_IEEE_AD':'OPEN'}))
if __name__=='__main__':main()
