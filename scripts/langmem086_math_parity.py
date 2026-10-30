#!/usr/bin/env python3
"""Actual full-model single-step memory/reference and local derivative checks.
No optimizer, no new DEV, no assertion of complete model Float32/AD proof.
"""
import argparse,importlib.util,json,sys
from dataclasses import replace
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'research/mathematical-engineering/langmem086')]
from compact_tac import CompactDualClippedTAC086

def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);a=p.parse_args()
    spec=importlib.util.spec_from_file_location('pnds086',a.reference);ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    torch.manual_seed(86);torch.set_num_threads(2);m=CompactDualClippedTAC086().double().eval()
    with torch.no_grad():_,state=m(torch.tensor([[40,41,42]]),owners=('operator-fixture',))
    Fm=state.tensors['fast_memory'].detach().clone().requires_grad_(True)
    state=replace(state,tensors={**state.tensors,'fast_memory':Fm})
    token=torch.tensor([[43]]);_,nxt,tr=m(token,state=state,owners=('operator-fixture',),return_trace=True)
    e=(m.embedding(token)+m.position(m.positions(3,1,1,torch.float64,token.device))).tanh()
    fg=m.fast_gate(e).sigmoid()[0,0].detach();rho=float(.97+.02*fg[0]);gate=float(fg[1])
    rho_s=float((.995+.004*m.slow_decay.sigmoid()).detach());eta_s=float((.01*m.slow_gate(e).sigmoid()[0,0,0]).detach())
    key=state.tensors['previous_writer'][0].detach().tolist();value=tr[0]['value'][0].detach().tolist()
    matrix=Fm[0].detach().tolist();expected=ref.clipped_update(matrix,key,value,rho,gate)
    slow=ref.residual_update(state.tensors['slow_memory'][0].detach().tolist(),key,value,rho_s,eta_s)
    fast_error=float((nxt.tensors['fast_memory'][0]-torch.tensor(expected,dtype=torch.float64)).abs().max().detach())
    slow_error=float((nxt.tensors['slow_memory'][0]-torch.tensor(slow,dtype=torch.float64)).abs().max().detach())
    if max(fast_error,slow_error)>1e-12:raise ValueError('actual model operator source parity')
    grad=torch.autograd.grad(nxt.tensors['fast_memory'].sum(),Fm)[0][0]
    max_grad_error=0.;h=1e-6
    for i,j in ((0,0),(2,5),(5,2),(7,7)):
        plus=[r[:] for r in matrix];minus=[r[:] for r in matrix];plus[i][j]+=h;minus[i][j]-=h
        f=lambda M:sum(sum(row) for row in ref.clipped_update(M,key,value,rho,gate))
        numerical=(f(plus)-f(minus))/(2*h)
        max_grad_error=max(max_grad_error,abs(numerical-float(grad[i,j])))
    if max_grad_error>1e-6:raise ValueError('actual clipped memory local gradient parity')
    print(json.dumps({'status':'PASS_PARTIAL_SOURCE_OPERATOR_AND_LOCAL_GRADIENT','actual_full_model_fast_error':fast_error,
        'actual_full_model_slow_error':slow_error,'local_memory_gradient_max_error':max_grad_error,'whole_model_IEEE_AD':'OPEN'}))
if __name__=='__main__':main()
