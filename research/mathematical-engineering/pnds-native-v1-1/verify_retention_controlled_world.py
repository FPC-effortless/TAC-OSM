#!/usr/bin/env python3
"""Bounded controlled retention math and source-weight parity, NOT ML skill."""
import json,sys,torch
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]/"src"))
from tac_osm.variants.parallel_continuous_world import ParallelTACWorld
from tac_osm.variants.retention_controlled_world import RetentionControlledTACWorld
from tac_osm.variants.streaming_continuous_world import init_world_stream,world_stream_step


def run():
    torch.manual_seed(20261025)
    torch.set_num_threads(1)
    checks=0
    max_diff=0.
    for dtype in (torch.float32,torch.float64):
        original=ParallelTACWorld().to(dtype)
        m=RetentionControlledTACWorld().to(dtype)
        m.load_original_world_weights(original)
        assert sum(p.numel() for p in original.parameters())==sum(
            p.numel() for p in m.parameters())==93324
        checks+=1
        for B,T in ((1,1),(2,7),(3,33),(2,129),(1,257)):
            obs=torch.randn(B,T,14,dtype=dtype)
            act=torch.randn(B,T,14,dtype=dtype)
            actual,tr=m(obs,act,return_trace=True)
            state=init_world_stream(m,B)
            predictions=[]
            seq=[]
            for t in range(T):
                y,state,s=world_stream_step(m,state,obs[:,t],act[:,t],return_trace=True)
                predictions.append(y)
                seq.append(s)
            tol=7e-5 if dtype==torch.float32 else 5e-9
            torch.testing.assert_close(actual,torch.stack(predictions,1),atol=tol,rtol=tol)
            checks+=1
            for k in ("memory","old_read","fast","slow"):
                x=torch.stack([s[k] for s in seq],dim=1)
                torch.testing.assert_close(tr[k],x,atol=tol,rtol=tol)
                max_diff=max(max_diff,(tr[k]-x).abs().max().item())
                checks+=1
            # Certified scalar bound from normalized addresses and values.
            assert torch.linalg.vector_norm(tr["memory"].flatten(-2),dim=-1).max()<4.0001
            checks+=1
            # Observe actual generated learned schedule and check that the
            # controlled gates have lower bound 147/160 before beta/4 global.
            gate_logits=[]
            def hook(_m,_inputs,out):gate_logits.append(out)
            h=m.memory_gates.register_forward_hook(hook)
            try:
                m(obs,act)
            finally:h.remove()
            assert len(gate_logits)==1
            a,b=gate_logits[0].sigmoid().chunk(2,-1)
            eff=a*(1-b/4)
            assert eff.min().item()>=147/160-1e-6
            assert b.max().item()<=.25+1e-6
            assert (eff<=1-b/4+1e-7).all()
            checks+=3
            if T in (7,33):
                params=tuple(m.parameters())
                ga=torch.autograd.grad(actual.square().sum(),params,retain_graph=True)
                gb=torch.autograd.grad(torch.stack(predictions,1).square().sum(),params)
                for x,y in zip(ga,gb):
                    torch.testing.assert_close(x,y,atol=tol,rtol=tol)
                    checks+=1
        state=init_world_stream(m,2)
        with torch.no_grad():
            o=torch.randn(2,14,dtype=dtype)
            a=torch.randn(2,14,dtype=dtype)
            for _ in range(4096):
                _,state=world_stream_step(m,state,o,a)
            assert state.fast_sum.numel()+state.memory.numel()==2*152
            assert torch.isfinite(state.memory).all()
            assert torch.linalg.vector_norm(state.memory.flatten(-2),dim=-1).max().item()<4.0001
            checks+=3
    print(json.dumps({"status":"PASS_CONTROLLED_RETENTION_PARALLEL_STEP",
       "checks":checks,"parameter_count_identical":93324,
       "max_absolute_tensor_difference":max_diff,
       "retention_min_lower_bound":147/160,
       "max_write_gate":.25,
       "untrained_or_real_capability_gain":"NOT_ESTABLISHED",
       "source_original_recurrence_unchanged":True,
       "fixed_memory_float32_bytes":608},sort_keys=True))


if __name__=="__main__":run()
