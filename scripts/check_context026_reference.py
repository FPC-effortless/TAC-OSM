#!/usr/bin/env python3
"""Actual new source to independent key/native recurrence, constructed FP64 only."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import urllib.request
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'research/mathematical-engineering/context026'))
from context_transition import ContextTransition026


def main():
    d=ROOT/'research/mathematical-engineering/context026';link=json.loads((d/'PNDS_MATH_LINK.json').read_text())
    if hashlib.sha256((d/'context_transition.py').read_bytes()).hexdigest()!=link['tac_source_sha256']:raise ValueError('actual candidate source differs')
    url=f"https://raw.githubusercontent.com/FPC-effortless/pnds-lean/{link['pnds_commit']}/foundation/v1/context026.py"
    with urllib.request.urlopen(url,timeout=30) as response:raw=response.read()
    if hashlib.sha256(raw).hexdigest()!=link['python_sha256']:raise ValueError('PNDS reference differs')
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/'context_reference.py';path.write_bytes(raw)
        spec=importlib.util.spec_from_file_location('independent_context026',path);ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
        torch.set_num_threads(2);torch.manual_seed(26026);model=ContextTransition026().double().eval()
        tokens=[4,9,4,10,4,20,8,7,6,9,4,11];x=torch.tensor([tokens])
        with torch.no_grad():_,tr=model(x,return_trace=True)
        keys=ref.contextual_keys(tokens,model.embedding.weight.tolist(),model.context_lag_weights.tolist(),model.native_key.weight.tolist(),model.native_key.bias.tolist())
        # Construct reference tensor in FP64; default FP32 would falsify strict correspondence.
        key_error=float((torch.tensor(keys,dtype=torch.float64)-tr['native_query_key'][0]).abs().max())
        memory=[[0.]*8 for _ in range(8)];max_error=0.
        for t in range(len(tokens)):
            memory=ref.native_step(memory,tr['native_write_key'][0,t].tolist(),tr['native_value'][0,t].tolist(),float(tr['native_alpha'][0,t]),float(tr['native_beta'][0,t]),bool(tr['native_admission'][0,t]))
            max_error=max(max_error,float((torch.tensor(memory,dtype=torch.float64)-tr['native_memory'][0,t]).abs().max()))
        after,before=ref.oblique_erase_counterexample()
        if key_error>1e-12 or max_error>1e-12 or not after>before:raise ValueError('actual source/reference correspondence failed')
        print(json.dumps(dict(status='PASS_CHANGED_KEY_AND_NATIVE_OPERATOR_FP64_FIXTURES_ONLY',key_max_error=key_error,native_serial_max_error=max_error,
            oblique_squared_norm_after=after,before=before,full_parent_graph_IEEE_AD='OPEN',candidate_trained=False)))


if __name__=='__main__':main()
