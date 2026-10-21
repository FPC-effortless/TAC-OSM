"""Independent export process, optimizer receipts, corruption and profiler audit."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
import torch
from torch.profiler import profile, ProfilerActivity
from scripts.me030_joint_state_research import source
from scripts.me014b_exact_action_memory import batch
from scripts.me028_native_readout_fit import file_sha, write_json


def run(root):
    root=Path(root).resolve()
    r=json.loads((root/'results.json').read_text())
    for row in r['rows']:
        for suffix,digest in row['files_sha256'].items():
            assert file_sha(root/(row['name']+suffix))==digest
    torch.set_num_threads(1)
    fit,dev,*_=source()
    summary={'fresh_torch_only_exports':[],'optimizer_receipts':[],'profiler_subtotals':[]}
    with tempfile.TemporaryDirectory() as td:
        tmp=Path(td)
        inputs=tmp/'inputs.pt';torch.save(batch(dev,slice(0,4)),inputs)
        code="""import sys,torch,numpy as np
assert not any(n.startswith(('tac_osm','scripts')) for n in sys.modules)
torch.set_num_threads(1)
net=torch.jit.load(sys.argv[1]);net.eval()
args=torch.load(sys.argv[2],weights_only=True)
with torch.no_grad():np.save(sys.argv[3],torch.sigmoid(net(*args)).flatten().numpy())
assert not any(n.startswith(('tac_osm','scripts')) for n in sys.modules)
"""
        for row in r['rows']:
            n=row['name'];ck=torch.load(root/(n+'.pt'),weights_only=True)
            assert {float(x['step']) for x in ck['optimizer']['state'].values()}=={300.}
            sampler=torch.Generator().manual_seed(row['seed']+300)
            for _ in range(300):torch.randint(len(fit['y']),(128,),generator=sampler)
            assert torch.equal(sampler.get_state(),ck['sampler_rng'])
            clocks=json.loads((root/(n+'-trace.json')).read_text())['per_update']
            assert len(clocks)==300 and [x['step'] for x in clocks]==list(range(1,301))
            summary['optimizer_receipts'].append({'name':n,'Adam_steps':300,'sampler_RNG_exact':True,
                                                 'per_update_clock_entries':len(clocks)})
            output=tmp/'actual.npy'
            subprocess.run([sys.executable,'-c',code,str(root/(n+'.jit.pt')),str(inputs),str(output)],
                           check=True,cwd=tmp,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            expected=np.load(root/(n+'.npy'))[:4];actual=np.load(output)
            np.testing.assert_allclose(actual,expected,atol=1e-7,rtol=0)
            summary['fresh_torch_only_exports'].append({'name':n,'real_inputs':4,
                'max_abs_error':float(np.max(np.abs(expected-actual)))})
            if row['seed']==14001:
                net=torch.jit.load(str(root/(n+'.jit.pt')));net.train()
                args=batch(fit,slice(0,8));y=torch.cat((fit['metadata'][:8,[32,64,96],1],fit['y'][:8]),-1)
                # Warm up TorchScript specialization before instrumented execution.
                net.prefix_logits(*args).sum().backward();net.zero_grad(set_to_none=True)
                with profile(activities=[ProfilerActivity.CPU],with_flops=True) as prof:
                    logits=net.prefix_logits(*args)
                    loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,y)
                    loss.backward()
                subtotal=sum(e.flops or 0 for e in prof.key_averages())
                summary['profiler_subtotals'].append({'arm':row['arm'],'FIT_windows':8,
                    'prefix_targets':4,'instrumented_forward_backward_FLOP_subtotal':subtotal,
                    'full_FLOPs':False,'optimizer_counted':False,
                    'limitations':'Operator coverage omits some elementwise, nonlinear and fused work; not a compute-parity certificate.'})
        corrupt=tmp/'corrupt';corrupt.mkdir()
        for f in root.iterdir():
            if f.is_file():(corrupt/f.name).symlink_to(f)
        f=corrupt/(r['rows'][0]['name']+'.pt');f.unlink();f.write_bytes(b'corrupt')
        p=subprocess.run([sys.executable,'-m','scripts.me030_joint_state_research','--out',str(corrupt),'--replay'],
                         capture_output=True,text=True)
        assert p.returncode and 'artifact hash mismatch' in p.stderr
        summary['corruption_fail_closed_before_source']=True
    summary['status']='PASS'
    write_json(root/'artifact-audit.json',summary)


if __name__=='__main__':run(sys.argv[1] if len(sys.argv)>1 else 'results/me030')
