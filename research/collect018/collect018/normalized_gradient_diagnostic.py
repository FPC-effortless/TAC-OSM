"""Post-failure loss-scaling diagnostic; never clears original sum-loss failures."""
import argparse,copy,json
from pathlib import Path
import torch
from .common import build,digest,fingerprint,require
from .reference import whole_reference
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1)
    r=json.loads((a.run/'FIT_RECEIPT.json').read_text());ck=a.run/r['checkpoint'];require(digest(ck)==r['checkpoint_sha256'],'checkpoint differs')
    m=build(r['arm'],r['seed']);m.load_state_dict(torch.load(ck,weights_only=True,map_location='cpu')['model']);n=copy.deepcopy(m);before=fingerprint(m)
    replay=torch.load(a.run/'REPLAY.pt',weights_only=True,map_location='cpu');x=replay['tokens'][:2,:9];y,_=m(x);z,_=whole_reference(n,x);y.square().mean().backward();z.square().mean().backward();failures=[];maximum=0.;numerator=denominator=0.
    for (name,p),(other,q) in zip(m.named_parameters(),n.named_parameters()):
        require(name==other and p.grad is not None and q.grad is not None,'missing gradient')
        try:torch.testing.assert_close(p.grad,q.grad,atol=4e-4,rtol=4e-4)
        except AssertionError as exc:failures.append(dict(parameter=name,error=str(exc)))
        maximum=max(maximum,float((p.grad-q.grad).abs().max()));numerator+=float((p.grad-q.grad).double().square().sum());denominator+=float(q.grad.double().square().sum())
    require(before==fingerprint(m),'diagnostic changed weights')
    a.output.write_text(json.dumps(dict(arm=r['arm'],seed=r['seed'],checkpoint_sha256=digest(ck),status='PASS_NEW_NORMALIZED_LOSS_DIAGNOSTIC' if not failures else 'FAILED_NEW_NORMALIZED_LOSS_DIAGNOSTIC',loss='mean(square(logits)); original sum-square probe failures retained',scale_divisor=y.numel(),max_abs_gradient_error=maximum,relative_l2_gradient_error=(numerator/max(denominator,1e-300))**.5,atol=4e-4,rtol=4e-4,failures=failures,optimizer_updates=0,clears_original_audit_failure=False),indent=2))
if __name__=='__main__':main()
