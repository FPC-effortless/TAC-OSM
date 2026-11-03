"""Read-only first-answer credit by actual byte position; no optimizer calls."""
import argparse,json
from pathlib import Path
import torch
from .common import build,digest,require
def profile(folder,step):
    r=json.loads((folder/'FIT_RECEIPT.json').read_text());ck=folder/f'checkpoint_{step:04}.pt'
    saved=torch.load(ck,weights_only=True,map_location='cpu');m=build(r['arm'],r['seed']);m.load_state_dict(saved['model']);m.eval()
    replay=torch.load(folder/'REPLAY.pt',weights_only=True,map_location='cpu');x=replay['tokens'];mask=replay['answer_mask'];target=replay['targets'];captures=[];states=[]
    def capture(module,args,out):out.retain_grad();captures.append(out)
    h=m.core.embedding.register_forward_hook(capture);state=None;pieces=[];owners=('profile-natural','profile-task')
    for lo in range(0,x.shape[1],128):
        z,state=m(x[:,lo:lo+128],state=state,owners=owners);pieces.append(z)
        if lo==0:
            states=list(state.phase)
            for p in states:
                if p.memory.requires_grad:p.memory.retain_grad()
        if r['arm']=='detach_128' and lo+128<x.shape[1]:state=state.detach()
    first=int(mask[1].nonzero()[0]);z=torch.cat(pieces,1)
    loss=torch.nn.functional.cross_entropy(z[1,first:first+1],target[1,first:first+1]);loss.backward();h.remove()
    gradient=torch.cat([torch.zeros_like(v) if v.grad is None else v.grad for v in captures],dim=1)
    require(float(gradient[0].abs().sum())==0.,'another owner received task-only activation credit')
    per_position=gradient[1].abs().sum(-1);early=float(per_position[:128].sum());late=float(per_position[256:].sum())
    return dict(arm=r['arm'],seed=r['seed'],checkpoint_step=step,checkpoint_sha256=digest(ck),first_answer_prediction_index=first,
        first_answer_nll=float(loss.detach()),first_chunk_gradient_l1=early,last_chunk_gradient_l1=late,
        first_to_last_gradient_ratio=None if late==0 else early/late,position_gradient_l1=per_position.tolist(),
        phase_state_credit=[None if not p.memory.requires_grad or p.memory.grad is None else float(p.memory.grad.abs().sum()) for p in states],
        cross_owner_gradient_l1=0.,optimizer_updates=0,caution='Local sensitivity at fixed weights, not proof of learned retrieval or of optimization feasibility')
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--steps',type=int,nargs='+',default=[512]);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(1);rows=[profile(a.run,s) for s in a.steps];a.output.write_text(json.dumps(dict(rows=rows,optimizer_updates=0),indent=2));print(json.dumps([{k:v for k,v in r.items() if k!='position_gradient_l1'} for r in rows],indent=2))
if __name__=='__main__':main()
