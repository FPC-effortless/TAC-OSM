"""Supplemental separation of current-value shortcut and prior-state output."""
import argparse,json
from pathlib import Path
import torch
from .common import ROOT,build,verify_data,digest,require
from collect012.train import fingerprint,batch

def current_part(module,x,mask=None):
    b,t,_=x.shape;h,v=module.heads,module.value_dim
    route=torch.nn.functional.one_hot(module.router(x).argmax(-1),h).to(x.dtype)
    gain=module.admission(x).sigmoid()*route
    if mask is not None:gain=gain*mask[...,None]
    value=module.value(x).reshape(b,t,h,v).tanh()
    return route[...,None]*gain[...,None]*value,gain

def check_identity():
    rows=[];torch.set_num_threads(1)
    for dtype in (torch.float32,torch.float64):
        before=build('balance',12001).to(dtype);after=build('write_first',12001).to(dtype)
        x=torch.sin(torch.arange(2*11*16,dtype=dtype).reshape(2,11,16));mask=torch.ones(2,11,dtype=torch.bool);mask[:,3]=False
        maximum=0.
        with torch.no_grad():
            for a,b in zip(before.core.blocks,after.core.blocks):
                old,_=a.phase_memory(x,write_mask=mask);new,_=b.phase_memory(x,write_mask=mask)
                current,gain=current_part(b.phase_memory,x,mask);predicted=(1-gain[...,None])*old+current
                err=float((new-predicted).abs().max());maximum=max(maximum,err)
                torch.testing.assert_close(new,predicted,atol=2e-6 if dtype==torch.float32 else 1e-10,rtol=2e-6 if dtype==torch.float32 else 1e-10)
        rows.append(dict(dtype=str(dtype),maximum_error=maximum))
    return rows

def evaluate(folder,output):
    torch.set_num_threads(1);r=json.loads((folder/'FIT_RECEIPT.json').read_text());require(r['arm'] in ('write_first','combined'),'read-split scope differs')
    ck=folder/r['checkpoint'];require(digest(ck)==r['checkpoint_sha256'] and r['optimizer_updates']==1024,'checkpoint differs')
    m=build(r['arm'],r['seed']);m.load_state_dict(torch.load(ck,weights_only=True,map_location='cpu')['model']);m.eval();before=fingerprint(m)
    data,_=verify_data();rows=[]
    for mode in ('normal','current_component_only','historical_component_only'):
        handles=[]
        if mode!='normal':
            def intervene(module,args,kwargs,out):
                read,state=out;current,_=current_part(module,args[0],kwargs.get('write_mask'))
                return (current if mode=='current_component_only' else read-current),state
            handles=[b.phase_memory.register_forward_hook(intervene,with_kwargs=True) for b in m.core.blocks]
        total=0.;g=torch.Generator().manual_seed(120099)
        with torch.inference_mode():
            for _ in range(16):
                x,y,_=batch(data['dev'],g,4,128,'cpu');logits,_=m(x)
                total+=float(torch.nn.functional.cross_entropy(logits.flatten(0,1),y.flatten(),reduction='sum'))
        for h in handles:h.remove()
        rows.append(dict(mode=mode,nll=total/8192,targets=8192))
    require(before==fingerprint(m),'read-only diagnostic changed parameters')
    result=dict(variant=r['arm'],seed=r['seed'],checkpoint_sha256=digest(ck),rows=rows,
        protocol_sha256=digest(ROOT/'ADDITIONAL_READ_SPLIT_PROTOCOL.json'),weights_unchanged=True,optimizer_updates=0,
        caution='supplemental diagnostic, not trained alternative or isolation of every historical path')
    output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.run:evaluate(a.run,a.output)
    else:a.output.write_text(json.dumps(dict(status='PASS_OPERATOR_READ_DECOMPOSITION',rows=check_identity(),optimizer_updates=0),indent=2))

if __name__=='__main__':main()
