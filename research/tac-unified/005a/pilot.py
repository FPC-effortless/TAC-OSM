"""Native CSL 10M no-fit interface check, TrainLab only."""
import os,sys,json,hashlib,importlib,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import prior004 as old
from native_model import selected_native_byte
ROOT=Path(__file__).resolve().parent
PROMPTS=[b'fixture operation alpha',b'fixture operation bravo',b'fixture operation charlie',b'fixture operation delta',b'fixture operation echo',b'fixture operation foxtrot']
VARIANTS=[b'alpha operation fixture',b'bravo operation fixture',b'charlie operation fixture',b'delta operation fixture',b'echo operation fixture',b'foxtrot operation fixture']
def digest_arr(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
def encoder_digest(m):
    h=hashlib.sha256()
    for name,t in sorted(list(m.named_parameters())+list(m.named_buffers()),key=lambda x:x[0]):
        h.update(name.encode()+b'\0');h.update(digest_arr(t.detach().cpu().numpy()).encode())
    return h.hexdigest()
def features(model,prompt,state=None,emit=False):
    xs=torch.tensor([list(prompt)],dtype=torch.long)
    seen=[]
    handle=model.core.final_norm.register_forward_hook(lambda module,args,out:seen.append(out))
    try: out,newstate=model.core(xs,state=state,stream_ids=('fixture',),emit_token_logits=emit)
    finally: handle.remove()
    assert len(seen)==1
    return seen[0][0,-1].clone(),newstate,'token_logits' in out[model.core.config.layers]
def binding(m,bank,tab):return (encoder_digest(m),digest_arr(bank.numpy()),digest_arr(tab))
def dispatch(m,bank,tab,expected,z,st):
    if binding(m,bank,tab)!=expected:raise ValueError('stale binding')
    slot=int((bank@F.normalize(z,dim=0)).argmax())
    return slot,tab[slot,st]
def main():
    seed=int(os.environ['TRAINLAB_SEED']);out=Path(os.environ['TRAINLAB_OUTPUT_DIR']);out.mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(seed)
    pins=json.loads((ROOT/'source-hashes.json').read_text())
    source_hashes={}
    for name,expected in pins.items():
        if not name.startswith('tac_osm.'):continue
        got=hashlib.sha256(Path(importlib.import_module(name).__file__).read_bytes()).hexdigest()
        assert got==expected,(name,got,expected)
        source_hashes[name]=got
    _,_,tab,truth=old.load(seed)
    m=selected_native_byte().eval()
    count=sum(p.numel() for p in m.parameters());assert count==10103943
    with torch.inference_mode():
        raw=[];parity=[];stream=[];no_logits=[]
        for p in PROMPTS:
            z,_,logits=features(m,p);raw.append(z);no_logits.append(not logits)
            ref,_,emitted=features(m,p,emit=True);assert emitted
            parity.append(float((z-ref).abs().max()))
            _,st,_=features(m,p[:len(p)//2]);chunk,_,_=features(m,p[len(p)//2:],st)
            stream.append(float((z-chunk).abs().max()))
        bank=F.normalize(torch.stack(raw),dim=1).contiguous()
        t=time.perf_counter_ns();pin=binding(m,bank,tab);hash_ms=(time.perf_counter_ns()-t)/1e6
        assert bool(torch.isfinite(bank).all())
        score=[];anchor_correct=0
        states=np.arange(256)
        for i,z in enumerate(raw):
            slot,got=dispatch(m,bank,tab,pin,z,states)
            anchor_correct+=int(np.sum(got==truth[i])) if slot==i else 0
        for i,p in enumerate(VARIANTS):
            z,_,_=features(m,p);sim=(bank@F.normalize(z,dim=0)).tolist()
            slot=int(np.argmax(sim));score.append(dict(expected=i,predicted=slot,scores=sim))
        key=(pin,0,1);cache={key:tab[1,tab[0,states]]}
        composition=int(np.sum(cache[key]==truth[1,truth[0,states]]))
        t=time.perf_counter_ns()
        for _ in range(3):
            for p in PROMPTS:features(m,p)
        feature_ms=(time.perf_counter_ns()-t)/18e6
        # Fixed mutation, not a training update; shadow weights restored below.
        weight=m.core.embedding.weight;original=weight.clone()
        rows=sorted(set(b for p in PROMPTS for b in p))
        weight[rows,:32]+=0.01
        changed=binding(m,bank,tab);stale_rejected=False
        try:dispatch(m,bank,tab,pin,raw[0],states)
        except ValueError:stale_rejected=True
        stale_cache_miss=(changed,0,1) not in cache
        moved=[];raw_slots=[]
        for i,p in enumerate(PROMPTS):
            z,_,_=features(m,p)
            moved.append(float(torch.linalg.vector_norm(z-raw[i])))
            raw_slots.append(int((bank@F.normalize(z,dim=0)).argmax()))
        weight.copy_(original)
        restored=binding(m,bank,tab)==pin
        slot,got=dispatch(m,bank,tab,pin,raw[0],states)
        restored=bool(restored and slot==0 and np.array_equal(got,truth[0]))
        t=time.perf_counter_ns()
        for _ in range(1000):r=tab[0,states]
        table_us=(time.perf_counter_ns()-t)/1000e3
    gates={'count':count==10103943,'pins':len(source_hashes)==6,'no_logits':all(no_logits),'parity':max(parity)<=1e-4,'stream':max(stream)<=1e-4,'finite':bool(torch.isfinite(bank).all()),'anchor':anchor_correct==1536,'compose':composition==256,'stale':stale_rejected and stale_cache_miss and changed!=pin,'restored':restored}
    result=dict(seed=seed,qualification='NOT_QUALIFIED',torch=torch.__version__,parameters=count,source_hashes=source_hashes,parity=parity,stream=stream,anchor_correct=anchor_correct,anchor_n=1536,variant=score,variant_correct=sum(x['predicted']==x['expected'] for x in score),composition_correct=composition,hash_ms=hash_ms,feature_ms=feature_ms,table_us=table_us,mutation_feature_l2=moved,unguarded_slot_agreement=sum(i==x for i,x in enumerate(raw_slots)),committed_pin=pin,changed_pin=changed,stale_rejected=stale_rejected,stale_cache_miss=stale_cache_miss,restored=restored,gates=gates)
    metrics={'passed_gates':sum(gates.values()),'anchor_accuracy':anchor_correct/1536,'variant_accuracy':result['variant_correct']/6,'feature_ms':feature_ms,'hash_ms':hash_ms}
    for name,obj in [('results',result),('metrics',metrics)]: (out/(name+'.json')).write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
    print(seed,metrics,gates,flush=True)
if __name__=='__main__':main()
