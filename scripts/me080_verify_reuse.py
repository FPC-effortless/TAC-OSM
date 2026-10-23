"""No-fit actual trained-model chunk/step equality on original real query inputs."""
import json
from pathlib import Path
import torch
from scripts.me080_train_world import integrity,frozen,SEEDS,OUT,ROOT,load_original,evaluation_origins,batch,write,sha
from tac_osm.variants.streaming_mixture_world import StreamingMixtureWorld


def main():
    torch.set_num_threads(1);integrity();frozen()
    _,_,rows,identified,_=load_original(Path('/tmp/me077-original-source'))
    queries,_=evaluation_origins(rows,identified);records=[]
    with torch.no_grad():
        for seed in SEEDS:
            folder=OUT/f'seed-{seed}';model=StreamingMixtureWorld(identified)
            model.load_state_dict(torch.load(folder/'final.pt',map_location='cpu',weights_only=True)['weights'])
            errors={};count=0
            for i in range(0,len(queries),24):
                x,a,_,_=batch(rows,queries[i:i+24]);full,fullstate=model(x,a,return_state=True)
                first,state=model(x[:,:3],a[:,:3],return_state=True)
                second,state=model(x[:,3:],a[:,3:],state,True)
                stream=[];carried=None
                for t in range(x.shape[1]):
                    p,carried=model.step(x[:,t],a[:,t],carried);stream.append(p)
                for key in full:
                    chunked=torch.cat((first[key],second[key]),1)
                    streamed=torch.stack([p[key] for p in stream],1)
                    for name,value in [('chunked',chunked),('step',streamed)]:
                        torch.testing.assert_close(full[key],value,rtol=1e-5,atol=1e-6)
                        label=name+'/'+key
                        errors[label]=max(errors.get(label,0.),float((full[key]-value).abs().max()))
                for v,w in zip(fullstate,carried):torch.testing.assert_close(v,w,rtol=1e-5,atol=1e-6)
                count+=len(x)
            records.append({'seed':seed,'checkpoint_sha256':sha(folder/'final.pt'),'original_query_windows':count,'status':'PASS','maximum_absolute_difference':errors})
    result={'status':'PASS','scope':'same eight-frame actual query computations; NOT arbitrary-history predictive accuracy or speed proof',
            'cache_float32_values_per_entity':157,'cache_tensor_payload_bytes_per_entity':628,'excluded':'tensor/container overhead, model weights, activations',
            'rtol':1e-5,'atol':1e-6,'records':records,'no_fit':True}
    path=ROOT/'TRAINED_STATE_REUSE_AUDIT.json'
    if path.exists() and json.loads(path.read_text())!=result:raise RuntimeError('Never overwrite original audit')
    if not path.exists():write(path,result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
