"""Post hoc existing-deployment baseline; no predictive fit or metric scoring."""
import json,math,time
import numpy as np
import torch
import scripts.me039_research as b


def run():
    torch.set_num_threads(1);rows=b.source();old=json.loads((b.ROOT/'results/me038/runtime.json').read_text())
    original={r['key']:r for r in old['models']};new=json.loads((b.OUT/'frozen-audit.json').read_text());current={r['key']:r for r in new['models']};results=[]
    for r in rows:
        if r['lane']!='exposure' or r['dataset'] not in ('ETTh1','electricity') or r['horizon'] not in (96,720):continue
        a=original[r['key']];c=current[r['key']]
        assert b.sha(b.ROOT/a['physical_path'])==a['physical_sha256']
        assert b.sha(b.ROOT/c['path'])==c['sha256']
        models={'original':torch.jit.load(str(b.ROOT/a['physical_path'])).eval(),
                'scripted_core':torch.jit.load(str(b.ROOT/c['path'])).eval()}
        folder=b.ROOT/'results/me038'/r['key'];x=torch.from_numpy(np.load(folder/'serving-input.npy'));gold=np.load(folder/'serving-output.npy')
        times={k:[] for k in models}
        with torch.no_grad():
            for name,model in models.items():
                assert np.array_equal(model(x).numpy(),gold)
                for _ in range(5):model(x[:1])
            for i in range(30):
                for name in (('original','scripted_core') if i%2==0 else ('scripted_core','original')):
                    t=time.perf_counter();models[name](x[:1]);times[name].append((time.perf_counter()-t)*1000)
        med={name:float(np.median(t)) for name,t in times.items()}
        results.append({'case':r['key'],'B1_physical_medians_ms':med,'speedup':med['original']/med['scripted_core'],'samples_ms':times})
    output={'post_hoc':True,'addendum_sha256':b.sha(b.REPORT/'SERVING-ADDENDUM.md'),'cases':results,
        'geometric_speedup':math.exp(np.mean([math.log(r['speedup']) for r in results])),
        'original_all60_export_bytes':sum(r['physical_bytes'] for r in original.values()),
        'new_all60_export_bytes':sum(r['bytes'] for r in current.values()),
        'all_original_weights_and_forecasts_unchanged':True,'new_accuracy_claim':False}
    b.dump(b.OUT/'serving-comparison.json',output);print(json.dumps(output,indent=2),flush=True)

if __name__=='__main__':run()
