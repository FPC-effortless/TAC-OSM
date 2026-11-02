"""Pre-results supplemental local-binding versus held-out-key-family assay."""
import argparse,json
from pathlib import Path
from unittest.mock import patch
import torch
from .common import ROOT,build,digest,verify_data,require,fingerprint
from . import evaluate as original
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(1);r=json.loads((a.run/'FIT_RECEIPT.json').read_text());ck=a.run/r['checkpoint'];require(digest(ck)==r['checkpoint_sha256'],'checkpoint differs')
    m=build(r['arm'],r['seed']);m.load_state_dict(torch.load(ck,weights_only=True,map_location='cpu')['model']);m.eval();data,_=verify_data();before=fingerprint(m)
    protocol=json.loads((ROOT/'SUPPLEMENTAL_PROTOCOL.json').read_text());saved_history=original.history;all_results=[]
    with torch.inference_mode():
      for spec in protocol['strata']:
        normal_histories={};counterfactuals=[]
        def make_history(*args,**kwargs):
            kwargs['partition']=spec['key_family'];h=saved_history(*args,**kwargs)
            key=(h['n'],h['identity'],h['filler_bytes'])
            if not kwargs.get('class_shift',0):normal_histories[key]=h
            else:
                ordinary=normal_histories[key]
                end=sum(5+8+1 for _ in h['facts'])
                h['prefix']=h['prefix'][:end]+ordinary['prefix'][end:]
                h['filler_offset']=ordinary['filler_offset']
                require(h['prefix'][end:]==ordinary['prefix'][end:],'controlled filler differs')
                counterfactuals.append(dict(identity=h['identity'],n=h['n'],gap=h['filler_bytes'],
                    matching_filler=True,filler_offset=h['filler_offset'],shifted_source_facts=h['facts']))
            return h
        with patch.object(original,'history',make_history):
            result=original.retrieval(m,data['dev'],dict(histories=24,key_counts=[spec['keys']],gaps=[spec['gap']]))
        for c in result['cohorts']:c.update(held_out_key_family=spec['key_family']=='dev',quote_source='DEV',key_family=spec['key_family'])
        all_results.append(dict(stratum=spec,controlled_counterfactuals=counterfactuals,**result))
    require(before==fingerprint(m),'read-only supplemental changed weights')
    a.output.write_text(json.dumps(dict(arm=r['arm'],seed=r['seed'],checkpoint_sha256=digest(ck),protocol_sha256=digest(ROOT/'SUPPLEMENTAL_PROTOCOL.json'),
        results=all_results,optimizer_updates=0,weights_unchanged=True,caution='Additional declared-before-full-results diagnosis; primary held-out-key-family protocol unchanged'),indent=2))
    print(json.dumps(dict(arm=r['arm'],seed=r['seed'],strata=len(all_results),status='COMPLETE_READ_ONLY_SUPPLEMENTAL')))
if __name__=='__main__':main()
