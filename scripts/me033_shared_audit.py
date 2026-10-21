"""Post-fit exact-storage audit, explicitly not an added fitted arm."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
from tac_osm.me033_shared_runtime import SharedFrozenForecaster
from tac_osm.me033_semantic_moments import build_model
from tac_osm.me032_conditional_memory import FrozenContext
from scripts.me033_audit import verify
from scripts.me033_semantic_research import ROOT,roles_manifest,load_source,subset,file_sha,write_json
from scripts.me014b_exact_action_memory import batch


def run(out,verify_existing=False):
    out=Path(out).resolve();record=verify(out);torch.set_num_threads(1)
    prior=json.loads((out/'shared-storage.json').read_text()) if verify_existing else None
    if prior is not None:
        for row in prior['rows']:
            assert file_sha(out/(row['name']+'-shared.jit.pt'))==row['sha256']
    data=load_source(ROOT/'results/me022')['train'];roles=roles_manifest(data)
    rows=[]
    for cell in record['cells']:
        inherited=torch.load(out/(cell['tag']+'-inherited.pt'),weights_only=True)
        cs=inherited['context_state'];context=FrozenContext(cs['emb.weight'],float(cs['duration_mean']),float(cs['duration_scale']),
            cs['item_popularity'],cs['feature_mean'],cs['feature_scale'])
        part=subset(data,torch.tensor(roles[cell['rotation']]['evaluation']['indices']))
        for st in cell['students']:
            if st['arm'] not in ('history_frozen','query_frozen'):continue
            ck=torch.load(out/(st['name']+'.pt'),weights_only=True)
            student=build_model(st['arm'],inherited['common_state'],inherited['embedding'],st['seed'])
            student.load_state_dict(ck['state_dict'],strict=True)
            new=SharedFrozenForecaster(context,student)
            count=sum(p.numel() for p in new.parameters());assert count==137977
            full=sum(p.numel() for p in context.parameters())+sum(p.numel() for p in student.parameters());assert full==265977
            path=out/(st['name']+'-shared.jit.pt')
            if verify_existing:
                original=next(row for row in prior['rows'] if row['name']==st['name'])
                assert file_sha(path)==original['sha256']
            else:torch.jit.script(new.eval()).save(str(path))
            net=torch.jit.load(str(path)).eval()
            assert sum(p.numel() for p in net.parameters())==137977
            assert net.context.emb.weight.data_ptr()==net.student.memory.emb.weight.data_ptr()
            with torch.no_grad():
                probability=torch.cat([net(*batch(part,slice(i,i+256)))[:,-1].sigmoid()
                                       for i in range(0,len(part['y']),256)]).numpy().astype(np.float64)
            expected=np.load(out/(st['arm']+'-'+cell['tag']+'-predictions.npy'))
            np.testing.assert_array_equal(probability,expected)
            rows.append({'name':st['name'],'original_unique_weight_parameters':full,'shared_unique_weight_parameters':count,
                'float32_tensor_weight_bytes_saved':512000,'new_export_bytes':path.stat().st_size,
                'forecasts':len(probability),'bit_exact':True,'sha256':file_sha(path)})
    result={'status':'PASS','scope':'Post-fit exact immutable storage engineering, no training, no new scored architecture arm',
        'rows':rows,'pipeline_paths':len(rows),'forecasts':sum(row['forecasts'] for row in rows),
        'tensor_weights_only_storage_reduction_percent':(265977-137977)/265977*100,
        'buffers_and_activations_included':False,'global_forecast_state_reduced_claim':False,
        'new_fit_count':0}
    if verify_existing:assert result==prior
    else:write_json(out/'shared-storage.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__=='__main__':run(sys.argv[1] if len(sys.argv)>1 else 'results/me033','--verify-existing' in sys.argv)
