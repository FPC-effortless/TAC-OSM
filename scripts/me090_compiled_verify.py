"""Read-only independent restoration of committed fixed-profile executables."""
import argparse,json
from pathlib import Path
import torch
import scripts.me090_train_world as R
from scripts.me090_compile import CompleteGraph
from tac_osm.variants.intent_world import IntentWorld


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',choices=tuple(R.CONFIG),default='me090');parser.add_argument('--cache',type=Path,default=Path('/tmp/me077-original-source'));args=parser.parse_args()
    R.configure(args.study);R.integrity();R.frozen();torch.set_num_threads(1)
    audit=json.loads((R.ROOT/'COMPILED_GRAPH_AUDIT.json').read_text())
    _,fit,_,identified,_=R.load_original(args.cache);entries=R.schedule(fit,identified,R.SEEDS[0])[0][0]
    path=R.OUT/f'seed-{R.SEEDS[0]}'/'final.pt'
    if R.sha(path)!=audit['checkpoint_sha256']:raise RuntimeError('Original checkpoint mismatch')
    m=IntentWorld(identified);m.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['weights'],strict=True);m.double().eval();wrapper=CompleteGraph(m)
    with torch.no_grad():
        for offset in (0,2):
            x,a,_,_,dt,_=R.batch(fit,entries[offset:offset+2]);x,a,dt=x.double(),a.double(),dt.double()
            _,carried=m(x[:,:-1],a[:,:-1],elapsed=dt[:,:-1],return_state=True)
            for name,inputs in [('prefill32',(x,a,dt,tuple(m.initial_state(x[:,0])))),('incremental_step',(x[:,-1:],a[:,-1:],dt[:,-1:],tuple(carried)))]:
                executable=R.ROOT/f'{name.upper()}_REFERENCE_EXECUTABLE.pt'
                if R.sha(executable)!=audit['profiles'][name]['executable_sha256']:raise RuntimeError('Derived executable bytes changed')
                program=torch.jit.load(str(executable),map_location='cpu')
                for expected,actual in zip(wrapper(*inputs),program(*inputs)):torch.testing.assert_close(expected,actual,rtol=1e-5,atol=1e-6)
    print('INTENT_COMPILED_ORIGINAL_REPLAY='+json.dumps({'status':'PASS','study':R.STUDY,'original_checkpoint_sha256':R.sha(path),'profiles':['B2_T32','B2_T1'],'actual_FIT_observation_histories':4,'output_fields_each':20,'rtol':1e-5,'atol':1e-6,'no_fit':True,'scope':'First original real-FIT fixed-profile engineering audit, not original REPORT score replay or general shape qualification'}))

if __name__=='__main__':main()
