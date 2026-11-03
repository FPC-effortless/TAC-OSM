import json
from .common import ROOT,ARMS,EXPECTED,build,digest,source_paths,verify_data,require,internal_gate
def main():
    verify_data();r=json.loads((ROOT/'LAUNCH_REQUEST.json').read_text());source={k:digest(v) for k,v in source_paths().items()};require(source==r['source_hashes'],'source differs from launch')
    for arm in ARMS:
        p=ROOT/'registrations'/arm/'FAIR_COMPARISON.json';m=json.loads(p.read_text());require(digest(p)==r['registration_sha256'][arm] and source==m['source_hashes'],'registration differs')
        require(digest(ROOT/'EXPERIMENT_PROTOCOL.json')==m['experiment_protocol_sha256'] and digest(ROOT/'EVALUATION_PROTOCOL.json')==m['evaluation_protocol_sha256'],'protocol changed')
        require(sum(p.numel() for p in build(arm,18001).parameters())==EXPECTED[arm.split('_')[0]],'count differs');internal_gate(m)
    print(json.dumps(dict(status='PASS_SOURCE_PINNED_LOCAL_ACQUISITION_PREFLIGHT',models=18,execution_level='internal_research'),indent=2))
if __name__=='__main__':main()
