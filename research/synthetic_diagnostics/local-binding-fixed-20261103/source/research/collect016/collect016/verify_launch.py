"""Closed launch-input identities; planning never substitutes for source checks."""
import json
from .common import ROOT,ARMS,EXPECTED,source_paths,digest,verify_data,require,internal_gate,build
def main():
    verify_data();r=json.loads((ROOT/'LAUNCH_REQUEST.json').read_text())
    source={p:digest(v) for p,v in source_paths().items()};require(r['source_hashes']==source,'launch source changed')
    gates={}
    for arm in ARMS:
        p=ROOT/'registrations'/arm/'FAIR_COMPARISON.json';m=json.loads(p.read_text());require(digest(p)==r['registrations'][arm],'registration differs')
        require(m['source_hashes']==source,'registered source differs')
        require(digest(ROOT/'EVALUATION_PROTOCOL.json')==m['evaluation_protocol_sha256'],'evaluation differs')
        require(digest(ROOT/'EXPERIMENT_PROTOCOL.json')==m['experiment_protocol_sha256'],'experiment differs')
        require(sum(v.numel() for v in build(arm,16001).parameters())==EXPECTED[arm],'parameter count differs');gates[arm]=internal_gate(m)
    require(r['gpu_minutes']==0 and r['jobs']==9,'allocation differs');print(json.dumps(dict(status='PASS_INTERNAL_CPU_SOURCE_PREFLIGHT',gates=gates),indent=2))
if __name__=='__main__':main()
