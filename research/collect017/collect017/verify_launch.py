import json
from .common import ROOT,source_paths,digest,verify_data,require,internal_gate,build
def main():
    verify_data();r=json.loads((ROOT/'LAUNCH_REQUEST.json').read_text());m=json.loads((ROOT/'registrations/attention/FAIR_COMPARISON.json').read_text())
    actual={k:digest(v) for k,v in source_paths().items()};require(actual==r['source_hashes']==m['source_hashes'],'source differs');require(digest(ROOT/'registrations/attention/FAIR_COMPARISON.json')==r['registration_sha256'],'registration differs')
    require(digest(ROOT/'EVALUATION_PROTOCOL.json')==m['evaluation_protocol_sha256'] and digest(ROOT/'EXPERIMENT_PROTOCOL.json')==m['experiment_protocol_sha256'],'protocol differs')
    require(sum(p.numel() for p in build('attention',16001).parameters())==99875,'live size differs');print(json.dumps(dict(status='PASS_SOURCE_PINNED_ATTENTION_SANITY_PREFLIGHT',gate=internal_gate(m)),indent=2))
if __name__=='__main__':main()
