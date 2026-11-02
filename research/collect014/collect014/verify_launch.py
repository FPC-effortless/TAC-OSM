"""Verify actual source, permitted data, registered CPU route and control IDs."""
import json,sys
from .common import ROOT,OLD,SPECS,source_paths,digest,verify_data,require,internal_gate

def main():
    request=json.loads((ROOT/'LAUNCH_REQUEST.json').read_text())
    require(request['jobs']==9 and request['max_parallel_jobs']==3 and request['gpu_minutes']==0,'allocation differs')
    actual={p:digest(f) for p,f in source_paths().items()};reports={}
    require(actual==request['source_hashes'],'launch executable differs')
    require(digest(ROOT/'EVALUATION_PROTOCOL.json')==request['evaluation_protocol_sha256'],'evaluation changed')
    require(digest(ROOT/'REUSED_CONTROLS.json')==request['controls_sha256'],'control registry changed')
    data,_=verify_data();sys.path.insert(0,str(OLD/'policy/src'))
    from tac_osm.phone_research_ops import decide
    for arm in SPECS:
        path=ROOT/'registrations'/arm/'FAIR_COMPARISON.json';manifest=json.loads(path.read_text())
        require(digest(path)==request['registrations'][arm] and manifest['source_hashes']==actual,'registration differs')
        proposal=json.loads((path.parent/'PHONE_PROPOSAL.json').read_text())
        require(digest(ROOT/'policy'/proposal['study_manifest'])==digest(path),'policy alias differs')
        plan=decide(proposal)
        require(plan['backend']=='github_actions' and plan['state']=='READY_INTERNAL_RESEARCH_PLAN' and not plan['blockers'],'CPU route not admitted')
        reports[arm]=dict(gate=internal_gate(manifest),plan=plan)
    print(json.dumps(dict(status='PASS_SOURCE_DATA_INTERNAL_CPU_LAUNCH',reports=reports,protected_test=False,qualified=False),indent=2))

if __name__=='__main__':main()
