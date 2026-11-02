"""Reject stale/unreviewed launches and verify the real internal CPU route."""
import hashlib,json,sys
from .common import ROOT,SPECS,internal_gate,verify_data,digest,require,source_paths

def main():
    request=json.loads((ROOT/'LAUNCH_REQUEST.json').read_text())
    require(request['status']=='REGISTERED_INTERNAL_CPU_FITS','wrong launch stage')
    require(request['gpu_minutes']==0 and request['max_parallel_jobs']==3,'unregistered compute')
    for path,sha in request['source_hashes'].items():
        require(digest(ROOT/path)==sha,'launch executable changed: '+path)
    data,_=verify_data();sys.path.insert(0,str(ROOT/'policy/src'))
    from tac_osm.phone_research_ops import decide
    reports={}
    for arm in SPECS:
        path=ROOT/'registrations'/arm/'FAIR_COMPARISON.json'
        manifest=json.loads(path.read_text())
        require(digest(path)==request['registrations'][arm],'registration hash changed')
        require(manifest['source_hashes']=={p:digest(f) for p,f in source_paths().items()},'model source changed')
        gate=internal_gate(manifest)
        proposal=json.loads((ROOT/'policy/research/phone_queue'/('collect012_'+arm+'.json')).read_text())
        plan=decide(proposal)
        require(plan['backend']=='github_actions' and plan['state']=='READY_INTERNAL_RESEARCH_PLAN' and not plan['blockers'],'CPU plan not admissible')
        reports[arm]=dict(gate=gate,plan=plan)
    print(json.dumps(dict(status='PASS_INTERNAL_CPU_LAUNCH',reports=reports,protected_test=False,qualified=False),indent=2))

if __name__=='__main__':main()
