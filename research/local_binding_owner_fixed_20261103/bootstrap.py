"""Reconstruct exact source/fixtures and check the owner-scoped CPU admission.

No model or optimizer is created. Source/policy bytes are authenticated before
being copied; generated fixture bytes must match the pre-existing protocol.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
POLICY={
 'admission_policy/amendment/research/RESEARCH-EXECUTION-LEVELS-2026-10-27.md':'research/collect012/policy/RESEARCH-EXECUTION-LEVELS-2026-10-27.md',
 'admission_policy/amendment/scripts/validate_research_parity.py':'research/collect012/policy/scripts/validate_research_parity.py',
 'admission_policy/amendment/src/tac_osm/research_execution.py':'research/collect012/policy/src/tac_osm/research_execution.py',
 'admission_policy/current/scripts/validate_research_parity.py':'scripts/validate_research_parity.py',
 'admission_policy/normative/REAL-WORLD-EVIDENCE-RULES.md':'research/REAL-WORLD-EVIDENCE-RULES.md',
 'admission_policy/vendored/scripts/validate_research_parity.py':'research/collect012/policy/scripts/validate_research_parity.py',
 'admission_policy/vendored/src/tac_osm/research_execution.py':'research/collect012/policy/src/tac_osm/research_execution.py'}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def install_bytes(target,raw,meta):
    if len(raw)!=meta['bytes'] or hashlib.sha256(raw).hexdigest()!=meta['sha256']:
        raise ValueError('SOURCE_BYTES_DIFFER '+str(target))
    if target.exists():
        if target.read_bytes()!=raw:raise ValueError('EXISTING_BYTES_DIFFER '+str(target))
        return
    target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-root',type=Path,default=REPO)
    parser.add_argument('--policy-capsule-root',type=Path)
    args=parser.parse_args()
    protocol=json.loads((ROOT/'PROTOCOL.json').read_bytes())
    source=json.loads((ROOT/'SOURCE_MANIFEST.json').read_bytes())
    for name,meta in source['files'].items():
        install_bytes(ROOT/'source'/name,(args.source_root/name).read_bytes(),meta)
    for dest,name in POLICY.items():
        sourcepath=(args.policy_capsule_root/dest) if args.policy_capsule_root else REPO/name
        install_bytes(ROOT/dest,sourcepath.read_bytes(),protocol['admission']['files'][dest])
    for name,meta in json.loads((ROOT/'CODE_MANIFEST.json').read_bytes())['files'].items():
        if sha(ROOT/name)!=meta['sha256']:raise ValueError('CODE_DIFFERS '+name)
    sys.path.insert(0,str(ROOT/'code'))
    import fixtures,admission,audit_scorer
    data=fixtures.generate()
    report=fixtures.audit(data)
    data['INPUT_AUDIT.json']=report
    data['SCORER_UNIT_TESTS.json']=audit_scorer.test()
    for name,value in data.items():
        install_bytes(ROOT/'fixtures'/name,(json.dumps(value,indent=2)+'\n').encode(),protocol['fixture_files'][name])
    admission_result=admission.inspect(ROOT,protocol,'fixed','default')
    if admission_result['status']!='PASS_ONE_OFF_OWNER_SYNTHETIC_DIAGNOSTIC':
        raise ValueError('OWNER_ADMISSION_DID_NOT_PASS')
    if 'torch' in sys.modules:raise RuntimeError('Torch imported during source-only bootstrap')
    # Run the unchanged repository planner. Its legacy blockers remain visible;
    # the reviewed one-off owner admission is separate from its venue decision.
    proposal='research/phone_queue/local_binding_owner_fixed_20261103.json'
    plan=subprocess.run([sys.executable,'-B',str(REPO/'scripts/phone_lab.py'),'plan',proposal],
                        cwd=REPO,capture_output=True,text=True,check=True)
    decision=json.loads(plan.stdout)
    if decision['backend']!='github_actions' or decision['approved_to_open_test'] is not False:
        raise ValueError('UNAPPROVED_COMPUTE_ROUTE')
    (ROOT/'PREFLIGHT_PLAN.json').write_text(json.dumps(decision,indent=2)+'\n')
    output={'status':'PASS_SOURCE_AND_SCOPED_ADMISSION_NO_MODEL_EXECUTION','source_files':len(source['files']),
            'fixture_files':len(data),'owner_admission':admission_result,'compute_plan':decision,
            'inherited_rejections_not_cleared':True,'models_constructed':0,'optimizer_updates':0}
    (ROOT/'PREFLIGHT_SOURCE.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
