"""Source-only bootstrap, exact inherited policy and fresh scoped admission."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[1]
POLICY={
 'admission_policy/amendment/research/RESEARCH-EXECUTION-LEVELS-2026-10-27.md':'research/collect012/policy/RESEARCH-EXECUTION-LEVELS-2026-10-27.md',
 'admission_policy/amendment/scripts/validate_research_parity.py':'research/collect012/policy/scripts/validate_research_parity.py',
 'admission_policy/amendment/src/tac_osm/research_execution.py':'research/collect012/policy/src/tac_osm/research_execution.py',
 'admission_policy/current/scripts/validate_research_parity.py':'scripts/validate_research_parity.py',
 'admission_policy/normative/REAL-WORLD-EVIDENCE-RULES.md':'research/REAL-WORLD-EVIDENCE-RULES.md',
 'admission_policy/vendored/scripts/validate_research_parity.py':'research/collect012/policy/scripts/validate_research_parity.py',
 'admission_policy/vendored/src/tac_osm/research_execution.py':'research/collect012/policy/src/tac_osm/research_execution.py'}
def install(target,raw,entry):
 if len(raw)!=entry['bytes'] or hashlib.sha256(raw).hexdigest()!=entry['sha256']:raise ValueError('SOURCE_BYTES_DIFFER '+str(target))
 if target.exists() and target.read_bytes()!=raw:raise ValueError('EXISTING_BYTES_DIFFER '+str(target))
 target.parent.mkdir(parents=True,exist_ok=True)
 if not target.exists():target.write_bytes(raw)
def main():
 protocol=json.loads((ROOT/'PROTOCOL.json').read_bytes());source=json.loads((ROOT/'SOURCE_MANIFEST.json').read_bytes())
 for name,entry in source['files'].items():install(ROOT/'source'/name,(REPO/name).read_bytes(),entry)
 for target,name in POLICY.items():install(ROOT/target,(REPO/name).read_bytes(),protocol['admission']['files'][target])
 for name,entry in json.loads((ROOT/'CODE_MANIFEST.json').read_bytes())['files'].items():install(ROOT/name,(ROOT/name).read_bytes(),entry)
 sys.path.insert(0,str(ROOT/'code'));import admission,fixtures
 data=fixtures.generate();fixed=data['fixed_train_cases.json'];schedule=data['fixed_schedule.json']*4
 for name,value in [('fixed_train_cases.json',fixed),('fixed_schedule.json',schedule),('no_heldout_groups.json',[])]:
  install(ROOT/'fixtures'/name,(json.dumps(value,indent=2)+'\n').encode(),protocol['fixture_files'][name])
 if len(fixed)!=8 or len(schedule)!=512 or {i:sum(row.count(i) for row in schedule) for i in [c['id'] for c in fixed]}!={c['id']:192 for c in fixed}:raise ValueError('EXPOSURE_DIFFERS')
 scoped=admission.inspect(ROOT,protocol,'fixed','default')
 plan=subprocess.run([sys.executable,'-B',str(REPO/'scripts/phone_lab.py'),'plan','research/phone_queue/local_binding_gru512_20261103.json'],cwd=REPO,capture_output=True,text=True,check=True)
 decision=json.loads(plan.stdout)
 if decision['backend']!='github_actions' or decision['approved_to_open_test'] is not False:raise ValueError('UNAPPROVED_ROUTE')
 if 'torch' in sys.modules:raise RuntimeError('SOURCE_PREFLIGHT_IMPORTED_TORCH')
 (ROOT/'PREFLIGHT_PLAN.json').write_text(json.dumps(decision,indent=2)+'\n')
 output={'status':'PASS_SOURCE_SCOPED_GRU512_ADMISSION_NO_MODEL_EXECUTION','source_files':len(source['files']),'fixed_cases':8,'exposures_per_case':192,'owner_admission':scoped,'compute_plan':decision,'models_constructed':0,'optimizer_updates':0,'TEST_opened':False}
 (ROOT/'PREFLIGHT_SOURCE.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))
if __name__=='__main__':main()
