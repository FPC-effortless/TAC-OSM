"""Read-only, fail-closed readiness for the exact registered LANGMATH101 fit.

Run with no arguments after checkout and dependency installation. This checks
actual public GitHub mathematics CI and immutable reciprocal source bytes; it
does not execute an optimizer or establish protected independent qualification.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import sys
import urllib.request

HERE=Path(__file__).resolve().parent
PREFIX='research/mathematical-engineering/langmath101/'
PNDS='FPC-effortless/pnds-lean'
TAC='FPC-effortless/TAC-OSM'
SHA40=re.compile(r'^[0-9a-f]{40}$')
SHA64=re.compile(r'^[0-9a-f]{64}$')
REQUIRED_PINNED_FILES=(
 'candidates/models.py','candidates/native098.py','candidates/conventional.py',
 'candidates/conventional_reference.py','adaptive/models.py','adaptive/reference.py',
 'math/reference.py','math/test_reference.py','math/FORMULATION.md','math/LANGMATH101.lean')
REQUIRED_RECIPROCAL={
 PREFIX+'math/LANGMATH101.lean':'PNDS/LANGMATH101.lean',
 PREFIX+'math/FORMULATION.md':'foundation/v1/langmath101-formulation.md',
 PREFIX+'math/reference.py':'foundation/v1/langmath101_reference.py',
 PREFIX+'adaptive/reference.py':'foundation/v1/langmath101_adaptive_reference.py'}
REQUIRED_STEPS=(
 'Fetch exact source graph and verify reciprocal reference bytes',
 'Check full independent functional references',
 'Compile the full forward module and inspect four narrow theorem axioms')


def require(value,reason):
 if not value:raise ValueError(reason)


def sha(data):return hashlib.sha256(data).hexdigest()


def fetch_bytes(url):
 request=urllib.request.Request(url,headers={'User-Agent':'LANGMATH101-source-readiness',
  'Accept':'application/vnd.github+json' if url.startswith('https://api.github.com/') else 'text/plain'})
 with urllib.request.urlopen(request,timeout=30) as response:
  data=response.read(8*1024*1024+1)
 require(len(data)<=8*1024*1024,'Public source response exceeds bounded readiness size')
 return data


def safe_path(path):
 require(isinstance(path,str) and path and not PurePosixPath(path).is_absolute()
  and '..' not in PurePosixPath(path).parts,'Unsafe registered source path')
 return path


def local_source(study,path):
 require(path.startswith(PREFIX),'Reciprocal source lies outside this exact study')
 target=Path(study)/path[len(PREFIX):]
 require(not target.is_symlink() and target.resolve().is_relative_to(Path(study).resolve()),
  'Reciprocal local source path escapes study')
 return target.read_bytes()


def verify_remote_math(register,*,study=HERE,fetch=fetch_bytes):
 math=register.get('math',{})
 require(math.get('repository')==PNDS,'Exact registered PNDS repository required')
 commit=math.get('commit');require(isinstance(commit,str) and SHA40.fullmatch(commit),
  'Exact PNDS commit required before fitting')
 run_id=math.get('run_id');require(type(run_id) is int and run_id>0,'Exact PNDS run ID required')
 require(isinstance(math.get('lean_sha256'),str) and SHA64.fullmatch(math['lean_sha256']),
  'Registered Lean byte digest required')
 link_path=math.get('source_link_path')
 require(link_path=='foundation/v1/langmath101-source-link.json','Exact reciprocal source-link path required')
 api='https://api.github.com/repos/'+PNDS+'/actions/runs/'+str(run_id)
 run=json.loads(fetch(api))
 require(run.get('id')==run_id and run.get('head_sha')==commit and
  run.get('status')=='completed' and run.get('conclusion')=='success',
  'Registered PNDS run has not succeeded on the exact registered commit')
 require(run.get('repository',{}).get('full_name','').lower()==PNDS.lower(),
  'PNDS run repository differs')
 require(run.get('path','').split('@')[0]=='.github/workflows/langmath101.yml',
  'Registered run is not the LANGMATH101 mathematics workflow')
 jobs=json.loads(fetch(api+'/jobs?per_page=100')).get('jobs',[])
 matches=[job for job in jobs if job.get('name')=='source-reference-and-lean']
 require(len(matches)==1 and matches[0].get('conclusion')=='success',
  'Required PNDS source-reference-and-lean job did not succeed')
 for name in REQUIRED_STEPS:
  steps=[step for step in matches[0].get('steps',[]) if step.get('name')==name]
  require(len(steps)==1 and steps[0].get('conclusion')=='success',
   'PNDS run omitted or failed required step: '+name)
 base='https://raw.githubusercontent.com/'+PNDS+'/'+commit+'/'
 link_raw=fetch(base+link_path);link=json.loads(link_raw)
 if 'source_link_sha256' in math:
  require(math['source_link_sha256']==sha(link_raw),'Registered source-link bytes differ')
 require(link.get('study')=='LANGMATH101','Reciprocal study identity differs')
 require(link.get('whole_model_IEEE_AD')=='OPEN','Reciprocal record must preserve open IEEE/AD scope')
 tac=link.get('tac',{});source_commit=tac.get('commit')
 require(tac.get('repository')==TAC and isinstance(source_commit,str) and SHA40.fullmatch(source_commit),
  'PNDS link must pin exact TAC source repository and commit')
 if 'tac_source_commit' in math:
  require(math['tac_source_commit']==source_commit,'Registered reciprocal TAC source commit differs')
 pins=tac.get('source_sha256')
 require(isinstance(pins,dict) and all(PREFIX+p in pins for p in REQUIRED_PINNED_FILES),
  'PNDS link omitted required candidate/reference/Lean source pins')
 tac_base='https://raw.githubusercontent.com/'+TAC+'/'+source_commit+'/'
 for path,expected in pins.items():
  safe_path(path)
  require(isinstance(expected,str) and SHA64.fullmatch(expected),'Invalid reciprocal source digest')
  actual=local_source(study,path)
  require(sha(actual)==expected,'Current source differs from PNDS-pinned source: '+path)
  require(fetch(tac_base+path)==actual,'Historical TAC source bytes differ: '+path)
 copies=link.get('reciprocal_copies',{})
 require(isinstance(copies,dict) and all(copies.get(k)==v for k,v in REQUIRED_RECIPROCAL.items()),
  'Required reciprocal Lean/reference copies omitted or redirected')
 checked={}
 for source,destination in copies.items():
  require(source in pins,'Reciprocal copy lacks source digest')
  safe_path(destination);actual=local_source(study,source)
  require(fetch(base+destination)==actual,'Reciprocal PNDS bytes differ: '+destination)
  checked[destination]=sha(actual)
 require(checked['PNDS/LANGMATH101.lean']==math['lean_sha256'],
  'Registered Lean digest differs from checked source and PNDS bytes')
 return {'repository':PNDS,'commit':commit,'run_id':run_id,'run_conclusion':'success',
  'tac_source_commit':source_commit,'source_link_sha256':sha(link_raw),
  'reciprocal_sha256':checked,'source_pins_checked':len(pins),
  'whole_model_IEEE_AD':'OPEN','independently_qualified':False}


def verify_local_profile(C,register):
 require(register.get('fit_enabled') is True,'Registration has not enabled fitting')
 manifest=C.read(C.HERE/'FAIR_COMPARISON.json')
 profile_path=C.ROOT/manifest['operator_profile_path'];raw=profile_path.read_bytes()
 require(sha(raw)==manifest['operator_profile_sha256'],'Frozen profile file digest differs')
 profile=json.loads(raw)
 require(profile.get('source_hashes')==C.source_hashes(),'Actual executable source differs from frozen profile')
 recipe_sha=C.digest(C.RECIPE)
 require(profile.get('training_recipe')==C.RECIPE and
  profile.get('training_recipe_sha256')==manifest.get('training_recipe_sha256')==recipe_sha,
  'Actual helper/fit recipe differs from frozen profile')
 shared=C.ROOT/'src/tac_osm/research_owner_two_lane.py'
 require(shared.is_file() and not shared.is_symlink() and
  shared.read_bytes()==(C.HERE/'owner_two_lane.py').read_bytes(),
  'Generic dispatcher helper differs from the actual profiled study helper')
 report=C.local('owner_two_lane').validate_program(manifest,'preflight',root=C.ROOT)
 require(report.get('eligible_for_architecture_superiority') is False,
  'Internal split lanes must preserve no-superiority qualification')
 if 'operator_profile_sha256' in register:
  require(register['operator_profile_sha256']==sha(raw),'Registration points to another profile')
 return {'operator_profile_sha256':sha(raw),'implementation_sha256':profile['implementation_sha256'],
  'training_recipe_sha256':recipe_sha,'shared_helper_sha256':sha(shared.read_bytes()),
  'status':report['status']}


def main():
 require(len(sys.argv)==1,'Run readiness.py with no arguments')
 spec=importlib.util.spec_from_file_location('langmath101_readiness_common',HERE/'common.py')
 C=importlib.util.module_from_spec(spec);sys.modules[spec.name]=C;spec.loader.exec_module(C)
 register=C.read(HERE/'REGISTER.json')
 local=verify_local_profile(C,register)
 math=verify_remote_math(register)
 print(json.dumps({'study_id':'LANGMATH101','status':'PASS_EXACT_REGISTERED_READINESS',
  'profile':local,'math':math,'protected_TEST_permitted':False,
  'independently_qualified':False,'whole_model_IEEE_AD':'OPEN'},sort_keys=True))
 return 0


if __name__=='__main__':
 try:sys.exit(main())
 except Exception as exc:
  print('LANGMATH101_READINESS_HARD_FAILURE: '+str(exc),file=sys.stderr)
  sys.exit(2)
