"""Fail-closed immutable source/math/registry correspondence, not independent science."""
import argparse,hashlib,json,re,urllib.request
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def remote(commit,path):
 if not re.fullmatch('[0-9a-f]{40}',commit):raise ValueError('immutable commit required')
 url='https://raw.githubusercontent.com/FPC-effortless/pnds-lean/'+commit+'/'+path
 with urllib.request.urlopen(url,timeout=30) as r:return r.read()
def run(out):
 m=json.loads((HERE/'SOURCE_MAP.json').read_text());pin=json.loads((HERE/'PNDS_PIN.json').read_text())
 assert pin['repository']=='FPC-effortless/pnds-lean'
 for path,h in m['tac_files_sha256'].items():
  assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==h,path
 for path,h in m['pnds_files_sha256'].items():
  assert hashlib.sha256(remote(pin['formulation_commit'],path)).hexdigest()==h,path
 local=json.loads((ROOT/'research/pnds-variant-math-links.json').read_text())
 peer=json.loads(remote(pin['registry_commit'],'foundation/v1/variant-math-links.json'))
 assert local==peer,'Reciprocal registry differs'
 rows=peer['prospective_studies']['LANGSCALE001']['variants'];assert len(rows)==4
 assert len(peer['variants'])==17
 for row in rows:
  assert row['tac']['commit']==pin['tac_graph_source_commit']
  assert row['pnds']['commit']==pin['formulation_commit']
  assert row['global_ieee_equivalence']=='OPEN' and row['trained_capability_inferred'] is False
 peerpin=json.loads(remote(pin['registry_commit'],'foundation/langscale001/TAC_PIN.json'))
 assert peerpin['commit']==pin['tac_registry_peer_commit']
 assert peerpin['graph_source_commit']==pin['tac_graph_source_commit']
 out.write_text(json.dumps({'status':'PASS_IMMUTABLE_SOURCE_MATH_REGISTRY_CHECKS','pnds_pin':pin,
  'tac_source_files_checked':list(m['tac_files_sha256']),'pnds_formulation_files_checked':list(m['pnds_files_sha256']),
  'historical_variant_rows':17,'new_variant_rows':4,'independent_scientific_qualification':False,'optimizer_updates':0},indent=2)+'\n')
 print('PASS_IMMUTABLE_SOURCE_MATH_REGISTRY_CHECKS_NOT_TRAINING_AUTHORIZATION')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.out)
