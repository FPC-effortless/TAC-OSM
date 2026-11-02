"""Audit canonical gates; never launches a trainer or manufactures receipts."""
import argparse,json,os,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'src'))
from tac_osm.phone_research_ops import decide
from tac_osm.research_reward_runner import verify_reward_attestation,RewardEvidenceError

def run(out):
 reports=[]
 for scale in ('small','medium'):
  path=ROOT/'research/mathematical-engineering'/('langscale001-'+scale)/'FAIR_COMPARISON.json'
  result=subprocess.run([sys.executable,str(ROOT/'scripts/validate_research_parity.py'),'--manifest',str(path),'--phase','preflight'],capture_output=True,text=True)
  try:verify_reward_attestation(json.loads(path.read_text()),'preflight',None,trusted_public_key_hex=os.environ.get('TAC_OSM_REWARD_VERIFIER_PUBLIC_KEY_HEX'))
  except RewardEvidenceError as e:signed={'status':'BLOCKED','reason':str(e)}
  else:raise AssertionError('Absent independent receipt was accepted')
  reports.append({'scale':scale,'canonical_preflight_exit':result.returncode,'canonical_preflight_stdout':result.stdout,'canonical_preflight_stderr':result.stderr,'independent_evidence':signed})
 proposal=json.loads((ROOT/'research/phone_queue/langscale001.json').read_text())
 report={'status':'BLOCKED_NOT_TRAINED','training_launched':False,'optimizer_updates':0,
  'compute_plan':decide(proposal),'preflight':reports,
  'additional_open_items':['Fresh independently qualified validation and TEST contract; do not reuse original DEV for selection',
    'Full forward/CE/backward/AdamW FLOP accounting; partial profiler totals do not suffice',
    'Private offline dual-T4 source staging, observed devices and quota execution grant'],
  'historical_source_order_receipt':json.loads((HERE/'evidence/SOURCE_ORDER.json').read_text()),
  'formal_execution':json.loads((HERE/'evidence/REMOTE_EXECUTION.json').read_text()),
  'reciprocal_check':json.loads((out.parent/'RECIPROCAL.json').read_text()) if (out.parent/'RECIPROCAL.json').exists() else {'status':'NOT_RECHECKED_THIS_INVOCATION'},
  'public_key_injected':bool(os.environ.get('TAC_OSM_REWARD_VERIFIER_PUBLIC_KEY_HEX'))}
 out.write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({'status':report['status'],'training_launched':False,'optimizer_updates':0}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.out)
