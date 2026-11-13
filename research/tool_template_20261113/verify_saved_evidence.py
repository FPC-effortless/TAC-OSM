"""Stdlib saved-receipt verifier; no source acquisition/model/audit rerun."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT/'research/evidence/tool_template_20261113'
PINS = {
    'HOSTED_AUDIT.zip':'1ff02f12d2be7ba2977624f57b14f9d8aed418fe939cc48fce4486d9892a41c0',
    'INITIAL_FAILED_CHECKS.zip':'d54972fdcc4729c8f0ffb2506f1af68d0005738af015bc69528024c7075fafb2',
}

def check(condition, message):
    if not condition:
        raise ValueError(message)

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def main():
    for name,pin in PINS.items():
        check(sha((EVIDENCE/name).read_bytes())==pin,'ZIP pin '+name)
    p=json.loads((ROOT/'research/tool_template_20261113/PROTOCOL.json').read_text())
    old=ROOT/'research/evidence/tool_lineage_20261107/HOSTED_AUDIT.zip'
    check(sha(old.read_bytes())==p['frozen_lineage_zip_sha256'],'old source archive')
    with ZipFile(old) as z:
        raw=z.read(p['frozen_lineage_member'])
    check(sha(raw)==p['frozen_lineage_sha256'],'old frozen lineage')
    original={r['record_index']:r for r in map(json.loads,raw.splitlines())}
    with ZipFile(EVIDENCE/'HOSTED_AUDIT.zip') as z:
        raw_result=z.read('audit/RESULTS.json');result=json.loads(raw_result)
        check(raw_result==(ROOT/'research/tool_template_20261113/LOCAL_RESULTS.json').read_bytes(),'local/hosted exact summary')
        for path,pin in result['source_code_sha256'].items():
            check(sha((ROOT/path).read_bytes())==pin,'frozen audit source '+path)
        rows=list(map(json.loads,z.read('audit/REQUEST_KEYS.jsonl').splitlines()))
        check(len(rows)==len(original)==10071,'full source coverage')
        check([r['record_index'] for r in rows]==list(range(1,10072)),'unique full sorted cohort')
        for r in rows:
            prior=original[r['record_index']]
            check(all(r[k]==prior[k] for k in ('raw_line_sha256','disposition','partition')),'immutable source/disposition join')
            check(set(r['keys'])=={'exact','numbers','quotes_numbers'},'all registered methods')
        for method,registered in result['methods'].items():
            groups=defaultdict(list)
            for r in rows:groups[r['keys'][method]].append(r)
            cross=[]
            for key,values in sorted(groups.items()):
                fit=[r['record_index'] for r in values if r['partition']=='FIT']
                dev=[r['record_index'] for r in values if r['partition']=='DEV']
                if fit and dev:cross.append({'key':key,'FIT':fit,'DEV':dev})
            ndev=sum(r['partition']=='DEV' for r in rows)
            affected=sum(len(g['DEV']) for g in cross)
            recomputed={'all_TRAIN_request_groups':len(groups),'accepted_FIT_records':sum(r['partition']=='FIT' for r in rows),'accepted_DEV_records':ndev,'accepted_cross_partition_groups':len(cross),'DEV_records_with_FIT_partner':affected,'DEV_fraction_with_FIT_partner':affected/ndev,'largest_all_TRAIN_group':max(map(len,groups.values())),'cross_partition_groups':cross}
            check(recomputed==registered,'recomputed '+method)
        suites=ET.fromstring(z.read('TESTS.xml')).findall('testsuite')
        check(sum(int(s.attrib['tests']) for s in suites)==59,'test count')
        check(all(int(s.attrib[k])==0 for s in suites for k in ('failures','errors','skipped')),'all tests passed')
        check(z.read('CPU_PLAN.txt')==(ROOT/'research/tool_template_20261113/COMPUTE_DECISION.json').read_bytes(),'CPU plan')
        check(result['source_is_synthetic'] and not result['independently_qualified'] and not result['training_authorized'] and not result['protected_TEST_opened'],'scope')
        check(result['model_executions']==result['optimizer_updates']==result['GPU_minutes']==result['tokenizer_fit_steps']==0,'no neural run')
    with ZipFile(EVIDENCE/'INITIAL_FAILED_CHECKS.zip') as z:
        suites=ET.fromstring(z.read('TESTS.xml')).findall('testsuite')
        check(sum(int(s.attrib['errors']) for s in suites)>0,'initial collection failure remains')
    print('PASS: original ZIPs/source pins, 10071 frozen request joins, all saved overlap summaries, 59 tests and initial collection error; no model or source audit executed')

if __name__=='__main__':
    main()
