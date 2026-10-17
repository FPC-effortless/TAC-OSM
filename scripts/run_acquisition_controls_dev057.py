"""Execute or separately audit the complete DEV-057 reference-control matrix."""
import argparse
import json
from pathlib import Path
from run_learned_stack_dev055 import ROOT,write,digest,git
from tac_osm.development.acquisition_controls import measure

PROTOCOL=ROOT/'research/dev057/protocol.json'


def run(directory,audit=False):
    p=json.loads(PROTOCOL.read_text());out=Path(directory)
    if not audit:
        if git('status','--porcelain','--untracked-files=no'):raise RuntimeError('commit source first')
        out.mkdir(parents=True,exist_ok=False)
    if audit:
        summary=json.loads((out/'summary.json').read_text())
        assert summary['protocol_sha256']==digest(PROTOCOL)
        for name,sha in json.loads((out/'manifest.json').read_text()).items():assert digest(out/name)==sha
        old=json.loads((out/'cells.json').read_text())
    cells=[]
    for seed in p['seeds']:
        for history in p['history_events']:
            for queries in p['queries']:
                for arm in p['arms']:cells.append(measure(seed,history,queries,arm))
    fixed=[c for c in cells if c['arm'] in ('equal_bits_scalar','hash_direct','hash_two_way')]
    assert all(c['payload_bytes']==768 for c in fixed)
    valid=all(c['success']==1 and c['work']['build_input_reads']==c['history'] and
              c['work']['record_probes']==c['queries']*(8 if c['arm']=='indexed_acquisition' else c['history'])
              for c in cells if c['arm'] in ('full_scan','indexed_acquisition'))
    if audit:
        def normalized(c):return {k:v for k,v in c.items() if k not in ('build_seconds','query_seconds')}
        # JSON normalizes acquired tuples to lists in the saved record.
        assert json.loads(json.dumps([normalized(c) for c in cells]))==[normalized(c) for c in old]
        record={'status':'PASS','cells':len(cells),'decision_reproduced':valid,'source':git('rev-parse','HEAD'),
                'limitation':'Same author/shared generator; timings excluded. API-level work, not CPU instructions.'}
        write(out/'audit.json',record);print(json.dumps(record));return
    write(out/'cells.json',cells)
    record={'study':'DEV-057','status':'COMPLETE_UNAUDITED','source':git('rev-parse','HEAD'),
            'protocol_sha256':digest(PROTOCOL),'cells':len(cells),'index_mechanism_gate':valid,
            'decision':'BOUNDED_HAND_CODED_INDEXING_REFERENCE' if valid else 'REFERENCE_GATE_FAILED',
            'scope':'No learned challenger; cannot establish TAC-OSM C5 or memory novelty.'}
    write(out/'summary.json',record)
    write(out/'manifest.json',{f.name:digest(f) for f in sorted(out.iterdir()) if f.is_file()})
    print(json.dumps(record))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);parser.add_argument('--audit',action='store_true')
    args=parser.parse_args();run(args.output,args.audit)
