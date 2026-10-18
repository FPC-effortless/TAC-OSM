"""Publish A evidence and learned forecasts; omit unopened reserved B payload."""
import argparse
import json
from pathlib import Path
import tarfile
from data_real_maintenance_002 import digest,write


def package(directory,exports,smoke,destination):
    root=Path(directory);out=Path(destination);assert json.loads((root/'summary.json').read_text())['status']=='COMPLETE';assert not json.loads((root/'_datasets/seal-state.json').read_text())['B_metrics_opened'];archives={}
    def filtered(info):return None if info.name.endswith('/_datasets/sealed-b.pt') else info
    for filename,path in [('evidence.tar.gz.part000',root),('learned-agent.tar.gz',Path(exports)),('smoke-only.tar.gz',Path(smoke))]:
        with tarfile.open(out/filename,'w:gz') as stream:stream.add(path,arcname=path.name,filter=filtered)
        assert (out/filename).stat().st_size<100*1024*1024,'Split larger archives before publishing';archives[filename]=digest(out/filename)
    write(out/'evidence-index.json',{'study':'real-maint-002','archives':archives,'evidence_parts':['evidence.tar.gz.part000'],'run':json.loads((root/'summary.json').read_text()),'decision':'NO_PROMOTION; original registered gates retained','reserved_B_payload_excluded':True,'reserved_B_sha256':json.loads((root/'_datasets/manifest.json').read_text())['data_hashes']['sealed-b'],'scope':'All80 fitted models,8 validation pilots, raw/EMA/selected weights, train/validation/A data, errors, curves, selection and source hashes. B payload omitted; regenerate by fixed preprocessing of verified public source for a separate future registration.'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);p.add_argument('--exports',required=True);p.add_argument('--smoke',required=True);p.add_argument('--destination',default='research/real-maint-002');a=p.parse_args();package(a.directory,a.exports,a.smoke,a.destination)
