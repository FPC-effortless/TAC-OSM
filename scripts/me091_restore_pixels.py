"""Restore lawful Git-owned exact input pixels without refitting or re-decoding."""
import argparse,hashlib,json,os,shutil
from pathlib import Path
import numpy as np


def restore(destination):
    root=Path('research/source-bundles/ME091');index=json.loads((root/'SOURCE_INDEX.json').read_text())
    reg=json.loads(Path('research/mathematical-engineering/me091/FAIR_COMPARISON.json').read_text())
    destination=Path(destination)
    for r in index['records']:
        identity=f"{r['task']}:{r['episode']}";contract=reg['visual_source']['records'][identity]
        for field in contract:
            if r[field]!=contract[field]:raise RuntimeError('Registered visual input provenance differs')
        source=root/r['task']/f"{r['episode']:06d}.npz"
        if hashlib.sha256(source.read_bytes()).hexdigest()!=r['cache_sha256']:raise RuntimeError('Git input artifact differs')
        with np.load(source) as z:images=z['images']
        if images.shape!=(r['frames'],1,32,32) or hashlib.sha256(images.tobytes()).hexdigest()!=r['pixel_sha256']:raise RuntimeError('Original training pixels differ')
        target=destination/r['task']/source.name;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest()!=r['cache_sha256']:raise RuntimeError('Never overwrite conflicting inputs')
        else:
            try:os.link(source,target)
            except OSError:shutil.copyfile(source,target)
        target.with_suffix('.json').write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
    (destination/'SOURCE_INDEX.json').write_text(json.dumps(index,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':'PASS','source_revision':index['source_revision'],'exact_observed_episode_inputs':len(index['records']),'pixel_bytes':index['cache_bytes'],'sealed_source_opened':False,'no_fit':True}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,default=Path('/tmp/me091-visual-source'));a=p.parse_args();restore(a.cache)
