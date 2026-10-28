#!/usr/bin/env python3
"""Gate original Git archive/source bytes, then replicate exposed-DEV TEMP021.

BLOCKED when originals are not supplied; separate CI is internal replication,
never independent qualification or a new untouched cohort.
"""
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from tac_osm.original_archive_shards import restore_original,file_sha
from tac_osm.replay_intake import safe_file

PREFIX='research/replay-bundles/TEMP021/'
PROTOCOL_SHA='4c7703dca3c5920055086687e07c24098766a2301b773cfb99c34f0a394a4b32'
RESULT_SHA='4c197b5623b6dca74f7c8a25ba6931b7d83b25b1198a66898458860c672c34cd'


class MissingOriginal(ValueError):pass


def compare_receipts(actual,original,path='root'):
    """Native NLL is separately guarded at1e-6 against original checkpoint.

    Here use absolute1e-6 for NLL/scalars/Brier/bits and relative1e-6 for PPL;
    counts, roles, identities and top-k accuracies remain exact. Not bit replay.
    """
    if type(actual)!=type(original):raise ValueError('receipt type differs: '+path)
    if isinstance(original,dict):
        if actual.keys()!=original.keys():raise ValueError('receipt fields differ: '+path)
        for key in original:compare_receipts(actual[key],original[key],path+'/'+key)
    elif isinstance(original,list):
        if len(actual)!=len(original):raise ValueError('receipt coverage differs: '+path)
        for i,(x,y) in enumerate(zip(actual,original)):compare_receipts(x,y,path+'/'+str(i))
    elif isinstance(original,float):
        tolerance=0. if path.endswith(('/top1','/top5')) else 1e-6
        relative=1e-6 if path.endswith('/ppl') else 0.
        if not math.isfinite(actual) or not math.isfinite(original) or not math.isclose(actual,original,abs_tol=tolerance,rel_tol=relative):
            raise ValueError('receipt numerical difference: '+path)
    elif actual!=original:raise ValueError('receipt identity differs: '+path)


def execute(root,output):
    root,output=Path(root),Path(output);output.mkdir(parents=True,exist_ok=True)
    proposal='research/phone_queue/temperature022-replay.json'
    plan=subprocess.check_output([sys.executable,str(root/'scripts/phone_lab.py'),'plan',proposal],cwd=root,text=True)
    (output/'COMPUTE_DECISION.json').write_text(plan)
    if json.loads(plan)!=json.loads((root/'research/mathematical-engineering/temperature022/REPLAY_COMPUTE_DECISION.json').read_text()):
        raise ValueError('preregistered replay compute decision differs')
    tracked=set(subprocess.check_output(['git','-C',str(root),'ls-tree','-rz','--name-only','HEAD']).decode().rstrip('\0').split('\0'))
    def original(relative):
        path=safe_file(root,PREFIX+relative)
        if PREFIX+relative not in tracked or not path.is_file():
            raise MissingOriginal('committed original unavailable: '+PREFIX+relative)
        return path
    try:
        rights=json.loads(original('SOURCE_RIGHTS.json').read_text())
        if (rights.get('publisher_url')!='https://github.com/vim/vim'
            or rights.get('source_manifest_sha256')!=PROTOCOL_SHA
            or rights.get('redistribution_reviewed') is not True
            or not isinstance(rights.get('license'),str) or not rights['license'].strip()):
            raise ValueError('reviewed Vim source/license record incomplete')
        if file_sha(original('VIM-LICENSE.txt'))!=rights.get('license_sha256'):
            raise ValueError('original license bytes differ')
        expected_path=original('ORIGINAL_RESULTS.json')
        if file_sha(expected_path)!=RESULT_SHA:raise ValueError('original full TEMP021 receipt differs')
        expected=json.loads(expected_path.read_text())
        manifest=original('original-archive/MANIFEST.json')
        shard_manifest=json.loads(manifest.read_text())
        for entry in shard_manifest['chunks']:
            # Only canonical numbered shard paths are accepted by restoration.
            original('original-archive/'+entry['path'])
        with tempfile.TemporaryDirectory(prefix='temp021-git-replay-') as directory:
            archive=Path(directory)/'original.zip'
            restored=restore_original(manifest.parent,archive)
            with zipfile.ZipFile(archive) as z:
                raw=z.read('tac_language_500k/SOURCE_PARTITION_MANIFEST.json')
            import hashlib
            if hashlib.sha256(raw).hexdigest()!=PROTOCOL_SHA:raise ValueError('original protocol differs')
            protocol=json.loads(raw)['source'];parts=protocol['doc_partitions']
            fit,dev=parts['fit'],parts['dev'];names=fit+dev
            if len(fit)!=30 or len(dev)!=6 or len(set(names))!=36:
                raise ValueError('original source partition identities differ')
            for name in names:
                if Path(name).name!=name:raise ValueError('noncanonical original document name')
                if file_sha(original('source/'+name))!=protocol['byte_sha256'][name]:
                    raise ValueError('original Vim document differs: '+name)
            script=root/'research/mathematical-engineering/temperature021/replay_original_500k_calibration.py'
            spec=importlib.util.spec_from_file_location('original_temp021_git_replay',script)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            measured=module.replay(archive,root/PREFIX/'source')
            compare_receipts(measured,expected)
            (output/'REPLAY_RESULTS.json').write_text(json.dumps(measured,indent=2,sort_keys=True,allow_nan=False)+'\n')
        report=dict(REPO_COLD_REPLAY='PASS_INTERNAL_NUMERICAL_REPLICATION',archive=restored,
                    original_TEMP021_gate_preserved=measured['gates']['Git_LFS_original_checkpoint_replay'],
                    independent_qualification='NOT_QUALIFIED',protected_TEST=False,new_neural_training=False)
        code=0
    except MissingOriginal as exc:
        report=dict(REPO_COLD_REPLAY='BLOCKED',reason=str(exc),original_model_execution='NOT_RUN',
                    independent_qualification='NOT_QUALIFIED',protected_TEST=False,new_neural_training=False)
        code=2
    (output/'STATUS.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    return code


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    try:sys.exit(execute(ROOT,args.output))
    except (ValueError,OSError,KeyError,TypeError,subprocess.CalledProcessError) as exc:
        args.output.mkdir(parents=True,exist_ok=True)
        (args.output/'STATUS.json').write_text(json.dumps(dict(REPO_COLD_REPLAY='FAIL',reason=str(exc),independent_qualification='NOT_QUALIFIED'),indent=2)+'\n')
        raise
