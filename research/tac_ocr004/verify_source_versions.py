"""Explicit historical/current source separation; no model code is imported."""
from pathlib import Path, PurePosixPath
import contextlib, hashlib, io, json, tempfile, types, urllib.request

ROOT=Path(__file__).resolve().parents[2]
MANIFEST_SHA256="5145d6c56d6508aa6250951a6a6ada39c98e336ed189744912c8470d9de1b409"
LIMIT=2_000_000

def require(ok,message):
    if not ok: raise ValueError(message)

def digest(raw): return hashlib.sha256(raw).hexdigest()

def local(base,path):
    require(isinstance(path,str) and path and not PurePosixPath(path).is_absolute() and '..' not in PurePosixPath(path).parts,'repository-relative path required')
    p=(Path(base)/path)
    require(p.resolve().is_relative_to(Path(base).resolve()) and p.is_file() and not p.is_symlink(),'local file escapes source root')
    return p.read_bytes()

def fetch_bytes(url):
    with urllib.request.urlopen(url,timeout=30) as response: raw=response.read(LIMIT+1)
    require(len(raw)<=LIMIT,'versioned response exceeds bound')
    return raw

def module_from_snapshot(path,raw,key):
    module=types.ModuleType(key);module.__file__=str(path)
    exec(compile(raw,str(path),'exec'),module.__dict__)
    return module

def verify(base=ROOT,read_remote=fetch_bytes):
    base=Path(base).resolve()
    data=json.loads(local(base,'research/tac_ocr004/SOURCE_VERSIONS.json'))
    require(digest(json.dumps(data,sort_keys=True,separators=(',',':')).encode())==MANIFEST_SHA256,'version boundary manifest drift')
    snapshots={}
    for group in ['unchanged_local_legacy_files','mandatory_current_contract']:
        for path,expected in data[group].items():
            raw=local(base,path);require(digest(raw)==expected,'version boundary local file drift: '+path);snapshots[path]=raw
    def fetch(url):
        raw=read_remote(url)
        require(isinstance(raw,bytes) and len(raw)<=LIMIT,'versioned response exceeds bound')
        return raw
    current=module_from_snapshot(base/'research/tac_ocr004/verify_immutable_peer.py',snapshots['research/tac_ocr004/verify_immutable_peer.py'],'ocr_current_version')
    current_result=current.verify_immutable_peer(base,'tac',json.loads(snapshots['research/tac_ocr004/IMMUTABLE_PEER_PIN.json']),fetch)
    with tempfile.TemporaryDirectory(prefix='ocr001-original-bytes-') as temporary:
        source=Path(temporary)/'source';math=Path(temporary)/'math'
        for target,repository,commit,hashes in [(source,data['historical_source_repository'],data['historical_source_commit'],data['historical_source_sha256']),(math,data['historical_math_repository'],data['historical_math_commit'],data['historical_math_sha256'])]:
            for path,expected in hashes.items():
                raw=fetch('https://raw.githubusercontent.com/'+repository+'/'+commit+'/'+path)
                require(digest(raw)==expected,'historical version bytes drift: '+path)
                p=target/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
        historical_result=current.verify_historical_bytes(source,math)
        # The old validator is unchanged. Its source root is explicit here,
        # rather than silently treating current OCR004 bytes as OCR001.
        p=source/'research/pnds-variant-math-links.json';p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes(local(base,'research/pnds-variant-math-links.json'))
        oldpath=base/'research/tac_ocr001/verify_math_pins.py'
        old=module_from_snapshot(oldpath,snapshots['research/tac_ocr001/verify_math_pins.py'],'ocr_original_version')
        old.ROOT=source
        old.urllib=types.SimpleNamespace(request=types.SimpleNamespace(urlopen=lambda url,timeout:io.BytesIO(fetch(url))))
        output=io.StringIO()
        with contextlib.redirect_stdout(output):old.verify()
        legacy_receipt=json.loads(output.getvalue())
        require(legacy_receipt.get('status')=='SOURCE_PIN_BYTES_MATCH_NOT_PROOF' and len(legacy_receipt.get('candidates',[]))==3,'original validator did not complete')
    result={'status':'VERSIONED_SOURCE_BYTES_NOT_WHOLE_MODEL_PROOF','historical_source_commit':data['historical_source_commit'],'current_source_commit':data['current_source_commit'],'historical_original_validator_executed':True,'historical_source_files':9,'historical_math_files':5,'current_ocr004_mandatory':True,'current_variants':current_result['variants'],'legacy_default_current_source_check_changed':False,'historical_current_source_equality_claimed':False,'whole_model_proved':False,'training_authorized':False}
    print(json.dumps({'versioned_ocr_source_boundary':result},sort_keys=True))
    return result

if __name__=='__main__':verify()
