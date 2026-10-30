"""Constructed guards for a no-optimizer profile, never capability evidence."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from capacity_profile import SHAPES,PRECISIONS,atomic,json_safe,sample,shards
from tokenizer_fit import validate_fit_manifest


def test_capacity_shards_cover_exact_matrix_and_observed_worker_seeds():
    jobs=shards()
    assert len(jobs)==2
    seen={(r['batch'],r['context'],r['precision']) for j in jobs for r in j}
    assert seen=={(b,t,p) for b,t in SHAPES for p in PRECISIONS}
    assert sum(map(len,jobs))==20
    for i,j in enumerate(jobs):
        assert {r['seed'] for r in j}=={540001+i}
        assert {r['precision'] for r in j}==set(PRECISIONS)


def test_source_owned_sampling_never_crosses_documents_and_pairs_modes():
    arrays={'a':np.arange(2000,dtype=np.uint16),'b':np.arange(500,dtype=np.uint16)}
    x,y,rows=sample(arrays,4,1024)
    assert x.shape==y.shape==(4,1024)
    assert all(r['array']=='a' for r in rows)
    assert np.array_equal(x.numpy()[:,1:],y.numpy()[:,:-1])
    x2,y2,rows2=sample(arrays,4,1024)
    assert np.array_equal(x.numpy(),x2.numpy()) and rows==rows2
    with pytest.raises(ValueError):sample({'short':np.arange(20,dtype=np.uint16)},2,128)
    with pytest.raises(ValueError):sample({'bad':np.full(200,4096)},1,128)


def test_negative_or_unsupported_cases_still_have_strict_json_evidence(tmp_path):
    value={'status':'FAIL','loss':float('nan'),'metric':np.float64(float('inf')),'count':np.int64(3)}
    p=tmp_path/'failed.json';atomic(p,value)
    stored=json.loads(p.read_text())
    assert stored=={'status':'FAIL','loss':None,'metric':None,'count':3}


def test_portable_fit_manifest_keeps_sha_and_partition_boundary(tmp_path):
    import hashlib
    p=tmp_path/'raw.py';p.write_bytes(b'def f():\r\n\treturn 1\r\n')
    row={'path':'raw.py','partition':'FIT','sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
         'license':'MIT','publisher':'fixture/example','commit':'a'*40}
    manifest=tmp_path/'FIT.json';manifest.write_text(json.dumps({'partition':'FIT','sources':[row]}))
    assert validate_fit_manifest(manifest)==[p.resolve()]
    row['partition']='DEV';manifest.write_text(json.dumps({'partition':'FIT','sources':[row]}))
    with pytest.raises(ValueError):validate_fit_manifest(manifest)


def test_actual_tokenizer_artifacts_and_portable_fit_roles():
    import hashlib
    import sentencepiece as spm
    root=HERE/'tokenizer';receipt=json.loads((root/'bpe4096.receipt.json').read_text())
    assert hashlib.sha256((root/'bpe4096.model').read_bytes()).hexdigest()==receipt['model_sha256']
    assert hashlib.sha256((root/'bpe4096.vocab').read_bytes()).hexdigest()==receipt['vocab_sha256']
    m=json.loads((root/'FIT_MANIFEST.json').read_text())
    assert m['partition']=='FIT' and len(m['sources'])==219
    assert all(s['partition']=='FIT' for s in m['sources'])
    assert receipt['all_FIT_sources_exact_roundtrip'] and receipt['neural_fits']==0
    tokenizer=spm.SentencePieceProcessor(model_file=str(root/'bpe4096.model'))
    assert tokenizer.vocab_size()==4096 and (tokenizer.unk_id(),tokenizer.bos_id(),tokenizer.eos_id(),tokenizer.pad_id())==(0,1,2,3)
    for text in ('def f():\r\n\treturn 1\r\n','  café 🧠\n','日本語とΕλληνικά\n'):
        assert tokenizer.decode(tokenizer.encode(text))==text


@pytest.mark.parametrize('unpacked',[False,True])
def test_profile_bootstrap_verifies_mounts_and_excludes_unlisted_secret(tmp_path,monkeypatch,unpacked):
    import hashlib,runpy,shutil,zipfile
    from package_capacity import package,bootstrap
    corpus=tmp_path/'corpus';corpus.mkdir()
    raw=b'public FIT fixture\n';(corpus/'raw.txt').write_bytes(raw)
    (corpus/'FIT_MANIFEST.json').write_text(json.dumps({'partition':'FIT','sources':[{'path':'raw.txt','partition':'FIT','sha256':hashlib.sha256(raw).hexdigest()}],'license_archives':[]}))
    (corpus/'SOURCE_IDENTITIES.json').write_text('{}')
    (corpus/'kaggle.json').write_text('THIS_MUST_NEVER_BE_PACKAGED')
    tokens=tmp_path/'tokens';tokens.mkdir();np.savez(tokens/'FIT_TOKENS.npz',doc=np.arange(200,dtype=np.uint16))
    (tokens/'TOKENS_RECEIPT.json').write_text('{}')
    stage=tmp_path/'stage';package(corpus,tokens,stage,'fixtureuser',require_clean=False)
    archive=stage/'dataset/bundle.zip';inputs=tmp_path/'input';inputs.mkdir();working=tmp_path/'working';working.mkdir()
    with zipfile.ZipFile(archive) as z:
        assert not any('kaggle.json' in n for n in z.namelist())
        if unpacked:z.extractall(inputs/'dataset')
    if not unpacked:shutil.copy(archive,inputs/'bundle.zip')
    calls=[];monkeypatch.setattr(runpy,'run_path',lambda p:{'run':lambda *a,**k:calls.append((p,a,k))})
    text=bootstrap(archive).replace('/kaggle/input',str(inputs)).replace('/kaggle/working',str(working))
    exec(compile(text,'capacity_bootstrap_test','exec'),{})
    assert Path(calls[0][0]).name=='capacity_profile.py' and calls[0][2]['max_seconds']==300
    assert calls[0][1][1]==working/'LANG10M001_CAPACITY'
    meta=json.loads((stage/'kernel/kernel-metadata.json').read_text())
    assert meta['is_private']=='true' and meta['enable_internet']=='false' and meta['machine_shape']=='NvidiaTeslaT4'
