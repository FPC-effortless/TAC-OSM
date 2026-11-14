"""Constructed software falsifiers, not language or memory capability evidence."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import pytest
from tac_osm.causal_byte_corpus import (
    ByteCorpusError,ByteDocument,ByteCarry,load_byte_corpus,causal_byte_chunks,
    model_input_values,incoming_byte_state,outgoing_byte_state,
)


def sha(raw):return hashlib.sha256(raw).hexdigest()

def entry(path,raw,role=None):
    r={'path':path,'bytes':len(raw),'sha256':sha(raw),'git_blob':hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()}
    if role is not None:r['role']=role
    return r


def fixture(tmp_path):
    root=tmp_path/'source';root.mkdir()
    docs=[('a.rst',bytes(range(256)),'FIT'),('b.rst','é🙂'.encode(),'DEV')]
    (root/'LICENSE').write_bytes(b'fixture license')
    for p,b,_ in docs:(root/p).write_bytes(b)
    m={'schema':'tac-osm.causal-byte-corpus.v1','DEV_exposure':'PREVIOUSLY_EXPOSED_NOT_TEST','license':entry('LICENSE',b'fixture license'),'files':[entry(*row) for row in docs]}
    path=tmp_path/'manifest.json'
    def save():
        path.write_text(json.dumps(m));return sha(path.read_bytes())
    return root,path,m,save

@pytest.mark.parametrize('size',[1,2,7,128,255,256,512])
def test_complete_pair_coverage_and_no_cross_document(tmp_path,size):
    root,path,m,save=fixture(tmp_path);docs=load_byte_corpus(root,path,expected_manifest_sha256=save())
    for role in ('FIT','DEV'):
        selected=[d for d in docs if d.role==role]
        chunks=list(causal_byte_chunks(docs,role=role,chunk_length=size))
        assert sum(map(lambda c:len(c.targets),chunks))==sum(len(d.raw)-1 for d in selected)
        for d in selected:
            parts=[c for c in chunks if c.document==d.path]
            assert tuple(v for c in parts for v in c.inputs)==tuple(d.raw[:-1])
            assert tuple(v for c in parts for v in c.targets)==tuple(d.raw[1:])
            assert [c.input_offset for c in parts]==list(range(0,len(d.raw)-1,size))
            assert sum(c.reset_state for c in parts)==1
            assert all(c.stream_id==d.stream_id for c in parts)


def test_minimum_doc_and_no_artificial_eos():
    d=ByteDocument('tiny','FIT','FIT',sha(b'ab'),b'ab')
    c=list(causal_byte_chunks((d,),role='FIT',chunk_length=128))
    assert len(c)==1 and c[0].inputs==(97,) and c[0].targets==(98,)


def test_carry_identity_reset_and_wrong_offset_owner(tmp_path):
    root,path,m,save=fixture(tmp_path);docs=load_byte_corpus(root,path,expected_manifest_sha256=save())
    chunks=list(causal_byte_chunks(docs,role='FIT',chunk_length=2));state=object()
    assert incoming_byte_state(chunks[0],ByteCarry('old',200,object())) is None
    carry=outgoing_byte_state(chunks[0],state)
    assert incoming_byte_state(chunks[1],carry) is state
    for bad in [None,replace(carry,stream_id='other'),replace(carry,next_input_offset=9),replace(carry,state=None)]:
        with pytest.raises(ByteCorpusError):incoming_byte_state(chunks[1],bad)
    with pytest.raises(ByteCorpusError):incoming_byte_state(chunks[2],carry)
    with pytest.raises(ByteCorpusError):outgoing_byte_state(chunks[0],None)
    dev=next(causal_byte_chunks(docs,role='DEV',chunk_length=2))
    assert incoming_byte_state(dev,carry) is None


def test_target_poison_excluded_from_input_projection(tmp_path):
    root,path,m,save=fixture(tmp_path);docs=load_byte_corpus(root,path,expected_manifest_sha256=save())
    c=next(causal_byte_chunks(docs,role='FIT',chunk_length=4))
    altered=replace(c,targets=tuple(255-v for v in c.targets))
    assert model_input_values(c)==model_input_values(altered)
    assert set(model_input_values(c))=={'tokens','stream_ids'}
    # Per-position input slice remains fixed when only that next-byte label changes.
    raw=b'ABCDE';changed=b'ABXDE';t=1
    assert raw[:t+1]==changed[:t+1] and raw[t+1]!=changed[t+1]

@pytest.mark.parametrize('change',['hash','length','blob','role','path','duplicate','cross_content','license'])
def test_manifest_or_source_tamper(tmp_path,change):
    root,path,m,save=fixture(tmp_path)
    if change=='hash':m['files'][0]['sha256']='0'*64
    elif change=='length':m['files'][0]['bytes']+=1
    elif change=='blob':m['files'][0]['git_blob']='0'*40
    elif change=='role':m['files'][0]['role']='TEST'
    elif change=='path':m['files'][0]['path']='../a.rst'
    elif change=='duplicate':m['files'].append(m['files'][0].copy())
    elif change=='cross_content':
        raw=(root/'a.rst').read_bytes();(root/'b.rst').write_bytes(raw);m['files'][1]=entry('b.rst',raw,'DEV')
    elif change=='license':(root/'LICENSE').write_bytes(b'changed')
    with pytest.raises(ByteCorpusError):load_byte_corpus(root,path,expected_manifest_sha256=save())


def test_raw_pin_symlink_and_manifest_duplicates(tmp_path):
    root,path,m,save=fixture(tmp_path);pin=save()
    with pytest.raises(ByteCorpusError):load_byte_corpus(root,path,expected_manifest_sha256='0'*64)
    original=(root/'a.rst').read_bytes();(root/'a.rst').unlink();(root/'a.rst').symlink_to(root/'b.rst')
    with pytest.raises(ByteCorpusError,match='symlink'):load_byte_corpus(root,path,expected_manifest_sha256=pin)
    (root/'a.rst').unlink();(root/'a.rst').write_bytes(original)
    text=path.read_text().replace('"schema":','"schema":"duplicate", "schema":')
    path.write_text(text)
    with pytest.raises(ByteCorpusError,match='duplicate'):load_byte_corpus(root,path,expected_manifest_sha256=sha(path.read_bytes()))

@pytest.mark.parametrize('role',['TEST','FIT_SUPPLEMENT','test','',None])
def test_unauthorized_requested_role(role):
    with pytest.raises(ByteCorpusError):list(causal_byte_chunks((),role=role,chunk_length=128))

@pytest.mark.parametrize('size',[0,-1,8193,True,1.5,None])
def test_invalid_chunk_size(size):
    with pytest.raises(ByteCorpusError):list(causal_byte_chunks((),role='FIT',chunk_length=size))


def test_mutated_document_and_duplicate_identity():
    d=ByteDocument('a','FIT','FIT',sha(b'ab'),b'ab')
    for docs in [(replace(d,raw=b'ac'),),(d,d)]:
        with pytest.raises(ByteCorpusError):list(causal_byte_chunks(docs,role='FIT',chunk_length=2))


def test_role_and_carry_metadata_tamper(tmp_path):
    root,path,m,save=fixture(tmp_path);docs=load_byte_corpus(root,path,expected_manifest_sha256=save())
    chunks=list(causal_byte_chunks(docs,role='FIT',chunk_length=1))
    carry=outgoing_byte_state(chunks[0],object())
    with pytest.raises(ByteCorpusError):incoming_byte_state(replace(chunks[1],role='DEV'),carry)
    with pytest.raises(ByteCorpusError):incoming_byte_state(chunks[1],replace(carry,next_input_offset=True))
    with pytest.raises(ByteCorpusError):list(causal_byte_chunks((replace(docs[0],role='DEV'),),role='DEV',chunk_length=1))


def test_parent_directory_symlink_and_file_budget(tmp_path,monkeypatch):
    import tac_osm.causal_byte_corpus as module
    root,path,m,save=fixture(tmp_path);pin=save()
    alias=tmp_path/'alias';alias.symlink_to(root,target_is_directory=True)
    with pytest.raises(ByteCorpusError,match='symlink'):load_byte_corpus(alias,path,expected_manifest_sha256=pin)
    monkeypatch.setattr(module,'MAX_FILE',100)
    with pytest.raises(ByteCorpusError,match='budget'):load_byte_corpus(root,path,expected_manifest_sha256=pin)
