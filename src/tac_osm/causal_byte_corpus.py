"""Local, hash-pinned raw-byte documents and explicit recurrent chunk ownership.

No tokenizer/neural import, model call, optimizer, source acquisition or TEST.
Integer source offsets are not wall-clock claims. A downstream model still needs
its own causal graph/masking and source-bound numeric checkpoint validation.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

MAX_FILE = 4*1024*1024
MAX_TOTAL = 16*1024*1024


class ByteCorpusError(ValueError):
    pass


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _path(name):
    if type(name) is not str or not name or '\\' in name or '\x00' in name:
        raise ByteCorpusError('invalid_source_path')
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or p.as_posix() != name:
        raise ByteCorpusError('invalid_source_path')
    return p


def _pin(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ByteCorpusError('external_SHA256_pin_required')


def _unique(pairs):
    result = {}
    for key,value in pairs:
        if key in result:
            raise ByteCorpusError('duplicate_manifest_key')
        result[key] = value
    return result


def _read(path, budget):
    # Quiescent owned filesystem required; this is not an atomic hostile-FS guard.
    if any(p.is_symlink() for p in (path,*path.parents)):
        raise ByteCorpusError('source_symlink')
    with path.open('rb') as f:
        raw = f.read(budget+1)
    if len(raw)>budget:
        raise ByteCorpusError('source_byte_budget')
    return raw


def _source(root, entry):
    name = _path(entry['path'])
    raw = _read(root/name,MAX_FILE)
    blob = hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    if (type(entry['bytes']) is not int or len(raw)!=entry['bytes'] or
        _sha(raw)!=entry['sha256'] or blob!=entry['git_blob']):
        raise ByteCorpusError('original_source_identity_mismatch')
    return raw


@dataclass(frozen=True)
class ByteDocument:
    path: str
    role: str
    source_role: str
    sha256: str
    raw: bytes

    @property
    def stream_id(self):
        # Opaque ownership metadata, never a learned/token feature.
        return json.dumps([self.role,self.path,self.sha256],separators=(',',':'))


def load_byte_corpus(root, manifest_path, *, expected_manifest_sha256):
    """Load only explicit pinned FIT/DEV files and LICENSE; no enumeration/network.

    Caller freezes raw manifest SHA separately. Hashes bind supplied bytes, not
    independent publisher authority, rights or semantic truth. No normalization.
    """
    _pin(expected_manifest_sha256)
    raw = _read(Path(manifest_path).absolute(),1024*1024)
    if _sha(raw)!=expected_manifest_sha256:
        raise ByteCorpusError('manifest_pin_mismatch')
    manifest = json.loads(raw,object_pairs_hook=_unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ByteCorpusError('nonfinite_manifest')))
    if (type(manifest) is not dict or manifest.get('schema')!='tac-osm.causal-byte-corpus.v1' or
        manifest.get('DEV_exposure')!='PREVIOUSLY_EXPOSED_NOT_TEST' or
        type(manifest.get('files')) is not list or not 1<=len(manifest['files'])<=128):
        raise ByteCorpusError('manifest_schema')
    root = Path(root).absolute()
    license_entry = manifest['license']
    if license_entry['path']!='LICENSE':
        raise ByteCorpusError('license_path')
    _source(root,license_entry)
    docs,paths,hashes,total = [],set(),{},0
    for entry in manifest['files']:
        if set(entry)!={'path','role','sha256','git_blob','bytes'}:
            raise ByteCorpusError('document_schema')
        if entry['role'] not in ('FIT','FIT_SUPPLEMENT','DEV'):
            raise ByteCorpusError('protected_or_unknown_source_role')
        path = entry['path']
        if path in paths:
            raise ByteCorpusError('duplicate_source_identity')
        value = _source(root,entry)
        if len(value)<2:
            raise ByteCorpusError('no_observed_next_byte')
        role = 'DEV' if entry['role']=='DEV' else 'FIT'
        if entry['sha256'] in hashes and hashes[entry['sha256']]!=role:
            raise ByteCorpusError('identical_bytes_cross_partition')
        hashes[entry['sha256']] = role
        total += len(value)
        if total>MAX_TOTAL:
            raise ByteCorpusError('corpus_byte_budget')
        paths.add(path)
        docs.append(ByteDocument(path,role,entry['role'],entry['sha256'],value))
    return tuple(sorted(docs,key=lambda d:d.path))


@dataclass(frozen=True)
class ByteChunk:
    stream_id: str
    role: str
    document: str
    input_offset: int
    inputs: tuple[int,...]
    targets: tuple[int,...]

    @property
    def next_input_offset(self):
        return self.input_offset+len(self.inputs)

    @property
    def reset_state(self):
        return self.input_offset==0


def causal_byte_chunks(documents, *, role, chunk_length):
    """Every next-byte pair once, no concatenation/padding/artificial EOS.

    Inputs cover[a,b); labels cover[a+1,b+1). Final partial chunks are retained.
    Labels are separate from the explicit model-input projection below.
    """
    if role not in ('FIT','DEV'):
        raise ByteCorpusError('protected_or_unknown_requested_role')
    if type(chunk_length) is not int or not 1<=chunk_length<=8192:
        raise ByteCorpusError('chunk_length')
    seen=set()
    for doc in sorted(documents,key=lambda d:d.path):
        if (type(doc) is not ByteDocument or doc.source_role not in ('FIT','FIT_SUPPLEMENT','DEV') or
            doc.role != ('DEV' if doc.source_role=='DEV' else 'FIT')):
            raise ByteCorpusError('document_contract')
        _path(doc.path)
        if doc.path in seen:
            raise ByteCorpusError('duplicate_source_identity')
        seen.add(doc.path)
        if type(doc.raw) is not bytes or len(doc.raw)<2 or _sha(doc.raw)!=doc.sha256:
            raise ByteCorpusError('document_bytes_changed')
        if doc.role!=role:
            continue
        for start in range(0,len(doc.raw)-1,chunk_length):
            stop=min(start+chunk_length,len(doc.raw)-1)
            yield ByteChunk(doc.stream_id,role,doc.path,start,
                            tuple(doc.raw[start:stop]),tuple(doc.raw[start+1:stop+1]))


def _check_chunk(chunk):
    if (type(chunk) is not ByteChunk or type(chunk.input_offset) is not int or chunk.input_offset<0 or
        chunk.role not in ('FIT','DEV') or type(chunk.stream_id) is not str or not chunk.stream_id or
        type(chunk.inputs) is not tuple or type(chunk.targets) is not tuple or not chunk.inputs or
        len(chunk.inputs)!=len(chunk.targets) or
        any(type(v) is not int or not 0<=v<=255 for v in (*chunk.inputs,*chunk.targets))):
        raise ByteCorpusError('chunk_contract')
    try:
        identity = json.loads(chunk.stream_id)
    except ValueError as exc:
        raise ByteCorpusError('stream_identity') from exc
    if (type(identity) is not list or len(identity)!=3 or identity[:2]!=[chunk.role,chunk.document]):
        raise ByteCorpusError('stream_identity')
    _path(chunk.document)
    _pin(identity[2])


def model_input_values(chunk):
    """Explicit input-only values; caller performs local tensor construction.

    This is not model kwargs until tokens are converted to the model's integer
    tensor type. No target/hash/offset/reset key is passed as a learned feature.
    """
    _check_chunk(chunk)
    return {'tokens':chunk.inputs,'stream_ids':(chunk.stream_id,)}


@dataclass(frozen=True)
class ByteCarry:
    stream_id: str
    next_input_offset: int
    state: object


def incoming_byte_state(chunk, previous):
    """Clear state at every document start; otherwise require exact continuation.

    Only metadata sequencing is checked, not numeric/source authenticity of the
    opaque state. Does not copy/cast/detach or modify caller state/AD history.
    """
    _check_chunk(chunk)
    if chunk.reset_state:
        return None
    if (type(previous) is not ByteCarry or type(previous.next_input_offset) is not int or
        previous.stream_id!=chunk.stream_id or
        previous.next_input_offset!=chunk.input_offset or previous.state is None):
        raise ByteCorpusError('missing_or_wrong_document_continuation')
    return previous.state


def outgoing_byte_state(chunk, state):
    _check_chunk(chunk)
    if state is None:
        raise ByteCorpusError('nonempty_model_state_required')
    return ByteCarry(chunk.stream_id,chunk.next_input_offset,state)
