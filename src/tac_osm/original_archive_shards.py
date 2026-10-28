"""Lossless Git-sized shards of the original language archive; never unpickle."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

ORIGINAL_SHA256='cde9a91be40a4e613975c0bd8caa4c19ac24e3d1d0e993db5cacfbd002fecc32'
CHUNK_BYTES=8*1024*1024
MAX_BYTES=128*1024*1024
SCHEMA='tac-osm.original-archive-shards.v1'


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


def split_original(archive: Path, destination: Path, *, expected_sha=ORIGINAL_SHA256):
    archive,destination=Path(archive),Path(destination)
    if archive.is_symlink() or not archive.is_file():raise ValueError('original regular archive required')
    size=archive.stat().st_size
    if not 0<size<=MAX_BYTES or file_sha(archive)!=expected_sha:
        raise ValueError('original archive size/hash differs')
    if destination.exists() or destination.is_symlink():raise ValueError('destination already exists')
    destination.parent.mkdir(parents=True,exist_ok=True)
    # Do not publish partial output or replace an existing evidence directory.
    with tempfile.TemporaryDirectory(prefix='original-shards-',dir=destination.parent) as temp:
        staging=Path(temp)/'payload';staging.mkdir();records=[];whole=hashlib.sha256()
        with archive.open('rb') as f:
            while data:=f.read(CHUNK_BYTES):
                name=f'archive-{len(records):04d}.part'
                (staging/name).write_bytes(data);whole.update(data)
                records.append(dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
        if sum(x['bytes'] for x in records)!=size or whole.hexdigest()!=expected_sha:
            raise ValueError('original archive changed during preservation')
        manifest=dict(schema=SCHEMA,archive_sha256=expected_sha,archive_bytes=size,
                      chunk_bytes=CHUNK_BYTES,chunks=records)
        (staging/'MANIFEST.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
        staging.rename(destination)
    return manifest


def restore_original(directory: Path, output: Path, *, expected_sha=ORIGINAL_SHA256):
    directory,output=Path(directory),Path(output)
    manifest_path=directory/'MANIFEST.json'
    if manifest_path.is_symlink():raise ValueError('symlink manifest')
    manifest=json.loads(manifest_path.read_text())
    if (set(manifest)!={'schema','archive_sha256','archive_bytes','chunk_bytes','chunks'}
        or manifest['schema']!=SCHEMA or manifest['archive_sha256']!=expected_sha
        or type(manifest['archive_bytes']) is not int or not 0<manifest['archive_bytes']<=MAX_BYTES
        or type(manifest['chunk_bytes']) is not int or manifest['chunk_bytes']!=CHUNK_BYTES):
        raise ValueError('original archive manifest differs')
    records=manifest['chunks'];size=manifest['archive_bytes']
    if not isinstance(records,list) or len(records)!=math.ceil(size/CHUNK_BYTES):
        raise ValueError('shard coverage differs')
    for i,record in enumerate(records):
        expected_size=min(CHUNK_BYTES,size-i*CHUNK_BYTES)
        if (not isinstance(record,dict) or set(record)!={'path','bytes','sha256'}
            or record['path']!=f'archive-{i:04d}.part' or type(record['bytes']) is not int
            or record['bytes']!=expected_size or not isinstance(record['sha256'],str)
            or re.fullmatch('[0-9a-f]{64}',record['sha256']) is None):
            raise ValueError('invalid shard order/path/size/hash')
    if output.exists() or output.is_symlink() or output.resolve().is_relative_to(directory.resolve()):
        raise ValueError('restoration output must be new and outside original shards')
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='original-restore-',dir=output.parent) as temp:
        path=Path(temp)/'archive.zip';whole=hashlib.sha256()
        with path.open('xb') as target:
            for record in records:
                shard=directory/record['path']
                if shard.is_symlink() or not shard.is_file() or shard.stat().st_size!=record['bytes']:
                    raise ValueError('original shard missing or size differs')
                data=shard.read_bytes()
                if hashlib.sha256(data).hexdigest()!=record['sha256']:
                    raise ValueError('original shard hash differs')
                target.write(data);whole.update(data)
        if path.stat().st_size!=size or whole.hexdigest()!=expected_sha:
            raise ValueError('restored original archive hash differs')
        # Hard-link installation refuses a concurrent existing output atomically.
        os.link(path,output)
    return dict(status='ORIGINAL_ARCHIVE_BYTES_RESTORED_NOT_SCORE_REPLAY',
                archive_sha256=expected_sha,archive_bytes=size,shards=len(records))
