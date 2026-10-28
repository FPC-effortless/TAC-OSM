"""TEMP023 offline no-unpickle audit of original TEMP022 handoff ZIP.

Never publishes original source or checkpoint weights, never infers rights approval.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tempfile
from zipfile import ZipFile

ROOT="research/replay-bundles/TEMP021/"
ARCHIVE_SHA="cde9a91be40a4e613975c0bd8caa4c19ac24e3d1d0e993db5cacfbd002fecc32"
PROTOCOL_SHA="4c7703dca3c5920055086687e07c24098766a2301b773cfb99c34f0a394a4b32"
RESULT_SHA="4c197b5623b6dca74f7c8a25ba6931b7d83b25b1198a66898458860c672c34cd"
CHUNK_BYTES=8*1024*1024
ARCHIVE_BYTES=84037332

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def check_member_names(infos):
    names=[]
    for entry in infos:
        name=entry.filename
        part=PurePosixPath(name)
        if (name.startswith("/") or "\\" in name or not part.parts
            or any(p in ("","..",".") for p in name.split("/")) or entry.is_dir()
            or entry.file_size>128*1024*1024
            or (entry.external_attr >> 16) & 0o170000 == 0o120000):
            raise ValueError("unsafe path, symlink, directory or oversized member")
        names.append(name)
    if len(set(names))!=len(names):
        raise ValueError("duplicate ZIP member paths")
    return set(names)

def check_manifest(m):
    if (type(m) is not dict
        or set(m)!={"schema","archive_sha256","archive_bytes","chunk_bytes","chunks"}
        or m["schema"]!="tac-osm.original-archive-shards.v1"
        or m["archive_sha256"]!=ARCHIVE_SHA
        or m["archive_bytes"]!=ARCHIVE_BYTES
        or m["chunk_bytes"]!=CHUNK_BYTES
        or not isinstance(m["chunks"],list)
        or len(m["chunks"])!=11):
        raise ValueError("invalid frozen original archive manifest")
    for i,r in enumerate(m["chunks"]):
        expected=min(CHUNK_BYTES,ARCHIVE_BYTES-i*CHUNK_BYTES)
        if (type(r) is not dict or set(r)!={"path","bytes","sha256"}
            or r["path"]!=f"archive-{i:04d}.part"
            or type(r["bytes"]) is not int or r["bytes"]!=expected
            or type(r["sha256"]) is not str or len(r["sha256"])!=64
            or not all(c in "0123456789abcdef" for c in r["sha256"])):
            raise ValueError("missing, misordered, or altered original shard")
    return m["chunks"]

def check_rights(rights,license_bytes,notice_bytes):
    if not (isinstance(rights,dict)
        and rights.get("publisher_url")=="https://github.com/vim/vim"
        and rights.get("source_manifest_sha256")==PROTOCOL_SHA
        and rights.get("license_sha256")==sha(license_bytes)
        and isinstance(rights.get("license"),str)
        and ("Open Publication" in rights["license"] or "OPL" in rights["license"])
        and b"License: OPL-1+" in license_bytes
        and b"Open Publication License" in notice_bytes):
        raise ValueError("unverified source license/notice provenance")
    if rights.get("redistribution_reviewed") is not False:
        raise ValueError("public redistribution approval is not established here")
    return "PUBLICATION_REVIEW_REQUIRED"

def audit(transfer):
    transfer=Path(transfer)
    if transfer.is_symlink() or not transfer.is_file():
        raise ValueError("regular transfer file required")
    if not 0<transfer.stat().st_size<=128*1024*1024:
        raise ValueError("transfer size out of registered bounds")
    with ZipFile(transfer) as bundle:
        names=check_member_names(bundle.infolist())
        def read(relative):
            path=ROOT+relative
            if path not in names:
                raise ValueError("missing original path "+path)
            return bundle.read(path)
        m=json.loads(read("original-archive/MANIFEST.json"))
        shards=check_manifest(m)
        chunk_names={ROOT+"original-archive/"+c["path"] for c in shards}
        receipt=read("ORIGINAL_RESULTS.json")
        if sha(receipt)!=RESULT_SHA:
            raise ValueError("original exact TEMP021 receipt hash changed")
        licence=read("VIM-LICENSE.txt")
        notice=read("UPSTREAM-VIM-LICENSE-NOTICE.txt")
        rights=json.loads(read("SOURCE_RIGHTS.json"))
        rights_status=check_rights(rights,licence,notice)
        with tempfile.TemporaryFile() as restored:
            whole=hashlib.sha256()
            for shard in shards:
                data=read("original-archive/"+shard["path"])
                if len(data)!=shard["bytes"] or sha(data)!=shard["sha256"]:
                    raise ValueError("original archive part drift")
                whole.update(data)
                restored.write(data)
            if restored.tell()!=ARCHIVE_BYTES or whole.hexdigest()!=ARCHIVE_SHA:
                raise ValueError("original byte-for-byte shard restoration failed")
            restored.seek(0)
            with ZipFile(restored) as archive:
                protocol=archive.read("tac_language_500k/SOURCE_PARTITION_MANIFEST.json")
                if sha(protocol)!=PROTOCOL_SHA:
                    raise ValueError("original protocol identity drift")
                source=json.loads(protocol)["source"]
                fit=source["doc_partitions"]["fit"]
                dev=source["doc_partitions"]["dev"]
                if len(fit)!=30 or len(dev)!=6 or len(set(fit+dev))!=36:
                    raise ValueError("original source split changed")
                if sorted(dev)!=sorted(("usr_45.txt","usr_50.txt","usr_51.txt",
                                       "usr_52.txt","usr_90.txt","usr_toc.txt")):
                    raise ValueError("original exposed DEV split changed")
                for seed in range(1701,1706):
                    if f"tac_language_500k/seed_{seed}.pt" not in archive.namelist():
                        raise ValueError("trained original checkpoint missing")
        source_names=set()
        for doc in fit+dev:
            if PurePosixPath(doc).name!=doc:
                raise ValueError("unsafe original chapter identity")
            original=read("source/"+doc)
            if sha(original)!=source["byte_sha256"][doc]:
                raise ValueError("original publisher chapter SHA mismatch: "+doc)
            source_names.add(ROOT+"source/"+doc)
        expected=chunk_names|source_names|{
            ROOT+"original-archive/MANIFEST.json",ROOT+"ORIGINAL_RESULTS.json",
            ROOT+"VIM-LICENSE.txt",ROOT+"UPSTREAM-VIM-LICENSE-NOTICE.txt",
            ROOT+"SOURCE_RIGHTS.json","README_FIRST.txt"}
        if names!=expected:
            raise ValueError("extra or missing bytes in frozen transfer handoff")
    return {"study":"TEMP023_ORIGINAL_TRANSFER_SHA_AND_RIGHTS_AUDIT",
            "evidence":"ORIGINAL_VERIFIED_ARCHIVE_AND_PUBLISHER_SOURCE_BYTES",
            "original_archive_sha256":ARCHIVE_SHA,
            "original_archive_bytes":ARCHIVE_BYTES,
            "original_shard_count":len(shards),
            "original_vim_document_sha256_verified":36,
            "original_frozen_result_sha256":RESULT_SHA,
            "rights_gate":rights_status,
            "public_git_upload":"BLOCKED_PENDING_MODEL_IP_AND_SOURCE_REDISTRIBUTION_REVIEW",
            "git_cold_model_replay":"NOT_RUN",
            "model_inference":"NOT_RUN",
            "protected_TEST":"NOT_ACCESSED"}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transfer",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    result=audit(args.transfer)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():
        raise ValueError("refusing to overwrite prior audit")
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,sort_keys=True))
if __name__=="__main__":
    main()
