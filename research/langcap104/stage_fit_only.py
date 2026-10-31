"""Independent no-fit staging: authenticated full publisher archive -> FIT-only tar.

The model worker never receives the original full corpus or DEV bytes.
This stage has no model imports, no optimizer, no DEV prediction/score.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"langcap103"))
from source import verify_package,digest,git_blob

def stage(receipt_path,full_archive,out_dir):
    receipt,raw=verify_package(receipt_path,full_archive)
    fit=[r for r in receipt["identity"]["files"] if r["split"]=="FIT"]
    dev=[r for r in receipt["identity"]["files"] if r["split"]=="DEV"]
    if len(fit)!=560 or len(dev)!=18:
        raise ValueError("Frozen publisher FIT/DEV file counts differ")
    out_dir=Path(out_dir)
    if out_dir.exists(): raise FileExistsError("No overwrite of FIT staging evidence")
    out_dir.mkdir(parents=True)
    target=out_dir/"FIT_ONLY.tar"
    with tarfile.open(target,"w",format=tarfile.USTAR_FORMAT) as archive:
        for record in sorted(fit,key=lambda r:r["path"]):
            content=raw[record["path"]]
            if digest(content)!=record["sha256"] or git_blob(content)!=record["git_blob"]:
                raise ValueError("Original FIT file hash differs")
            entry=tarfile.TarInfo(record["path"])
            entry.size=len(content)
            entry.mode=0o444
            entry.uid=0;entry.gid=0;entry.mtime=0
            archive.addfile(entry,io.BytesIO(content))
    with tarfile.open(target) as archive:
        names=archive.getnames()
        if set(names)!={r["path"] for r in fit} or len(names)!=len(fit):
            raise ValueError("FIT-only archive names mismatch")
        if set(names)&{r["path"] for r in dev}:
            raise ValueError("DEV files leaked to trainable archive")
    report={"status":"AUTHENTIC_FIT_ONLY_STAGED_NO_TRAINING",
            "source_sha256":receipt["source_sha256"],
            "archive_sha256":hashlib.sha256(target.read_bytes()).hexdigest(),
            "fit_files":len(fit),"fit_bytes":sum(r["bytes"] for r in fit),
            "original_dev_files_delivered_to_worker":0,
            "model_imported":False,"optimizer_updates":0,"test_access":False}
    (out_dir/"FIT_STAGE.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report))
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--receipt",required=True,type=Path)
    p.add_argument("--full-archive",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    stage(a.receipt,a.full_archive,a.output)
