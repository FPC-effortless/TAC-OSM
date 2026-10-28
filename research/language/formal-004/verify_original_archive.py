"""Hash-pinned safe local replay of original TAC-LANG-GATE-003 archive.

Requires the user's original ZIP and optionally identical Vim 9.1 manuals.
No copyright text or checkpoint binaries are committed by this verifier.
No fitting, optimizer update or protected evaluation occurs.
"""
import argparse
import hashlib
import io
import json
import pathlib
import subprocess
import sys
import tempfile
import zipfile

import torch

ARCHIVE_SHA256 = "2490e13186d31167a8100d12d709a4c125419a6165ccb8e12f64687abe0c3d11"
ROOT = "tac_lang_gate003/"
EXPECTED_CHECKPOINTS = {(mode, seed) for mode in ("tanh","silu") for seed in range(1701,1706)}

def checked_archive(archive: pathlib.Path):
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("archive is not the pinned original source")
    with zipfile.ZipFile(archive) as z:
        if any(not name.startswith(ROOT) or ".." in pathlib.PurePosixPath(name).parts
               for name in z.namelist()):
            raise ValueError("unsafe archive paths")
        manifest = json.loads(z.read(ROOT+"MANIFEST_SHA256.json"))["files"]
        for name,record in manifest.items():
            b=z.read(ROOT+name)
            if hashlib.sha256(b).hexdigest() != record["sha256"] or len(b) != record["bytes"]:
                raise ValueError("incorrect archive member "+name)
        recorded=json.loads(z.read(ROOT+"RESULTS.json"))
        observed=set()
        for entry in recorded["runs"]:
            mode,seed=entry["mode"],entry["seed"]
            if (mode,seed) in observed:raise ValueError("duplicate model")
            observed.add((mode,seed))
            buf=z.read(ROOT+entry["checkpoint_file"])
            if hashlib.sha256(buf).hexdigest() != entry["checkpoint_sha256"]:
                raise ValueError("checkpoint digest mismatch")
            # Restricted PyTorch load; do not execute arbitrary checkpoint pickle.
            obj=torch.load(io.BytesIO(buf),weights_only=True,map_location="cpu")
            if obj["seed"] != seed or obj["mode"] != mode or obj["learned_parameters"] != 100680:
                raise ValueError("checkpoint geometry mismatch")
            if obj["sample_sha256"] != entry["sample_sha256"]:
                raise ValueError("sample-order hash mismatch")
            if not all(t.isfinite().all() for t in obj["state"].values()):
                raise ValueError("nonfinite model tensors")
            if not all(k in obj for k in ("optimizer","torch_rng","numpy_rng","python_rng")):
                raise ValueError("incomplete optimizer/RNG source")
        if observed != EXPECTED_CHECKPOINTS:raise ValueError("missing original checkpoints")
    return manifest,recorded

def full_source_replay(archive:pathlib.Path,source:pathlib.Path)->None:
    with tempfile.TemporaryDirectory(prefix="tac-lang-gate003-") as root:
        dest=pathlib.Path(root)
        with zipfile.ZipFile(archive) as z:
            for entry in z.infolist():
                if entry.is_dir():continue
                item=pathlib.PurePosixPath(entry.filename)
                filename=dest.joinpath(*item.parts[1:])
                filename.parent.mkdir(parents=True,exist_ok=True)
                content=z.read(entry)
                if item.name=="replay_portable.py":
                    content=content.replace(b"weights_only=False",b"weights_only=True")
                filename.write_bytes(content)
        subprocess.run([sys.executable,str(dest/"replay_portable.py"),
                        "--source",str(source)],check=True,cwd=dest)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--bundle",required=True,type=pathlib.Path)
    parser.add_argument("--source",type=pathlib.Path)
    args=parser.parse_args()
    manifest,recorded=checked_archive(args.bundle)
    print(f"PASS archive {len(manifest)} hashes and {len(recorded['runs'])} restricted-load original checkpoints",flush=True)
    if args.source is not None:
        full_source_replay(args.bundle,args.source)

if __name__=="__main__":main()
