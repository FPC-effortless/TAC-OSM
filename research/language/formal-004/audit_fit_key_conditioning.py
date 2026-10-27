#!/usr/bin/env python3
"""TAC-LANG-REPAIR-016 source-verified FIT-only checkpoint key conditioning.

Research numerical audit, NOT protected TEST, not a prediction score,
and not a proposed candidate checkpoint comparison. Does not fit/update
weights and fails closed if the original Vim source or model bytes change.

Usage:
 python audit_fit_key_conditioning.py --archive-dir /path/to/tac_lang_gate003 \
  --vim-root /usr/share/vim/vim91/doc --output fit_key_conditioning.json

The archive directory is the *extracted original* 10-checkpoint experiment,
not an implicit path to a public/GitHub-hosted replay archive.
"""
import argparse,hashlib,json,pathlib,sys
import torch

def run(source_dir:pathlib.Path,vim_root:pathlib.Path):
    protocol=json.loads((source_dir/"PROTOCOL.json").read_text())
    results=json.loads((source_dir/"RESULTS.json").read_text())
    from pathlib import Path
    for name,want in protocol["source"]["byte_sha256"].items():
        digest=hashlib.sha256((vim_root/name).read_bytes()).hexdigest()
        if digest!=want:
            raise ValueError(f"Vim chapter source mismatch: {name}")
    sys.path.insert(0,str(source_dir))
    from parallel_tac_language_gate import ParallelTACLanguage
    from tac_language_probe import ascii_doc
    names=protocol["source"]["doc_partitions"]["fit"][:4]
    data=[ascii_doc((vim_root/n).read_bytes()) for n in names]
    seqs=[]
    for data_doc in data:
        for offset in (0,512,2048,4096):
            if len(data_doc)>=offset+128:
                seqs.append(torch.tensor(data_doc[offset:offset+128],dtype=torch.long))
    xx=torch.stack(seqs)
    torch.set_num_threads(1)
    reports=[]
    with torch.no_grad():
        for record in results["runs"]:
            mode=record["mode"]
            filename=record["checkpoint_file"]
            path=source_dir/filename
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != record["checkpoint_sha256"]:
                raise ValueError(f"checkpoint mismatch {filename}")
            checkpoint=torch.load(path,map_location="cpu",weights_only=True)
            model=ParallelTACLanguage(decoder=75,position_mode="multiscale",gate_mode=mode).eval()
            model.load_state_dict(checkpoint["state"],strict=True)
            batch,seq=xx.shape
            features=torch.tanh(model.embedding(xx)+model.time_proj(
                model.time_features(batch,seq,model.embedding.weight.dtype,xx.device)))
            raw_native=model.key_native(features)
            raw_value=model.value_native(features)
            phi,_=model.keys(batch,seq,xx.device,features.dtype,0)
            raw_aligned=torch.einsum("bti,hij->bthj",phi,model.rotation)
            def measure(tensor):
                norm=tensor.norm(dim=-1).flatten()
                return {
                    "min":float(norm.min()),"p01":float(torch.quantile(norm,.01)),
                    "median":float(norm.median()),"below_0p01":int((norm<.01).sum()),
                    "below_1e_6":int((norm<1e-6).sum())
                }
            singular_values=torch.linalg.svdvals(model.rotation)
            reports.append({
                "mode":mode,"seed":record["seed"],"checkpoint_sha256":digest,
                "native":measure(raw_native),"value":measure(raw_value),
                "aligned":measure(raw_aligned),
                "rotation_min_singular":float(singular_values.min())
            })
    return {
        "scope":"Real Vim 9.1 FIT chapter pre-normalization key diagnostic ONLY",
        "all_36_source_chapters_sha256_verified":True,
        "model_architecture":"original 100680 parameters, no training or mutation",
        "training_or_test_access":"none",
        "source_documents":names,
        "per_document_offsets":[0,512,2048,4096],
        "sequence_length":128,
        "token_positions_per_checkpoint":int(xx.numel()),
        "checkpoint_count":len(reports),
        "checkpoint_measurements":reports,
    }

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--archive-dir",type=pathlib.Path,required=True)
    a.add_argument("--vim-root",type=pathlib.Path,required=True)
    a.add_argument("--output",type=pathlib.Path,required=True)
    options=a.parse_args()
    report=run(options.archive_dir,options.vim_root)
    options.output.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    print(f"FIT-only key conditioning verified: {report['checkpoint_count']} checkpoints")

if __name__=="__main__":
    main()
