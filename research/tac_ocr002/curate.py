"""Decoded-pixel identity and explicit train-only filtering, no model scoring."""
from __future__ import annotations

from collections import Counter,defaultdict
import hashlib
import io
import json
from pathlib import Path

from PIL import Image

from .prepare import approved_files


def pixel_audit(raw,root):
    import pyarrow.parquet as pq
    raw,root=Path(raw),Path(root)
    docs=[json.loads(line) for line in (root/"documents.jsonl").open()]
    lookup={(d["source_shard"],d["source_row"]):d for d in docs}
    groups=defaultdict(list)
    for item in approved_files():
        p=pq.ParquetFile(raw/item["path"],memory_map=True)
        for i,batch in enumerate(p.iter_batches(batch_size=1,use_threads=False)):
            r=batch.to_pylist()[0];d=lookup[(item["path"],i)]
            payload=r["image"]["bytes"]
            if hashlib.sha256(payload).hexdigest()!=d["source_image_sha256"]:
                raise ValueError("source image bytes differ from recorded parent")
            original_annotations=r["ground_truth"].encode("utf-8")
            if original_annotations!=(root/d["annotation_path"]).read_bytes():
                raise ValueError("saved publisher annotations changed")
            with Image.open(io.BytesIO(payload)) as im:
                rgb=im.convert("RGB")
                digest=hashlib.sha256(f"RGB:{rgb.width}:{rgb.height}:".encode()+rgb.tobytes()).hexdigest()
            d["decoded_pixel_sha256"]=digest
            groups[digest].append({"document_id":d["document_id"],"split":d["split"]})
    cross=[{"decoded_pixel_sha256":sha,"documents":rows} for sha,rows in groups.items() if len({r["split"] for r in rows})>1]
    temporary=root/"documents.pixels.tmp"
    with temporary.open("w") as stream:
        for d in docs:stream.write(json.dumps(d,ensure_ascii=False)+"\n")
    temporary.replace(root/"documents.jsonl")
    report={"schema":"tac-ocr002.original-pixel-audit.v1","source_image_and_annotation_bytes_verified":len(docs),
            "decoded_pixel_cross_split_groups":len(cross),"cross_split_groups":cross,
            "scope":"encoded-file equality and decoded-pixel equality are different checks","new_optimizer_updates":0}
    (root/"PIXEL_AUDIT.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


def filter_training(root):
    root=Path(root)
    docs={d["document_id"]:d for d in map(json.loads,(root/"documents.jsonl").open())}
    if any("decoded_pixel_sha256" not in d for d in docs.values()):
        raise ValueError("verify decoded publisher pixels before filtering")
    findings=json.loads((root/"duplicate_findings.json").read_text())
    validation_pixels={d["decoded_pixel_sha256"] for d in docs.values() if d["split"]=="validation"}
    overlaps={row[0] for row in findings["cross_split_exact_crops"]}
    conflicts={row[0] for row in findings["conflicting_crop_labels"]}
    reasons=Counter();counts=Counter();validation_bytes=hashlib.sha256()
    with (root/"train_filtered.jsonl").open("wb") as train,(root/"validation_original.jsonl").open("wb") as validation,(root/"training_filter_exclusions.jsonl").open("w") as exclusions:
        for line in (root/"records.jsonl").open("rb"):
            r=json.loads(line)
            if r["split"]=="validation":
                validation.write(line);validation_bytes.update(line);counts["validation_regions_unchanged"]+=1
                continue
            why=[]
            if docs[r["document_id"]]["decoded_pixel_sha256"] in validation_pixels:why.append("same_decoded_receipt_pixels_as_validation")
            if r["image_sha256"] in overlaps:why.append("exact_crop_also_in_validation")
            if r["image_sha256"] in conflicts:why.append("conflicting_transcriptions_for_identical_crop")
            if r["flags"]:why.append("flagged_source_geometry_or_blank_crop")
            counts["original_training_regions"]+=1
            if why:
                counts["excluded_training_regions"]+=1;reasons.update(why)
                exclusions.write(json.dumps({"sample_id":r["sample_id"],"document_id":r["document_id"],"reasons":why})+"\n")
            else:
                train.write(line);counts["retained_training_regions"]+=1
    near=findings["cross_split_near_image_candidates"]
    report={"schema":"tac-ocr002.proposed-train-filter.v1","counts":dict(counts),"overlapping_reason_counts":dict(reasons),
            "validation_original_sha256":validation_bytes.hexdigest(),
            "rules_registered":"AFTER_SOURCE_QA_BEFORE_ANY_MODEL_FIT_OR_PREDICTION","validation_rows_removed_or_reassigned":0,
            "original_records_preserved":True,"remaining_near_image_candidates_for_manual_review":len(near),
            "validation_label_conflicts_preserved_for_adjudication":True,
            "training_preflight_qualified":False,"model_quality_measured":False}
    (root/"TRAIN_FILTER.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--raw",type=Path,required=True);p.add_argument("--prepared",type=Path,required=True)
    a=p.parse_args();print(json.dumps(pixel_audit(a.raw,a.prepared)));print(json.dumps(filter_training(a.prepared),indent=2))
