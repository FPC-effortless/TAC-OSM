"""Independent joins and checks on the prepared region corpus, no predictions."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from PIL import Image


def audit(root):
    root=Path(root)
    docs={};annotations={};issues=Counter();examples=[];groups=Counter();seen=set()
    with (root/"documents.jsonl").open() as stream:
        for line in stream:
            d=json.loads(line);owner=d["document_id"]
            if owner in docs:raise ValueError("duplicate prepared document ID")
            docs[owner]=d
            a=json.loads((root/d["annotation_path"]).read_text())
            annotations[owner]=a
            size=a.get("meta",{}).get("image_size",{})
            if size!={"width":d["width"],"height":d["height"]}:
                issues["source_image_size_metadata_mismatch"]+=1
                examples.append({"document_id":owner,"issue":"image_dimensions"})
            groups[d["split"]]+=1
    region_count=Counter();char_count=Counter();flag_count=Counter();selected=[];unique_crops=set()
    with (root/"records.jsonl").open() as stream:
        for line in stream:
            r=json.loads(line);sample=r["sample_id"];owner=r["document_id"]
            if sample in seen:raise ValueError("duplicate prepared sample ID")
            seen.add(sample)
            if owner not in docs or r["split"]!=docs[owner]["split"]:
                raise ValueError("orphan or split-crossed crop")
            a=annotations[owner]
            words=a["valid_line"][r["annotation_index"]]["words"]
            expected=" ".join(w["text"] for w in words if str(w.get("row_id"))==r["publisher_row_id"])
            if r["text"]!=expected:raise ValueError("prepared transcription differs from original publisher words/order")
            encoded=r["text"].encode("utf-8")
            if not encoded or hashlib.sha256(encoded).hexdigest()!=r["text_sha256"]:
                raise ValueError("text integrity failure")
            p=(root/r["image_path"]).resolve()
            if not p.is_relative_to(root.resolve()):raise ValueError("escaped crop path")
            raw=p.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=r["image_sha256"]:
                raise ValueError("crop content hash failure")
            with Image.open(p) as image:
                box=r["bbox"]
                if image.size!=(box[2]-box[0],box[3]-box[1]):raise ValueError("crop/box size mismatch")
                if image.mode!="RGB":raise ValueError("image encoder needs RGB crops")
            if r["source_image_sha256"]!=docs[owner]["source_image_sha256"]:
                raise ValueError("wrong original image parent")
            region_count[r["split"]]+=1;char_count.update(r["text"]);flag_count.update(r["flags"])
            unique_crops.add(r["image_sha256"])
            if sum(x["split"]==r["split"] for x in selected)<6:selected.append(r)
    result={"schema":"tac-ocr002.independent-prepared-joins.v1","receipt_counts":dict(groups),"region_counts":dict(region_count),
            "unique_sample_ids":len(seen),"unique_crop_hashes":len(unique_crops),"source_dimension_issues":dict(issues),
            "source_word_order_and_text_hash":"PASS_ALL_PREPARED_RECORDS","crop_bytes_size_rgb_and_parent_links":"PASS_ALL_PREPARED_RECORDS",
            "flags":dict(flag_count),"naira_symbol_count":char_count["₦"],"model_accuracy_scored":False,
            "visual_alignment_review":"SAMPLED_12_CONTACT_SHEET_NOT_ALL_REGION_TRUTH",
            "scientific_source_authentication":"NOT_AN_INDEPENDENTLY_SIGNED_ATTESTATION","test_payloads_opened":0,
            "examples":examples[:20]}
    (root/"INDEPENDENT_CHECKS.json").write_text(json.dumps(result,indent=2)+"\n")
    return result,selected


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("root",type=Path)
    args=parser.parse_args();report,_=audit(args.root);print(json.dumps(report,indent=2))
