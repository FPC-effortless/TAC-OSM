"""Pinned CORD v2 train/validation preparation, with no model fit or scoring.

Only publisher-annotated regions are OCR targets. CORD does not provide a
complete transcription of every visible page, nor Nigerian Pidgin labels.
"""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sqlite3
import statistics
import time

from PIL import Image, ImageStat

HERE=Path(__file__).resolve().parent
SOURCE=json.loads((HERE/"SOURCE.json").read_text())


def sha_file(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def approved_files(source=SOURCE):
    if source["repo_id"]!="naver-clova-ix/cord-v2" or not re.fullmatch(r"[0-9a-f]{40}",source["revision"]):
        raise ValueError("immutable publisher identity required")
    files=[row for row in source["files"] if re.fullmatch(r"data/(train|validation)-[A-Za-z0-9-]+\.parquet",row["path"])]
    if len(files)!=5 or any(not re.fullmatch(r"[0-9a-f]{64}",row.get("lfs_sha256", "")) for row in files):
        raise ValueError("five SHA256-pinned original shards required")
    return files


def verify_originals(raw):
    raw=Path(raw)
    verified=[]
    for item in approved_files():
        path=raw/item["path"]
        if not path.is_file() or path.stat().st_size!=item["size"] or sha_file(path)!=item["lfs_sha256"]:
            raise ValueError("missing/mismatched original publisher shard: "+item["path"])
        verified.append({**item,"verification":"PASS_MATCHES_PINNED_PUBLISHER_FILE"})
    if list(raw.glob("data/test-*.parquet")):
        raise ValueError("this source-only staging directory must not contain TEST payloads")
    return verified


def download_originals(raw):
    from huggingface_hub import snapshot_download
    paths=[row["path"] for row in approved_files()]+["README.md","dataset_infos.json"]
    snapshot_download(repo_id=SOURCE["repo_id"],repo_type="dataset",revision=SOURCE["revision"],
                      allow_patterns=paths,local_dir=str(raw),max_workers=2)
    return verify_originals(raw)


def teacher_forcing(text):
    from research.tac_ocr001.tokenizer import encode
    ids=encode(text)
    return ids[:-1],ids[1:]


def publisher_split_matches(shard_split,annotation_split):
    aliases={"train":{"train"},"validation":{"dev","valid","validation"}}
    return annotation_split is None or annotation_split in aliases[shard_split]


def quad_points(quad):
    if not isinstance(quad,dict) or set(quad)!={f"{axis}{i}" for i in range(1,5) for axis in ("x","y")}:
        raise ValueError("complete publisher quadrilateral required")
    values=list(quad.values())
    if any(type(x) not in (int,float) or not math.isfinite(x) for x in values):
        raise ValueError("finite numeric source coordinates required")
    return [(quad[f"x{i}"],quad[f"y{i}"]) for i in range(1,5)]


def regions(annotation,width,height):
    lines=annotation.get("valid_line")
    if not isinstance(lines,list):
        raise ValueError("publisher valid_line annotation required")
    for line_index,line in enumerate(lines):
        if not isinstance(line,dict) or not isinstance(line.get("words"),list):
            yield {"excluded":True,"annotation_index":line_index,"reason":"missing_words"}
            continue
        groups=OrderedDict()
        for word in line["words"]:
            key=word.get("row_id") if isinstance(word,dict) else None
            # No ordering/geometry inferred from the semantic category.
            groups.setdefault(str(key),[]).append(word)
        for row_id,words in groups.items():
            base={"annotation_index":line_index,"publisher_row_id":row_id}
            try:
                if not words or any(not isinstance(w,dict) or not isinstance(w.get("text"),str) or not w["text"].strip() for w in words):
                    raise ValueError("empty_or_missing_source_word")
                points=[p for w in words for p in quad_points(w.get("quad"))]
                min_x,min_y=min(p[0] for p in points),min(p[1] for p in points)
                max_x,max_y=max(p[0] for p in points),max(p[1] for p in points)
                if max_x<=min_x or max_y<=min_y:
                    raise ValueError("degenerate_source_geometry")
                flags=[]
                if row_id=="None":flags.append("missing_publisher_row_id_field_region_only")
                if min_x<0 or min_y<0 or max_x>width or max_y>height:
                    flags.append("source_geometry_outside_image_clamped")
                box=[max(0,math.floor(min_x)-2),max(0,math.floor(min_y)-2),
                     min(width,math.ceil(max_x)+2),min(height,math.ceil(max_y)+2)]
                if box[2]<=box[0] or box[3]<=box[1]:
                    raise ValueError("source_region_outside_image")
                text=" ".join(w["text"] for w in words)
                text.encode("utf-8",errors="strict")
                yield {**base,"text":text,"bbox":box,"source_quadrilaterals":[w["quad"] for w in words],
                       "source_word_count":len(words),"region_unit":"publisher_row_within_field",
                       "flags":flags,"excluded":False}
            except (ValueError,KeyError,TypeError,UnicodeError) as exc:
                yield {**base,"excluded":True,"reason":str(exc)}


def difference_hash(image):
    small=image.convert("L").resize((9,8),Image.Resampling.LANCZOS)
    pixels=list(small.tobytes())
    value=0
    for row in range(8):
        for column in range(8):
            value=(value<<1)|int(pixels[row*9+column]>pixels[row*9+column+1])
    return f"{value:016x}"


def prepare(raw,output,*,verify=True):
    import pyarrow.parquet as pq
    raw,output=Path(raw).resolve(),Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("choose an empty prepared-data directory; never overwrite an earlier corpus")
    output.mkdir(parents=True,exist_ok=True)
    verified=verify_originals(raw) if verify else []
    started=time.perf_counter()
    db=sqlite3.connect(output/"audit.sqlite")
    db.execute("PRAGMA cache_size=-2048")
    db.execute("CREATE TABLE documents(owner TEXT PRIMARY KEY, split TEXT, image_sha TEXT, dhash TEXT)")
    db.execute("CREATE TABLE samples(sample_id TEXT PRIMARY KEY, owner TEXT, split TEXT, crop_sha TEXT, text_sha TEXT)")
    counts=Counter();flags=Counter();chars={k:Counter() for k in ("train","validation")}
    lengths={k:[] for k in chars};annotation_versions=Counter();splits_seen=Counter();geometry=[]
    crops=output/"crops";crops.mkdir()
    annotations=output/"annotations";annotations.mkdir()
    try:
        with (output/"records.jsonl").open("w",encoding="utf-8") as records,(output/"documents.jsonl").open("w",encoding="utf-8") as documents,(output/"exclusions.jsonl").open("w",encoding="utf-8") as exclusions:
            for item in approved_files():
                split="train" if item["path"].startswith("data/train-") else "validation"
                parquet=pq.ParquetFile(raw/item["path"],memory_map=True)
                if set(parquet.schema_arrow.names)!={"image","ground_truth"}:
                    raise ValueError("unexpected publisher source schema")
                for row_number,batch in enumerate(parquet.iter_batches(batch_size=1,use_threads=False)):
                    row=batch.to_pylist()[0]
                    image_bytes=row["image"].get("bytes")
                    if not isinstance(image_bytes,bytes) or not image_bytes:
                        raise ValueError("image bytes required; never follow source-supplied filesystem paths")
                    image_sha=hashlib.sha256(image_bytes).hexdigest()
                    owner=hashlib.sha256((SOURCE["revision"]+":"+item["path"]+":"+str(row_number)).encode()).hexdigest()
                    annotation=json.loads(row["ground_truth"])
                    meta=annotation.get("meta",{})
                    original_split=meta.get("split")
                    if not publisher_split_matches(split,original_split):
                        raise ValueError("publisher annotation and shard split disagree")
                    splits_seen[str(original_split)]+=1
                    annotation_versions[str(meta.get("version"))]+=1
                    with Image.open(io.BytesIO(image_bytes)) as original:
                        if original.width*original.height>40_000_000:
                            raise ValueError("source image exceeds preregistered pixel budget")
                        image=original.convert("RGB")
                    width,height=image.size
                    dhash=difference_hash(image)
                    db.execute("INSERT INTO documents VALUES(?,?,?,?)",(owner,split,image_sha,dhash))
                    (annotations/(owner+".json")).write_text(row["ground_truth"],encoding="utf-8")
                    doc={"document_id":owner,"split":split,"source_shard":item["path"],"source_row":row_number,
                         "source_image_sha256":image_sha,"width":width,"height":height,
                         "publisher_image_id":meta.get("image_id"),"publisher_split":original_split,
                         "annotation_path":"annotations/"+owner+".json","complete_page_transcription":False}
                    documents.write(json.dumps(doc,ensure_ascii=False)+"\n")
                    counts[split+"_receipts"]+=1
                    for region in regions(annotation,width,height):
                        base={"document_id":owner,"split":split,"annotation_index":region["annotation_index"],
                              "publisher_row_id":region.get("publisher_row_id")}
                        if region["excluded"]:
                            exclusions.write(json.dumps({**base,"reason":region["reason"]})+"\n")
                            counts[split+"_excluded_regions"]+=1
                            continue
                        crop=image.crop(region["bbox"])
                        blank=ImageStat.Stat(crop.convert("L")).stddev[0]<1.
                        if blank:region["flags"].append("near_blank_annotated_crop")
                        stream=io.BytesIO();crop.save(stream,format="PNG",compress_level=3)
                        data=stream.getvalue();crop_sha=hashlib.sha256(data).hexdigest()
                        crop_file=crops/(crop_sha+".png")
                        if not crop_file.exists():crop_file.write_bytes(data)
                        text=region["text"]
                        sample_id=hashlib.sha256((owner+":"+str(region["annotation_index"])+":"+region["publisher_row_id"]).encode()).hexdigest()
                        text_sha=hashlib.sha256(text.encode()).hexdigest()
                        record={**base,"sample_id":sample_id,"image_path":"crops/"+crop_sha+".png",
                                "image_sha256":crop_sha,"text":text,"text_sha256":text_sha,
                                "source_image_sha256":image_sha,"bbox":region["bbox"],
                                "source_quadrilaterals":region["source_quadrilaterals"],
                                "region_unit":region["region_unit"],"source_word_count":region["source_word_count"],
                                "flags":region["flags"]}
                        records.write(json.dumps(record,ensure_ascii=False,separators=(",",":"))+"\n")
                        db.execute("INSERT INTO samples VALUES(?,?,?,?,?)",(sample_id,owner,split,crop_sha,text_sha))
                        counts[split+"_regions"]+=1;flags.update(region["flags"]);chars[split].update(text)
                        lengths[split].append(len(text.encode()))
                        geometry.append((crop.width,crop.height))
                    if sum(counts[k+"_receipts"] for k in chars)%50==0:
                        db.commit()
                        print(json.dumps({"receipts_processed":sum(counts[k+"_receipts"] for k in chars),"regions":sum(counts[k+"_regions"] for k in chars)}),flush=True)
            db.commit()
        if counts["train_receipts"]!=800 or counts["validation_receipts"]!=100:
            raise ValueError("original publisher split cardinalities changed")
        overlap=db.execute("SELECT image_sha,COUNT(*) FROM documents GROUP BY image_sha HAVING COUNT(DISTINCT split)>1").fetchall()
        crop_overlap=db.execute("SELECT crop_sha,COUNT(*) FROM samples GROUP BY crop_sha HAVING COUNT(DISTINCT split)>1").fetchall()
        conflicts=db.execute("SELECT crop_sha,COUNT(*) FROM samples GROUP BY crop_sha HAVING COUNT(DISTINCT text_sha)>1").fetchall()
        docs=list(db.execute("SELECT owner,split,dhash FROM documents"))
        near=[]
        for i,(owner,split,dhash) in enumerate(docs):
            for other,other_split,other_hash in docs[:i]:
                if split!=other_split and (int(dhash,16)^int(other_hash,16)).bit_count()<=2:
                    near.append({"document_a":owner,"document_b":other,"hamming_distance":(int(dhash,16)^int(other_hash,16)).bit_count()})
        (output/"duplicate_findings.json").write_text(json.dumps({"cross_split_exact_source_images":overlap,
            "cross_split_exact_crops":crop_overlap,"conflicting_crop_labels":conflicts,"cross_split_near_image_candidates":near},indent=2)+"\n")
        report={"schema":"tac-ocr002.cord-preparation-receipt.v1","source_repository":SOURCE["repo_id"],
                "source_revision":SOURCE["revision"],"source_files":verified,"counts":dict(counts),
                "annotation_versions":dict(annotation_versions),"publisher_splits_observed":dict(splits_seen),
                "flags":dict(flags),"unique_crop_files":db.execute("SELECT COUNT(DISTINCT crop_sha) FROM samples").fetchone()[0],
                "cross_split_exact_source_image_groups":len(overlap),"cross_split_exact_crop_groups":len(crop_overlap),
                "conflicting_crop_label_groups":len(conflicts),"cross_split_near_image_candidate_pairs":len(near),
                "text_length_utf8_bytes":{split:{"min":min(values),"median":statistics.median(values),"p95":sorted(values)[math.ceil(.95*len(values))-1],"max":max(values)} for split,values in lengths.items()},
                "character_inventory":{split:dict(counter) for split,counter in chars.items()},
                "naira_symbol_count":sum(counter["₦"] for counter in chars.values()),
                "elapsed_preparation_seconds":time.perf_counter()-started,
                "pidgin_coverage":"NOT_ESTABLISHED_INDONESIAN_RECEIPT_SOURCE",
                "complete_page_gold":False,"region_boxes":"PUBLISHER_ANNOTATED_ORACLE_REGIONS_FOR_RECOGNITION_ONLY",
                "test_payloads_downloaded_by_this_job":0,"test_payloads_opened_by_this_job":0,
                "test_pristine_access_and_pretraining_contamination":"UNAUDITED_PUBLIC_WEB_EXAMPLES_MAY_EXIST",
                "new_optimizer_updates":0,"model_accuracy_measured":False,"training_prefit_qualified":False,
                "further_review_required":bool(overlap or crop_overlap or conflicts or near or flags),
                "source_authentication_scope":"content matches pinned publisher objects; not an independently signed scientific attestation",
                "source_code_sha256":sha_file(Path(__file__))}
        (output/"AUDIT.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
        (output/"SOURCE.json").write_text(json.dumps(SOURCE,indent=2)+"\n")
        (output/"PREPARATION_PROTOCOL.json").write_bytes((HERE/"PREPARATION_PROTOCOL.json").read_bytes())
        return report
    finally:
        db.close()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--download",action="store_true")
    args=parser.parse_args()
    if args.download:download_originals(args.raw)
    result=prepare(args.raw,args.output)
    print(json.dumps({key:result[key] for key in ("counts","flags","unique_crop_files","elapsed_preparation_seconds","training_prefit_qualified")},indent=2))
