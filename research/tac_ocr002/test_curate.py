import hashlib
import json

import pytest

from .curate import filter_training


def test_filter_only_training_and_preserve_every_validation_byte(tmp_path):
    docs=[{"document_id":"t1","split":"train","decoded_pixel_sha256":"pixel-v"},
          {"document_id":"t2","split":"train","decoded_pixel_sha256":"pixel-other"},
          {"document_id":"v","split":"validation","decoded_pixel_sha256":"pixel-v"}]
    rows=[]
    for sample,owner,crop,flags in [("a","t1","good",[]),("b","t2","overlap",[]),
                                    ("c","t2","conflict",[]),("d","t2","flagged",["source_flag"]),
                                    ("e","t2","clear",[]),("f","v","overlap",[]),("g","v","conflict",[])]:
        rows.append({"sample_id":sample,"document_id":owner,"image_sha256":crop,
                     "split":"validation" if owner=="v" else "train","flags":flags,"text":"unchanged"})
    (tmp_path/"documents.jsonl").write_text("".join(json.dumps(d)+"\n" for d in docs))
    encoded=[(json.dumps(r)+"\n").encode() for r in rows]
    (tmp_path/"records.jsonl").write_bytes(b"".join(encoded))
    (tmp_path/"duplicate_findings.json").write_text(json.dumps({"cross_split_exact_crops":[["overlap",2]],
        "conflicting_crop_labels":[["conflict",2]],"cross_split_near_image_candidates":[]}))
    result=filter_training(tmp_path)
    expected=b"".join(line for row,line in zip(rows,encoded) if row["split"]=="validation")
    assert (tmp_path/"validation_original.jsonl").read_bytes()==expected
    assert result["validation_original_sha256"]==hashlib.sha256(expected).hexdigest()
    assert result["counts"]=={"original_training_regions":5,"excluded_training_regions":4,
                             "retained_training_regions":1,"validation_regions_unchanged":2}
    retained=[json.loads(line) for line in (tmp_path/"train_filtered.jsonl").open()]
    assert [r["sample_id"] for r in retained]==["e"]


def test_no_filter_without_decoded_pixel_provenance(tmp_path):
    (tmp_path/"documents.jsonl").write_text(json.dumps({"document_id":"a"})+"\n")
    with pytest.raises(ValueError,match="decoded"):
        filter_training(tmp_path)
