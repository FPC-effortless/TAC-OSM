import hashlib
import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from .prepare import SOURCE,approved_files,quad_points,regions,teacher_forcing,publisher_split_matches
from .dataset import RegionDataset,collate_regions


def word(text,row,x=5):
    return {"text":text,"row_id":row,"quad":{"x1":x,"y1":5,"x2":x+10,"y2":5,"x3":x+10,"y3":15,"x4":x,"y4":15}}


def test_only_publisher_train_validation_can_be_downloaded():
    assert len(approved_files())==5
    assert all("/test-" not in item["path"] for item in approved_files())


@pytest.mark.parametrize("alias",["dev","valid","validation"])
def test_publisher_validation_vocabulary_is_not_a_resplit(alias):
    assert publisher_split_matches("validation",alias)
    assert not publisher_split_matches("train",alias)


def test_annotation_split_contradiction_still_fails():
    assert not publisher_split_matches("validation","train")
    assert not publisher_split_matches("train","test")


def test_row_groups_preserve_unicode_and_source_order():
    annotation={"valid_line":[{"words":[word("Wetin",0),word("₦20?",0,20),word("next",1)]}]}
    actual=list(regions(annotation,100,50))
    assert len(actual)==2
    assert actual[0]["text"]=="Wetin ₦20?" and actual[1]["text"]=="next"
    assert actual[0]["bbox"]==[3,3,32,17]


def test_missing_row_id_is_flagged_not_invented():
    w=word("a",0);del w["row_id"]
    actual=list(regions({"valid_line":[{"words":[w]}]},100,50))[0]
    assert "missing_publisher_row_id_field_region_only" in actual["flags"]


@pytest.mark.parametrize("bad",[None,{"x1":0},{**word("a",0)["quad"],"x1":float("nan")}])
def test_bad_geometry_is_not_silently_used(bad):
    with pytest.raises(ValueError):quad_points(bad)


def test_invalid_regions_are_explicit_exclusions():
    actual=list(regions({"valid_line":[{"words":[{"text":"","quad":None}]}]},100,50))
    assert actual[0]["excluded"] and actual[0]["reason"]


def test_teacher_forcing_keeps_final_character_and_eos():
    inputs,labels=teacher_forcing("₦5")
    assert inputs==[256,*"₦5".encode()]
    assert labels==[*"₦5".encode(),257]
    assert len(inputs)==len(labels)


def make_fixture(root):
    crop=root/"crops";crop.mkdir()
    p=crop/"one.png";Image.new("RGB",(20,10),"white").save(p)
    digest=hashlib.sha256(p.read_bytes()).hexdigest()
    row={"split":"train","image_path":"crops/one.png","image_sha256":digest,
         "text":"Wetin?","document_id":"a","sample_id":"b","flags":[]}
    (root/"records.jsonl").write_text(json.dumps(row)+"\n"+json.dumps({**row,"split":"validation"})+"\n")
    return p,row


def test_lazy_loader_and_padding_do_not_leak_labels_into_pixels(tmp_path):
    _,row=make_fixture(tmp_path)
    data=RegionDataset(tmp_path,"train")
    assert len(data)==1 and data[0]["image"].size==(20,10)
    batch=collate_regions([data[0],{**data[0],"input_ids":[256,65],"labels":[65,257]}])
    assert batch["labels"][1,2:].eq(-100).all()
    assert "text" not in batch and "document_id" not in batch
    assert len(RegionDataset(tmp_path,"validation"))==1
    with pytest.raises(ValueError):RegionDataset(tmp_path,"test")


def test_changed_crop_rejected_before_predictor(tmp_path):
    p,_=make_fixture(tmp_path);data=RegionDataset(tmp_path)
    p.write_bytes(b"altered")
    with pytest.raises(ValueError,match="changed"):data[0]
