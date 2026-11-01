"""Lazy region loader; labels stay outside image encoding and inference IO."""
import hashlib
import json
from pathlib import Path

from PIL import Image

from .prepare import teacher_forcing


class RegionDataset:
    def __init__(self,root,split="train",*,verify_crops=True):
        if split not in ("train","validation"):
            raise ValueError("this prepared corpus intentionally excludes TEST")
        self.root=Path(root).resolve()
        self.verify_crops=verify_crops
        self.offsets=[]
        preferred="train_filtered.jsonl" if split=="train" else "validation_original.jsonl"
        self.path=self.root/preferred if (self.root/preferred).exists() else self.root/"records.jsonl"
        with self.path.open("rb") as stream:
            while True:
                offset=stream.tell();line=stream.readline()
                if not line:break
                if json.loads(line)["split"]==split:self.offsets.append(offset)

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self,index):
        with self.path.open("rb") as stream:
            stream.seek(self.offsets[index]);row=json.loads(stream.readline())
        path=(self.root/row["image_path"]).resolve()
        if not path.is_relative_to(self.root) or path.suffix!=".png":
            raise ValueError("crop escaped the prepared corpus")
        raw=path.read_bytes()
        if self.verify_crops and hashlib.sha256(raw).hexdigest()!=row["image_sha256"]:
            raise ValueError("crop bytes changed after preparation")
        import io
        with Image.open(io.BytesIO(raw)) as source:image=source.convert("RGB")
        inputs,labels=teacher_forcing(row["text"])
        # IDs are control/audit metadata, never a model tensor or text prompt.
        return {"image":image,"input_ids":inputs,"labels":labels,
                "document_id":row["document_id"],"sample_id":row["sample_id"],"flags":row["flags"]}


def collate_regions(samples):
    """Use the SAME samples/order and frozen encoder for all decoder arms."""
    import torch
    from research.tac_ocr001.tokenizer import PAD
    if not samples:raise ValueError("nonempty region batch required")
    length=max(len(s["input_ids"]) for s in samples)
    inputs=torch.full((len(samples),length),PAD,dtype=torch.long)
    labels=torch.full((len(samples),length),-100,dtype=torch.long)
    for i,sample in enumerate(samples):
        n=len(sample["input_ids"])
        inputs[i,:n]=torch.tensor(sample["input_ids"])
        labels[i,:n]=torch.tensor(sample["labels"])
    return {"images":[s["image"] for s in samples],"input_ids":inputs,"labels":labels,
            "lengths":torch.tensor([len(s["input_ids"]) for s in samples]),
            "metadata":[{"document_id":s["document_id"],"sample_id":s["sample_id"],"flags":s["flags"]} for s in samples]}
