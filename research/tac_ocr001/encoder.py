"""Optional, immutable-revision TrOCR *vision encoder only* integration.

Single-line/region input, not a claim of pretrained whole-page understanding.
Heavy dependencies and downloads are optional and never implicit.
"""
from __future__ import annotations

import hashlib
import os
import re
import tempfile
from pathlib import Path

import torch


def spatial_patch_features(features, image_size, patch_size):
    height,width=(image_size,image_size) if isinstance(image_size,int) else image_size
    ph,pw=(patch_size,patch_size) if isinstance(patch_size,int) else patch_size
    rows,columns=height//ph,width//pw
    prefix=features.shape[1]-rows*columns
    # ViT has a CLS token; DeiT also has a distillation token. Neither has
    # a spatial box and neither may be silently assigned to an image patch.
    if prefix not in (1,2) or rows<1 or columns<1:
        raise ValueError("expected ViT/DeiT spatial patch grid")
    ys,xs=torch.meshgrid(torch.arange(rows,device=features.device),torch.arange(columns,device=features.device),indexing="ij")
    boxes=torch.stack((xs/columns,ys/rows,(xs+1)/columns,(ys+1)/rows),-1).reshape(1,-1,4).to(features.dtype)
    return features[:,prefix:].contiguous(),boxes.expand(features.shape[0],-1,-1).contiguous()


class FrozenTrOCREncoder(torch.nn.Module):
    def __init__(self, model_id, revision, *, local_files_only=True):
        super().__init__()
        if not re.fullmatch(r"[0-9a-f]{40}",revision):
            raise ValueError("pin an immutable publisher commit, not main/latest")
        from transformers import AutoImageProcessor, VisionEncoderDecoderModel
        # The publisher distributes a combined checkpoint. Load once, retain
        # only its frozen encoder. Do not deploy/count its unused text decoder.
        self.processor=AutoImageProcessor.from_pretrained(model_id,revision=revision,
                            trust_remote_code=False,local_files_only=local_files_only,use_fast=False)
        full,loading=VisionEncoderDecoderModel.from_pretrained(model_id,revision=revision,
                            trust_remote_code=False,local_files_only=local_files_only,use_safetensors=True,
                            output_loading_info=True)
        encoder=full.get_encoder()
        # TrOCR distributes sequence features without a trained pooled head.
        # Transformers may initialize that unused head; remove it entirely.
        if hasattr(encoder,"pooler"):
            encoder.pooler=None
        missing=[key for key in loading.get("missing_keys",[]) if key.startswith("encoder.") and not key.startswith("encoder.pooler.")]
        if missing:
            raise ValueError("publisher checkpoint is missing deployed encoder weights: "+str(missing))
        self.encoder=encoder.eval().requires_grad_(False)
        del full
        self.model_id,self.revision=model_id,revision
        self.feature_dim=self.encoder.config.hidden_size
        self.learned_parameters=sum(p.numel() for p in self.encoder.parameters())
        self.identity=hashlib.sha256((model_id+":"+revision+":"+str(self.processor.to_dict())).encode()).hexdigest()

    def train(self,mode=True):
        super().train(False)
        return self

    @torch.inference_mode()
    def forward(self, images):
        device=next(self.encoder.parameters()).device
        pixels=self.processor(images=images,return_tensors="pt").pixel_values.to(device)
        features=self.encoder(pixel_values=pixels).last_hidden_state
        return spatial_patch_features(features,self.encoder.config.image_size,self.encoder.config.patch_size)


class FeatureCache:
    """CPU/disk feature cache keyed by *image bytes* and immutable encoder.

    This API never accepts annotations, source filenames or document text.
    File identity is content-addressed; disk growth is explicitly unbounded.
    """
    def __init__(self, root):
        self.root=Path(root)
        self.root.mkdir(parents=True,exist_ok=True)

    @staticmethod
    def key(image_bytes,encoder_identity):
        return hashlib.sha256(encoder_identity.encode()+b"\0"+image_bytes).hexdigest()

    def get(self,key):
        if not re.fullmatch(r"[0-9a-f]{64}",key):
            raise ValueError("content cache key required")
        path=self.root/(key+".pt")
        if not path.exists():
            return None
        value=torch.load(path,map_location="cpu",weights_only=True)
        if set(value)!={"key","features","boxes"} or value["key"]!=key:
            raise ValueError("feature cache identity mismatch")
        if (value["features"].ndim!=3 or value["boxes"].shape!=(*value["features"].shape[:2],4) or
                not torch.isfinite(value["features"]).all() or not torch.isfinite(value["boxes"]).all()):
            raise ValueError("invalid cached features")
        return value["features"],value["boxes"]

    def put(self,key,features,boxes):
        if not re.fullmatch(r"[0-9a-f]{64}",key):
            raise ValueError("content cache key required")
        path=self.root/(key+".pt")
        descriptor,name=tempfile.mkstemp(prefix=key+"-",suffix=".tmp",dir=self.root)
        os.close(descriptor)
        temporary=Path(name)
        try:
            torch.save({"key":key,"features":features.detach().cpu().contiguous().clone(),
                        "boxes":boxes.detach().cpu().contiguous().clone()},temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
