"""ME066 Git byte-storage probe. Untrained, no source or heldout trajectories."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import torch
ROOT=Path("research/replay-bundles/ME066")
ROOT.mkdir(parents=True,exist_ok=True)
file=ROOT/"UNTRAINED_STORAGE_PROBE.pt"
if file.exists():raise RuntimeError("probe already written; never silently overwrite original bytes")
torch.save({"ME066_untrained_storage_only":True,
            "no_training":True,"no_source_data":True,
            "checkpoint_format":"torch serialization; full future optimizer/RNG state required",
            "probe_tensor":torch.arange(4,dtype=torch.float32)},file)
manifest={
    "status":"GIT_STORAGE_PROBE_PASS",
    "path":str(file),
    "sha256":hashlib.sha256(file.read_bytes()).hexdigest(),
    "file_bytes":file.stat().st_size,
    "trained_model_count":0,
    "publisher_trajectories_opened":0,
    "heldout_trajectories_opened":0,
    "warning":"This untrained probe is neither a fitted model nor skill evidence",
}
(ROOT/"STORAGE_PROBE_MANIFEST.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
print("ME066_GIT_STORAGE_PROBE="+json.dumps(manifest,sort_keys=True),flush=True)
