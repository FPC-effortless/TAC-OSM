"""Numeric-only probe of a pinned historical checkpoint on constructed input.

This does not evaluate real language quality or causal memory usefulness.
No training, source download, protected data or optimizer is involved.
"""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

from checkpoints import BASE_COMMIT, ROOT, audit_checkpoint, required_source_hashes, sha256

CHECKPOINT = "research/replay-bundles/CODEVALUE003/tac-1701/MODEL-010240.pt"
CHECKPOINT_SHA256 = "2e24990a1cbce06b7ca9e7e764b3f6d9106a9ac7706006f502bfdaa8614949a6"


def run():
    path = ROOT / CHECKPOINT
    if not path.is_file():
        return {"status": "INCONCLUSIVE", "reason": "pinned historical checkpoint unavailable locally",
                "checkpoint": CHECKPOINT, "optimizer_steps": 0}
    if sha256(path) != CHECKPOINT_SHA256:
        return {"status": "FAIL", "reason": "historical checkpoint hash differs", "optimizer_steps": 0}
    # Artificial byte sequence, not a source-authentic prediction benchmark.
    sequence = (b"def add(a, b):\n    return a + b\n" * 8)[:129]
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        inputs, receipt = folder/'numeric.npy', folder/'receipt.json'
        np.save(inputs,np.frombuffer(sequence,dtype=np.uint8).astype(np.int64)[None],allow_pickle=False)
        receipt.write_text(json.dumps({"schema":"langmath095-checkpoint-v1", "model_kind":"500k",
            "partition":"DEV", "publisher":"LANGMATH095 constructed numerical fixture, not a corpus",
            "license":"not applicable: generated diagnostic byte sequence",
            "checkpoint_sha256":CHECKPOINT_SHA256, "inputs_sha256":sha256(inputs),
            "source_commit":BASE_COMMIT, "source_sha256":required_source_hashes('500k')}))
        result = audit_checkpoint(path,inputs,receipt)
    result.update(checkpoint=CHECKPOINT, input_evidence="CONSTRUCTED_NUMERICAL_FIXTURE",
                  historical_training_metadata="CODEVALUE003 seed1701 step10240; internally measured, not independently qualified",
                  quality_or_causal_utility_claim=False)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result=run()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result,handle,indent=2,sort_keys=True,allow_nan=False)
        handle.write('\n')
    print(json.dumps({'status':result['status'],'output':str(args.output),'optimizer_steps':0}))
    return 1 if result['status'].startswith(('FAIL','BLOCKED')) else 0


if __name__=='__main__':
    raise SystemExit(main())
