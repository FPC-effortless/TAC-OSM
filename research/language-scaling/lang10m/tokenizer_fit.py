"""Train a reproducible 4K byte-fallback SentencePiece BPE using FIT only.

No model training, inference, or remote I/O. All publisher source bytes must
already be locally authenticated against their immutable SHA256 receipts.
This does not by itself prove publisher authenticity or rights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

VOCAB = 4096
MODEL_TYPE = "bpe"
NORMALIZATION = "identity"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_fit_manifest(path: Path) -> list[Path]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("partition") != "FIT":
        raise ValueError("tokenizer may only fit to a FIT manifest")
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("nonempty authenticated FIT sources required")
    paths = []
    seen = set()
    for entry in sources:
        if entry.get("partition") != "FIT":
            raise ValueError("DEV, TEST and unknown partitions are forbidden")
        if not all(entry.get(k) for k in ("path", "sha256", "license", "publisher", "commit")):
            raise ValueError("source identity, license, commit and SHA256 required")
        source = Path(entry["path"]).expanduser().resolve(strict=True)
        if source in seen or "," in str(source):
            raise ValueError("repeated or non-representable FIT path")
        seen.add(source)
        if sha256(source) != entry["sha256"]:
            raise ValueError(f"source SHA256 mismatch: {source}")
        source.read_text(encoding="utf-8", errors="strict")
        paths.append(source)
    return paths


def train(manifest_path: Path, model_prefix: Path) -> dict:
    import sentencepiece as spm

    paths = validate_fit_manifest(manifest_path)
    model_prefix = model_prefix.expanduser().resolve()
    model_prefix.parent.mkdir(parents=True, exist_ok=True)
    if any(Path(str(model_prefix) + suffix).exists() for suffix in (".model", ".vocab")):
        raise FileExistsError("never silently overwrite a fitted tokenizer")
    args = dict(
        input=",".join(map(str, paths)),
        model_prefix=str(model_prefix),
        model_type=MODEL_TYPE,
        vocab_size=VOCAB,
        byte_fallback=True,
        hard_vocab_limit=True,
        normalization_rule_name=NORMALIZATION,
        add_dummy_prefix=False,
        remove_extra_whitespaces=False,
        split_by_whitespace=False,
        input_sentence_size=0,
        shuffle_input_sentence=False,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
    )
    spm.SentencePieceTrainer.train(**args)
    processor = spm.SentencePieceProcessor(model_file=str(model_prefix) + ".model")
    if processor.vocab_size() != VOCAB:
        raise ValueError("trained vocabulary size mismatch")
    for probe in ("def f(x):\n    return x + 1\n", " café 🧠\n", "\tif value is None:\n"):
        if processor.decode(processor.encode(probe)) != probe:
            raise ValueError("tokenizer does not roundtrip exact source text")
    receipt = {
        "variant": "LANG10M001",
        "status": "TOKENIZER_ONLY",
        "partition": "FIT",
        "vocabulary": VOCAB,
        "model_type": MODEL_TYPE,
        "normalization": NORMALIZATION,
        "byte_fallback": True,
        "fit_manifest_sha256": sha256(manifest_path),
        "fit_sources": [
            {"path": str(p), "sha256": sha256(p)} for p in paths
        ],
        "model_sha256": sha256(Path(str(model_prefix) + ".model")),
        "vocab_sha256": sha256(Path(str(model_prefix) + ".vocab")),
        "dev_accessed": False,
        "test_accessed": False,
        "neural_fits": 0,
    }
    receipt_path = Path(str(model_prefix) + ".receipt.json")
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-manifest", type=Path, required=True)
    parser.add_argument("--model-prefix", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(train(args.fit_manifest, args.model_prefix), sort_keys=True))


if __name__ == "__main__":
    main()
