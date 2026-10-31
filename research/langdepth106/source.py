"""Pinned Tiny Shakespeare bytes and partition-isolated source preparation.

No tokenization is fitted: the model's fixed vocabulary is all 256 byte IDs.
TRAIN, DEV and TEST are contiguous source-byte ranges with no boundary-spanning
source/target pair. Preparing files is a source-integrity operation, not a model
fit, evaluation, external permission attestation, or general-language benchmark.
Only ``training/train.bin`` and its metadata-only receipt may enter fit staging;
``evaluation`` is a separate directory for frozen post-fit evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPOSITORY = "karpathy/char-rnn"
COMMIT = "6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e"
PUBLISHER_PATH = "data/tinyshakespeare/input.txt"
SOURCE_URL = f"https://raw.githubusercontent.com/{REPOSITORY}/{COMMIT}/{PUBLISHER_PATH}"
GIT_BLOB_SHA1 = "7dcb3a2d4cc3b48b6283dd46870bfeb78f88aac9"
SOURCE_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
SOURCE_BYTES = 1_115_394
README_BLOB_SHA1 = "a14aa2def8aa8894ecaf63165a6651df4766aa52"
README_SHA256 = "6540438971f884ae42315e517df7c9d0e1ca520daa120f0d0845b50944485ec4"
PARTITION_SHA256 = {
    "train": "a9e24e23a1ec77744dad26844bfd5a09b6e041954e1eef0000e7f24cba6db735",
    "dev": "c64e90c160750d91a9d2d9b1ace5eab1a440f5a1701d32a8e97a179b7839f1c8",
    "test": "8b5a8c8af45905547ad03b7aae6980a277a5dd78924a6e6ac64b75d63d7d40bd",
}
FORMAT = "langdepth106-pinned-raw-byte-source-v1"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def verify_original(data: bytes) -> dict[str, Any]:
    """Require exact bytes matching both the frozen Git blob and SHA-256."""
    if type(data) is not bytes:
        raise TypeError("source must be raw bytes")
    if len(data) != SOURCE_BYTES or sha256(data) != SOURCE_SHA256 or git_blob_sha1(data) != GIT_BLOB_SHA1:
        raise ValueError("source differs from pinned publisher bytes")
    text = data.decode("utf-8", errors="strict")
    return {
        "repository": REPOSITORY,
        "commit": COMMIT,
        "publisher_path": PUBLISHER_PATH,
        "source_url": SOURCE_URL,
        "git_blob_sha1": git_blob_sha1(data),
        "sha256": sha256(data),
        "bytes": len(data),
        "encoding": "utf-8",
        "utf8_characters": len(text),
        "unique_characters": len(set(text)),
        "unique_bytes": len(set(data)),
        "fixed_byte_vocabulary": 256,
        "fit_preprocessing": "none; exact raw byte IDs, no fitted vocabulary",
        "corpus_scope": "small diagnostic corpus, not broad language competence",
    }


def verify_publisher_metadata(tree_bytes: bytes, readme_bytes: bytes) -> dict[str, Any]:
    """Check retained publisher metadata and the actual README license declaration.

    The retrieved tree and README are provenance evidence, not an independently
    signed source/outcome permission attestation. No separate LICENSE file exists
    in the inspected pinned root. Its README's MIT declaration is reported without
    inventing a standalone full license text or conflating it with Shakespeare's
    underlying public-domain works.
    """
    if sha256(readme_bytes) != README_SHA256 or git_blob_sha1(readme_bytes) != README_BLOB_SHA1:
        raise ValueError("publisher README differs from pinned bytes")
    readme = readme_bytes.decode("utf-8", errors="strict")
    if not readme.rstrip().endswith("## License\n\nMIT"):
        raise ValueError("expected publisher README license declaration is missing")
    tree = json.loads(tree_bytes)
    if not isinstance(tree, dict) or tree.get("truncated") is not False:
        raise ValueError("publisher tree must be a complete metadata response")
    if tree.get("sha") != COMMIT:
        raise ValueError("publisher tree response is not the inspected pinned reference")
    entries = tree.get("tree")
    if not isinstance(entries, list):
        raise ValueError("publisher tree entries are missing")
    paths = [entry.get("path") for entry in entries if isinstance(entry, dict)]
    if (len(paths) != len(entries) or any(type(path) is not str for path in paths)
            or len(set(paths)) != len(paths)):
        raise ValueError("publisher tree has malformed or duplicate paths")
    by_path = {entry["path"]: entry for entry in entries}
    source = by_path.get(PUBLISHER_PATH, {})
    readme_entry = by_path.get("Readme.md", {})
    if source.get("type") != "blob" or source.get("sha") != GIT_BLOB_SHA1 or source.get("size") != SOURCE_BYTES:
        raise ValueError("publisher source-blob metadata mismatch")
    if readme_entry.get("type") != "blob" or readme_entry.get("sha") != README_BLOB_SHA1 or readme_entry.get("size") != len(readme_bytes):
        raise ValueError("publisher README metadata mismatch")
    root_license_paths = sorted(path for path in paths
                                if "/" not in path and ("license" in path.lower() or "copying" in path.lower()))
    if root_license_paths:
        raise ValueError("pinned root license-file inventory differs from inspected metadata")
    return {
        "tree_url": f"https://api.github.com/repos/{REPOSITORY}/git/trees/{COMMIT}?recursive=1",
        "tree_response_sha256": sha256(tree_bytes),
        "readme_url": f"https://github.com/{REPOSITORY}/blob/{COMMIT}/Readme.md",
        "readme_git_blob_sha1": README_BLOB_SHA1,
        "readme_sha256": README_SHA256,
        "publisher_license_declaration": "MIT, declared in pinned Readme.md",
        "standalone_root_license_paths": root_license_paths,
        "underlying_works": "Shakespeare public-domain text; publisher compilation/README evidence recorded separately",
        "independent_permission_attestation": "not supplied by this source-preparation script",
    }


def partition_ranges(length: int) -> dict[str, tuple[int, int]]:
    """Floor 90% and 95% cut points using integer arithmetic only."""
    if type(length) is not int or length < 40:
        raise ValueError("source length must be an integer >= 40")
    train_end = 9 * length // 10
    dev_end = 19 * length // 20
    return {"train": (0, train_end), "dev": (train_end, dev_end), "test": (dev_end, length)}


def split_raw_bytes(data: bytes) -> tuple[dict[str, bytes], dict[str, Any]]:
    """Split without normalizing, shuffling, fitting transforms, or crossing cuts."""
    if type(data) is not bytes:
        raise TypeError("source must be raw bytes")
    ranges = partition_ranges(len(data))
    partitions = {name: data[start:end] for name, (start, end) in ranges.items()}
    metadata = {
        name: {
            "source_byte_start_inclusive": ranges[name][0],
            "source_byte_end_exclusive": ranges[name][1],
            "bytes": len(part),
            "sha256": sha256(part),
            "adjacent_source_target_pairs": max(0, len(part) - 1),
        }
        for name, part in partitions.items()
    }
    return partitions, metadata


def source_target_window(partition: bytes, start: int, sequence_length: int) -> tuple[bytes, bytes]:
    """Return equally sized next-byte inputs/targets wholly inside one partition."""
    if type(partition) is not bytes:
        raise TypeError("partition must be bytes")
    if type(start) is not int or type(sequence_length) is not int or start < 0 or sequence_length < 1:
        raise ValueError("invalid window start or sequence length")
    if start + sequence_length >= len(partition):
        raise ValueError("source-target window would leave its partition")
    return partition[start:start + sequence_length], partition[start + 1:start + sequence_length + 1]


def verify_partition(data: bytes, name: str) -> None:
    """Verify partition bytes against source-frozen hashes, not a mutable receipt."""
    if type(data) is not bytes or name not in PARTITION_SHA256:
        raise ValueError("invalid partition name or non-byte payload")
    start, end = partition_ranges(SOURCE_BYTES)[name]
    if len(data) != end - start or sha256(data) != PARTITION_SHA256[name]:
        raise ValueError(f"{name} bytes differ from the frozen publisher partition")


def verify_training_stage(directory: Path) -> dict[str, Any]:
    """Fail closed if fit data contains anything beyond TRAIN bytes and metadata."""
    directory = Path(directory)
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("training stage must be a real directory")
    if {p.name for p in directory.iterdir()} != {"train.bin", "source_receipt.json"}:
        raise ValueError("training stage must contain only train.bin and its receipt")
    if any(p.is_symlink() or not p.is_file() for p in directory.iterdir()):
        raise ValueError("training stage cannot contain links or non-files")
    receipt = json.loads((directory / "source_receipt.json").read_text(encoding="utf-8"))
    if receipt.get("format") != FORMAT or receipt.get("stage_access") != "TRAIN_ONLY":
        raise ValueError("training receipt is not the declared TRAIN-only format")
    source = receipt.get("source", {})
    if source.get("sha256") != SOURCE_SHA256 or source.get("git_blob_sha1") != GIT_BLOB_SHA1 or source.get("commit") != COMMIT:
        raise ValueError("training receipt does not identify the pinned source")
    data = (directory / "train.bin").read_bytes()
    verify_partition(data, "train")
    train = receipt["partitions"]["train"]
    if len(data) != train["bytes"] or sha256(data) != train["sha256"]:
        raise ValueError("training partition differs from receipt")
    expected_start, expected_end = partition_ranges(SOURCE_BYTES)["train"]
    if train["source_byte_start_inclusive"] != expected_start or train["source_byte_end_exclusive"] != expected_end or train["bytes"] != expected_end - expected_start:
        raise ValueError("training source range differs from frozen split")
    return receipt


def prepare_source(source_path: Path, output_dir: Path, *, publisher_tree_path: Path,
                   publisher_readme_path: Path) -> dict[str, Any]:
    """Verify originals first, then create separated training/evaluation files.

    The caller supplies already retrieved source bytes explicitly. This function
    performs no network requests, executes no publisher code, and refuses to
    overwrite any destination. Dataset content must remain outside source control.
    """
    data = Path(source_path).read_bytes()
    source = verify_original(data)
    publisher = verify_publisher_metadata(Path(publisher_tree_path).read_bytes(),
                                          Path(publisher_readme_path).read_bytes())
    partitions, partition_metadata = split_raw_bytes(data)
    for name, part in partitions.items():
        verify_partition(part, name)
    source["split_sha256"] = {name: meta["sha256"] for name, meta in partition_metadata.items()}
    receipt = {
        "format": FORMAT,
        "source": source,
        "publisher": publisher,
        "split_policy": "contiguous bytes: floor(9*N/10), floor(19*N/20); no cross-partition pairs",
        "partitions": partition_metadata,
        "representation": "raw bytes; fixed 256 symbols; no vocabulary fit",
        "boundary_policy": "reset carried state at each independent stream; never carry TRAIN state into DEV/TEST",
        "evaluation_policy": "separate files; post-fit access requires the frozen evaluation protocol",
        "preparation_is_training_authorization": False,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    training = output / "training"
    evaluation = output / "evaluation"
    training.mkdir()
    evaluation.mkdir()
    (training / "train.bin").write_bytes(partitions["train"])
    (evaluation / "dev.bin").write_bytes(partitions["dev"])
    (evaluation / "test.bin").write_bytes(partitions["test"])
    training_receipt = {**receipt, "stage_access": "TRAIN_ONLY"}
    (training / "source_receipt.json").write_text(json.dumps(training_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "source_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    verify_training_stage(training)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--publisher-tree", required=True, type=Path)
    parser.add_argument("--publisher-readme", required=True, type=Path)
    args = parser.parse_args()
    receipt = prepare_source(args.source, args.output_dir, publisher_tree_path=args.publisher_tree,
                             publisher_readme_path=args.publisher_readme)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
