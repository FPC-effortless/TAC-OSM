from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re


WORD_RE = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*")


def words(text: str) -> list[str]:
    return [match.group(0).lower().replace("’", "'") for match in WORD_RE.finditer(text)]


def speaker_labels(text: str) -> list[str]:
    labels = []
    for line in text.splitlines():
        candidate = line.strip()
        if candidate.endswith(":") and 1 <= len(candidate) <= 40:
            labels.append(candidate[:-1].strip().lower())
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Score matched Tiny Shakespeare generations.")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--samples", type=Path, default=Path(__file__).with_name("MATCHED_SAMPLES.json"))
    parser.add_argument("--benchmarks", type=Path, default=Path(__file__).with_name("RECORDED_BENCHMARKS.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    corpus = args.corpus.read_text(encoding="utf-8")
    corpus_words = words(corpus)
    corpus_vocab = set(corpus_words)
    corpus_word_counts = Counter(corpus_words)
    corpus_labels = set(speaker_labels(corpus))
    benchmark_rows = json.loads(args.benchmarks.read_text(encoding="utf-8"))["rows"]
    benchmarks = {row["model"]: row for row in benchmark_rows}

    samples = json.loads(args.samples.read_text(encoding="utf-8"))
    by_model: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        by_model[sample["model"]].append(sample)

    results = []
    for model, model_samples in by_model.items():
        generated_words = []
        labels = []
        replacement_chars = 0
        repeated_word_bigrams = 0
        total_word_bigrams = 0
        for sample in model_samples:
            sample_words = words(sample["text"])
            generated_words.extend(sample_words)
            labels.extend(speaker_labels(sample["text"]))
            replacement_chars += sample["text"].count("\ufffd")
            bigrams = list(zip(sample_words, sample_words[1:]))
            counts = Counter(bigrams)
            repeated_word_bigrams += sum(count - 1 for count in counts.values() if count > 1)
            total_word_bigrams += len(bigrams)

        known_words = sum(word in corpus_vocab for word in generated_words)
        frequent_words = sum(corpus_word_counts[word] >= 5 for word in generated_words)
        valid_labels = sum(label in corpus_labels for label in labels)
        unknown_counts = Counter(word for word in generated_words if word not in corpus_vocab)
        invalid_labels = sorted({label for label in labels if label not in corpus_labels})
        benchmark = benchmarks[model]
        results.append(
            {
                "model": model,
                "samples": len(model_samples),
                "generated_bytes": sum(len(sample["text"].encode("utf-8")) for sample in model_samples),
                "word_tokens": len(generated_words),
                "corpus_vocabulary_rate": known_words / len(generated_words),
                "frequent_corpus_word_rate": frequent_words / len(generated_words),
                "speaker_labels": len(labels),
                "valid_speaker_label_rate": valid_labels / len(labels) if labels else None,
                "most_common_unknown_words": unknown_counts.most_common(12),
                "invalid_speaker_labels": invalid_labels,
                "replacement_characters": replacement_chars,
                "repeated_bigram_rate": repeated_word_bigrams / total_word_bigrams,
                "median_generation_ms": sorted(sample["generation_ms"] for sample in model_samples)[len(model_samples) // 2],
                "test_carry_nll": benchmark["test_carry_nll"],
                "test_carry_accuracy": benchmark["test_carry_accuracy"],
                "original_decode_ms_per_byte": benchmark["original_decode_ms_per_byte"],
            }
        )

    results.sort(key=lambda row: (-row["corpus_vocabulary_rate"], row["model"]))
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
