# TINY_SHAKESPEARE_READABILITY001

Post-hoc, internally measured readability diagnostic for the nine completed
approximately one-million-parameter Tiny Shakespeare checkpoints:

- TAC-OSM, four layers, seeds 10601–10603
- TAC-OSM, eight layers, seeds 10601–10603
- causal Transformer, four layers, seeds 10601–10603

## Scientific status

This directory preserves an exploratory diagnostic of already-trained
checkpoints. No new training, checkpoint modification, protected holdout
access, checkpoint promotion, or release decision occurred. Tiny Shakespeare is a
language-model fixture, not real-world capability evidence. The review was
post-hoc and was not preregistered or blinded, so it cannot establish
architectural superiority or independent generalization.

Evidence level: **internally measured, not independently qualified**.

## Contents

- `RESULTS.md`: decision-oriented interpretation and limitations
- `MANIFEST.json`: source, checkpoint, implementation, and protocol hashes
- `MATCHED_SAMPLES.json`: two matched 1024-byte generations from every model
- `SAMPLING_SENSITIVITY.json`: temperature sweep for the strongest recorded
  TAC and Transformer checkpoints
- `RECORDED_BENCHMARKS.json`: frozen metrics copied from the model-lab result
  records, with source hashes
- `METRICS.json`: generated-text diagnostics joined to recorded benchmarks
- `analyze_samples.py`: reproducible lexical/format diagnostic

The checkpoints and Tiny Shakespeare corpus are intentionally not duplicated
here. Their exact hashes are recorded in `MANIFEST.json`.

## Recompute the generated-text diagnostics

Use the exact Tiny Shakespeare byte source whose SHA-256 is
`86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`:

```bash
python research/evidence/tiny_shakespeare_readability001/analyze_samples.py \
  --corpus /path/to/tiny-shakespeare/input.txt \
  --output /tmp/tiny-shakespeare-readability-metrics.json
```

Then parse and compare `/tmp/tiny-shakespeare-readability-metrics.json` with
the committed `METRICS.json` (line endings can differ by platform). The
corpus-vocabulary measure only checks whether an
alphabetic token occurs in the corpus; it does not measure grammar or meaning.

## Source

The source bytes were fetched from the public `karpathy/char-rnn` Tiny
Shakespeare fixture at:

`https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`

This post-hoc diagnostic records the byte hash and upstream location. It did
not independently re-audit upstream licensing or publisher provenance.
