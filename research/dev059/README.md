# DEV-059 recorded research evidence

The full report is [DEV-059-RESULTS](../../docs/DEV-059-RESULTS.md).
The main grid did not pass its overall superiority gate. DEV-059-R failed its
physical-reliability guard. Failed weights are research evidence, not an
approved model update.

- `protocol.json`: prospective five-arm, ten-initialization grid registration.
- `repair/protocol.json`: separately registered inherited-checkpoint fine-tune.
- `statistics.json`: all seed/cell summaries, paired inference and gate results.
- `evidence-index.json`: archive digest and restored replay counts.
- `evidence.tar.gz`: all 60 grid fits and their recorded evaluations, two rejected
  fine-tunes, same-world teacher/public reference and retrospective DEV-058 audit.
- `learning-curves.png`, `seed-completions.png`: all main seeds, without selection.

To reproduce without training, install the project, pinned CPU PyTorch and the
Veritas commit in `.github/workflows/dev059-reliability-controls.yml`, then:

```sh
mkdir -p results
tar -xzf research/dev059/evidence.tar.gz -C results
tar -xzf research/dev058/evidence.tar.gz -C results
python scripts/audit_reliability_controls_dev059.py \
  --directory results/dev059-main --repair-directory results/dev059-repair \
  --veritas-root /path/to/pinned/veritas --output results/dev059-recheck.json
python scripts/summarize_reliability_dev059.py \
  --directory results/dev059-main --output results/dev059-recomputed-statistics.json
python scripts/audit_reference_dev059.py --output results/dev059-reference-recheck.json
python scripts/audit_evidence_gaps_dev059.py \
  --directory results/dev058-main --output results/dev059-dev058-recheck.json
```

Compare recomputed statistics to the frozen `statistics.json` and reference to
`results/dev059-main-reference.json`. Floating diagnostic probabilities allow
`rtol=1e-5, atol=1e-6`; all recorded action choices, receipts, costs and work
counters must match exactly. Native effect replay uses the same synthetic laws.
The dataset seed/configuration fingerprints are in the archive; redundant
training tensor caches are excluded. Training source and checkpoint digests
remain recorded per fit. No test-driven model replacement was accepted.
