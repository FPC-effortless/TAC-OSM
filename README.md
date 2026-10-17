# DEV-049A / DEV-050A public development evidence

Thirty fresh fitted configurations across ten seeds, checkpoint inference and simulator replay, plus all three public Veritas engineering revisions. No sealed or canonical Veritas benchmark is included.

The raw archive contains `dev049/analysis.json`, all 30 checkpoints, raw probabilities/physical outcomes, ten independent replay reports, `dev050`, `dev050-v2`, and final amended `dev050-v3` native traces. An inner SHA-256 manifest covers 96 raw files. Source bundles retain the exact original local scientific and audit commits. Published source branches have byte-identical trees where connector publication produced additional commit identities.

Files are split into numbered binary parts because shell Git credentials expired during publication; the connected GitHub app published identical bytes. Clone this evidence branch and run:

```bash
python reconstruct.py
python -m tarfile -e dev049-050-raw.tar.gz extracted
```

`reconstruct.py` verifies every part and complete asset against `manifest.json`. Source bundles can be inspected with `git bundle list-heads` and cloned/fetched using standard Git.

Results and scope: PRs [#208](https://github.com/FPC-effortless/TAC-OSM/pull/208) and [#209](https://github.com/FPC-effortless/TAC-OSM/pull/209). These are development findings, not a zero-risk certificate, learned transfer result or frontier qualification. Checkpoint/simulator replay does not independently repeat optimization. Earlier Veritas runs and the amended protocol are preserved rather than relabeled as untouched confirmation.
