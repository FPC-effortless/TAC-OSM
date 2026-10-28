# TEMP021 — Original 500K frozen language calibration, exposed DEV only

Preregistration committed before any calibration fit on 2026-10-28. This extends TEMP020 #549 and must not alter original GRAPH013–019 weights or experiments.

## Recovered original evidence

- Original user-owned \`tac_language_long003_replay.zip\` SHA256 \`cde9a91be40a4e613975c0bd8caa4c19ac24e3d1d0e993db5cacfbd002fecc32\`.
- Original unmodified \`tac_language_500k/tac_wave_ced_500k.py\` SHA256 \`acf80cb9051b7f2d2c3900d1676eed212890208cd5213088c6751f9cfd24392c\`.
- Original immutable user-owned 500K trained checkpoints \`seed_1701.pt\` through \`seed_1705.pt\` inside archive, with original optimizer and Python/NumPy/Torch RNG. No model training/fine-tuning; only one scalar fitted per seed.
- Original authenticated Vim9.1 publisher corpus from \`/usr/share/vim/vim91/doc\`: all 30 FIT + 6 previously EXPOSED DEV chapters match byte SHA256 in \`tac_language_500k/SOURCE_PARTITION_MANIFEST.json\`. \`usr_45.txt\`, \`usr_50.txt\`, \`usr_51.txt\`, \`usr_52.txt\`, \`usr_90.txt\`, \`usr_toc.txt\` are known exposed DEV, not new TEST.

## Frozen disjoint roles before score

Lexicographic first 3 DEV chapters for scalar selection: **CALIBRATION = usr_45.txt, usr_50.txt, usr_51.txt**.
Remaining 3 **SCORE = usr_52.txt, usr_90.txt, usr_toc.txt**. Never fit scalar to score labels, change split after viewing NLL, use FIT labels or use protected TEST. For each document, use the exact earlier 500K score protocol: **eight length129 contiguous excerpts at NumPy int64 \`linspace(0, len(document)-129, 8)\`**, reset model state at each excerpt. Input x[0:128] at each excerpt predicts target x[1:129]. Original byte mapping: values >=128 mapped to ID63 \`?\`. Expected mapped targets: 3 docs*8 windows*128 next symbols = **3072 CALIBRATION and 3072 SCORE per seed**. Same documents and offsets for all five seeds; repeated labels across seeds are not independent cases. Keep each document's provenance and source hash in a machine-readable receipt.

Before optimization, strictly verify frozen original checkpoint SHA256 against the archive's independent member manifest, model and source bytes, 36 authentic chapter SHA256 identities, and original complete-six-document 6144-target NLL within 1e-6 of checkpoint's original \`dev.nll\` (handling expected float-reduction tolerance).

## Algorithm and evaluation

Use TEMP020 pure mathematical contract unchanged, calibrated by inverse temperature a in [.25,4] with 64 deterministic bisections on calibration CE gradient, comparison against endpoints and identity a=1, choose identity in ties. Equivalent batched FP64 implementation may accelerate but must compare against TEMP020 \`src/tac_osm/scalar_calibration.py\` on a deterministic genuine-source subset (derivative, loss, selected scalar tolerance), and it must preserve exact fit/scoring role disjointness, immutable trace hash and finite-law controls. Compute per-seed fitted a, CALIBRATION NLL, disjoint SCORE NLL for a vs identity, per-score-document NLL, five-seed mean, top1/top5 (must be unchanged under positive scalar except floating ties), categorical Brier/ECE if measured, logit and fitted scalar hashes, all failures. Do not report raw BPB for non-injective codec. Report bits per *mapped* symbol only.

Primary effect: mean 5-seed SCORE NLL(a)-NLL(identity), source-window weights fixed; improving negative is provisional **EXPOSED DEV** only; calibration fitting improvement is guaranteed by fallback and does NOT count as held-out skill. Preserve adverse scores per document/seed. Original old checkpoint replay not the same as Git/LFS repository cold replay: \`REPO_REPLAY=BLOCKED\` unless original trained binary files and lawful source available to independent clean checkout. No PNDS whole-model IEEE/AD proof claimed; scoped Lean proofs #61 unchanged. No GPU, new neural training, original checkpoint mutation or TEST exposure.

Study is temperature calibration, not new language-model capability or native intelligence. Source arch already exposed, so no independent cohort qualification. Stop rather than substituting synthetic metrics for real measurements.
