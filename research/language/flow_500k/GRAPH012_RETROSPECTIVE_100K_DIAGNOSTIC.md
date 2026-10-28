# GRAPH012 — historical 100K equal-cache exploratory diagnosis (28 October 2026)

**Tier: AUTHENTIC PREVIOUSLY EXPOSED DEVELOPMENT ONLY; NOT 500K OR ARCHITECTURE QUALIFICATION.** This frozen read-only calculation neither retrains models nor opens protected TEST. The registered prospective 500K same-cache comparison remains pending; original GRAPH009–011 negative and positive results stay unchanged.

## Source and original-artifact audit

- Authentic Vim 9.1 manual: all **36/36** original chapter SHA256 identities checked against the archived 100K source protocol.
- Historical fitted models: five 100,680-parameter TAC language checkpoints and five 100,692-parameter custom selective diagonal SSM checkpoints, each byte SHA verified against the original archived RESULTS_CURRENT.json.
- Same original sampled FIT sequence per TAC/SSM seed verified by archived sampling SHA. Training losses were **NOT comparable**: TAC also had lag64 auxiliary historical reconstruction, whereas SSM trained on next-char CE only. Full training FLOP/tuning parity unavailable.
- Retrieval: five FIT-trained GRAPH011 learned-query checkpoints (2,816 weights each), SHA256 verified against GRAPH011_ARCHIVE_MANIFEST.json. Shared FIT prior identical across seeds. Both backbones given exactly the same per-token source-only cache and approximate query distribution.
- Previously exposed DEV: six actual manual chapters, eight length129 windows/chapter; 6,144 next-character targets/seed. Each window resets. Exact-count order3, capacity512, tau=.05, lambda=.6; learned approximate top8, temperature .25, lambda=.1 on exact misses only. 813 exact-cache hit positions/seed (13.23%).

## Mean five-seed observed NLL — lower better

| Condition | Historical TAC 100K | Custom SSM 100K | TAC minus SSM |
|---|---:|---:|---:|
| Native frozen backbone | 2.815531 | **2.524472** | +0.291058 |
| Both + same exact count cache | 2.699896 | **2.436128** | +0.263768 |
| Both + same exact + learned approximate cache | 2.688935 | **2.440045** | +0.248891 |

The external count cache improves **both** models. The shared trained approximate query additionally improves TAC's NLL by ~0.01096 but worsens SSM's by ~0.00392 on these already-examined windows. The custom SSM still leads after both cache stages for **all five paired seeds**.

This is useful evidence of an *interaction* between the frozen native predictor and exactly the same external adapter, not proof that TAC has superior native memory, nor a legitimate modern Mamba comparison. It does not replace original 500K pilot numbers, and 100K/500K scores cannot be compared as the same controlled experiment.

## Repeatability and gates

Original artifact ZIPs, currently in the user's file library, are:
- tac_language_current_math_benchmark.zip (historical 100K trained models and controls).
- graph011_five_seed_external_evidence.zip (five learned query model/optimizer/RNG checkpoints).
- TAC_500K_GRAPH009_Source_and_Replay.zip (source count cache reference).

The separate conversation evidence bundle includes the **read-only 100K diagnostic runner**, full per-seed/per-document JSON results, and original ZIP SHA256 receipts. This supporting material is not Git-backed and does **not** satisfy the independent repo-cold replay rule. Only a source-independent code/checkpoint audit plus Git-held original bytes can resolve it.

**Explicit gates:** REPO_REPLAY=BLOCKED; TEST=NOT_RUN; TRAINING_COMPUTE_PARITY=NOT_ESTABLISHED; ARCHITECTURAL_SUPERIORITY=NOT_ESTABLISHED. No merge, scientific promotion or production language-model claims.

Next: obtain original frozen **500K** TAC/SSM checkpoint/optimizer/RNG binaries, matched objectives/FLOP receipts, source hashes and lawful new TEST documents; then execute the registered GRAPH012 primary comparison. Do not use these exposed Vim DEV windows as fresh evidence.
