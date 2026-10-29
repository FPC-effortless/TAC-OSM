# LANGDECODE063 — source-matched shared-upper prototype

**Status:** PROPOSED / CODED; independent original-checkpoint GitHub CPU replay **NOT YET VERIFIED**. See preregistration and workflow artifact after CI.

The previous [LANGDECODE062 negative](https://github.com/FPC-effortless/TAC-OSM/pull/601) remains retained unchanged, not combined into a falsely positive measurement. A 3x ordinal kernel speedup yielded only 4.35–4.49% full median benefit and failed p95. New hypothesis: eliminate one duplicate upper normalization+projection per emitted token without changing any of the original state transitions, logits, or its 3904-byte FP32 state.

Acceptance is gated by source/checkpoint bitwise correctness AND 5% median improvement AND <=1.10 p95 ratio at both 128/512 actual original FIT prompts, followed by independent repeatability and PNDS review; no outcome yet and no promotion.
