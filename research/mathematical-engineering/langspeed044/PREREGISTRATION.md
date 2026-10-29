# LANGSPEED044 — original-trained cached inference, prospective

Registered **before this study's first trained CPU measurement**. This is a no-fit internal research continuation of #572, not training, protected TEST use, independent qualification or model promotion.

## Question and controls

Does source-faithful persistent state deliver an inference-cost advantage against a **real trained Transformer with proper KV caching**, under identical growing-prefix semantics? Run same physical Ubuntu CPU, PyTorch 2.10 eager FP32, two threads, batch1, original PSF CPython 3.13.0 FIT owner and prompt bytes, lengths 128/512, 16 greedy continuation updates, two warmups plus seven measured repeats with alternating model order. Each architecture chooses its own generated continuation; no cross-model text identity is required.

Frozen originals, seed1701, 320 AdamW updates with same original 14-source seed schedule, batch8 and 128 original next-byte targets per example:
- TAC LANGLAYER040 **state48_no_graph**, 499,623 learned params, streamed via LANGSTREAM043 cap_all profile; original checkpoint from GitHub CPU run 40797105091.
- LANGCOMPARE031 **causal Transformer**, 496,960 params, four 96D pre-norm TransformerEncoder layers; original checkpoint SHA256 ab36031debf5b68abc25dfc3c8ca362110854b49877e2b2f19e3df806e7786ae from run 40780215655 and source Git blob 935decb17dfac48419a82d84dc03aaf4b0327ecd pinned at upstream experiment commit 1ce108a3696db91c9bc21b9c9d4d5703a7db8d81.

Model parameter gap <1%; training FLOPs are not matched. Check binary hashes, original optimizer/RNG and source order before inference. **Mandatory gate:** before timing, compare last-byte logits for all 16 incremental updates and both prompts against their own original full-growing-prefix model; original FP32 tolerances atol3e-5,rtol1e-4. Any mismatch stops ranking. No gradients, source substitutions, synthetic competence claims or mutable checkpoint edits.

Report raw timing vectors and median/p95 per-token latency, separate bulk prefill, setup/verification overhead, prefill+generation, actual retained state (KV growth versus constant TAC), original machine and source/checkpoint identities. This is present implementation efficiency, not theoretical asymptotic or phone energy.

## Honest exclusions

Official BDH has not yet passed trained original-source incremental-cache parity; **BDH cached NOT RUN**. Official Mamba-3 lacks a qualified matching CPU implementation; **SSM cached NOT RUN**. Do not use unoptimized earlier BDH measurements as a cached competitor or invent Mamba benchmark. Fresh untouched quality evaluation is separate; original CPython DEV already exposed. No architecture superiority, memory-retrieval or deployment claim.

## Next study — LANGMEMORY045

Register a source-grounded cheap complementary two-memory + selective costly verification architecture only after this benchmark. Compare no-memory, single-memory variants, paired memories without verification, paired memories with learned budget-gated verification, shuffled-key/donor controls and equivalent parameter/computation budgets. Target truthful distant source-byte retrieval and genuine new-document NLL, not internal cosine similarity. Require original real publisher source, nonleaking time boundaries, five+ seeds, source and checkpoint replay, PNDS full math obligations and fresh untouched evaluation. No fit or capability result in this PR.
