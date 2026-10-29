# LANGMEMORY051 — exact source key residency is the dominant diagnostic

**Actual original publisher source, internally measured 29 October 2026.** [Draft PR #583](https://github.com/FPC-effortless/TAC-OSM/pull/583), [successful CPU Actions #40829184373](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40829184373), original [artifact #12876904969](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40829184373/artifacts/12876904969). Eight constructed causal/no-future/no-eviction/key-capacity/owner tests PASSED; original publisher source/licensing, scientific preflight and phone no-fit research planner passed. No trained TAC/Transformer/SSM/BDH checkpoint loaded, training, GPU, DEV/TEST, VRAM benchmark or model promotion.

**Authentic original 14 CPython3.13.0 FIT owners**, already used repeatedly in 045–050; this is not an independent held-out scientific generalization. True original x[t] scored after reading only actual earlier x[:t], with full 3-byte exact-key identity per owner. Prefix empirical entropy and counts used ONLY prior source observations. Anchor <=1536: 4928 eligible next-source-byte positions; extension <=4096:13245; genuinely longer source prefix <=8192:11784. Original unbounded exact history and 050 original bounded K1 controls replayed to identical correct counts.

## Results: original real-source 8192 prefix (11784 next-byte source targets)

| Arm | Correct of 11784 | Covered | Previous actual key retained among 8243 recoverable positions |
|---|---:|---:|---:|
| Bounded K2 two-way 1344B + recent64 |1564|2409|2086 (25.31%)|
| Bounded K3 two-way 1344B + recent64 |1432|2200|1752 (21.25%)|
| Bounded K2 two-way 4096B + recent64 |2703|4248|4274 (51.85%)|
| Bounded K3 two-way 4096B + recent64 |2454|3855|3809 (46.21%)|
| Previous bounded K1 two-way 1344B |2148|3297|3342 (40.54%)|
| Previous bounded K1 two-way 4096B |3343|5151|5695 (69.09%)|
| No-eviction MG K2 + recent64, **UNBOUNDED** |**4792**|7962|8243 (100%)|
| No-eviction MG K3 + recent64, **UNBOUNDED** |**4828**|8134|8243 (100%)|
| Exact original full-prefix mode, UNBOUNDED |**4861**|8243|8243 (100%)|

Only **2086 of 8243** source-recoverable context keys remain resident in the bounded K2 1344B table; **6157** recoverable original keys are missing (74.69%). In contrast, disabling all key eviction but leaving the *same K2 Misra–Gries continuation-update rule* raises the correct score from1564 to4792 out of11784; only69 correct below the exact majority reference4861. This is strong evidence that finite key survival/collision is the dominant failure mechanism in this source task. **It is NOT a fixed-memory win**: the no-eviction oracle has a growing Python dictionary and unlimited key storage.

## Prior-key empirical continuation entropy, source-only

| Property | 4928 anchor | 13245 extension | 11784 longer original prefix |
|---|---:|---:|---:|
| Distinct/new original context queries |2451|4839|3541|
| Previously seen/recoverable exact contexts |2477|8406|8243|
| Repeat contexts with >1 previously seen continuation value |891|3950|4459|
| Repeat contexts with >=3 previous continuation values |373|2160|2747|
| Prior-only mean per-query conditional continuation entropy on repeat keys (bits) |0.4395|0.6478|0.8019|
| Exact historical majority correct |1465|4943|4861|
| Exact historical latest observed continuation correct |1414|4783|4715|
| Positions where majority correct and latest wrong |115|497|594|
| Positions where latest correct and majority wrong |64|337|448|

At long horizon 4459/8243 (54.09%) recoverable queries genuinely had >1 earlier continuation; 2747 had >=3. The *median/average entropy is not zero*, so K value diversity matters for exact frequency, but unbounded K2 MG almost closes the gap to exact original key majority. Last-only source value predictions trail majority (4715 vs4861) even with unbounded full history. Original key identities have to survive before value modeling can help.

## Technical limitations and falsifier

Existing bounded models' 'correct' includes an exact recent64 byte-ring fallback, so key-residency and correct predictions are overlapping conditional factors; naive subtraction cannot be called a causal effect size of a single intervention under fixed total bytes. The no-eviction K2 oracle removes eviction by **adding unbounded key memory**, violating matched storage and update work, and its conditional-frequency approximation is not itself a trained model. The 1344B bounded K1 has more key slots (160) than K2 (80) or K3 (64), while 4096B has K1=504, K2=252, K3=200. Key hash congestion/recency policies and source repetition jointly affect results; do not attribute every missing key uniquely to 'hash collisions' or every failure to 'compression' without new controls. Original current source owners have been exposed repeatedly; no new publisher-heldout result.

**Decision:** stop adding more continuation slots in existing finite hash storage. To improve the key-residency versus retrieval-cost frontier, investigate byte-efficient exact-key packing, key priority based on ONLY past frequency/recency, and collision-aware eviction, while matching bytearray payload and measured reads/writes. If none beats equal-byte raw ring and K1 baseline on **untouched** publisher owners, preserve negative; only then consider actual trained 500K TAC neural integration with PNDS Lean/Python full typed neural graph and source owner-disjoint, comparable checkpoints. The future GRAPH011 recomputation proposal is orthogonal: activation/checkpoint replay versus persistent source-key capacity, and no current model-VRAM savings are established by this source task.

## Source evidence

Original GitHub successful workflow receipt and original numeric per-owner JSON preserved; no signatures fabricated and original checkpoint replay unaffected. Complete original artifact ZIP saved in persistent ChatGPT Library at `/TAC-OSM/Research Archives/LANGMEMORY046-047/LANGMEMORY051-run40829184373-raw-evidence.zip` beyond 90-day Actions retention. See SHA256/source log ledger committed with this PR.
