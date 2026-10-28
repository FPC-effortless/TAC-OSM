# GRAPH017 — preregister native multi-horizon predictive-state development

Date: 2026-10-28. System-1/language lane only; no seventh TAC family. **Frozen before candidate fitting.** Stacked on GRAPH016 #545. Keep GRAPH009–016 positive and negative evidence unchanged.

## Objective and useful ability
Train an **integrated 500K native TAC** to predict actually observed future bytes at h=1,4,16 from the same causal recurrent/graph representation. This is direct multi-horizon *marginal prediction* (not free-running generation or reasoning). Current GRAPH015's bounded delta-memory training failed relative to prior TAC; avoid further retention tweaks without task-relevant gradients. Use original WaveCED500K recurrence and concept/wave state, and add two narrow explicit horizon-conditioned readout heads from the decoder's 280-dim causal state.

Fit loss: L=L1 + 0.15 L4 + 0.15 L16, with each categorical CE averaged over every valid source-known input position with a future observed target within the **same original real 129-byte training window**. A h-step target at observation ordinal t is x[t+h]; h=1 is original task, and t+h <129 required. No training future value can be passed to the prediction function. No loss on truncated/censored targets, no retroactive label imputation. All current trained coordinates must receive useful gradient; no dummy params.

New live parameter target 500,000 ±5% [475k,525k]; original baseline 499,290. Count actual trained and deployed coordinates. Two low-rank horizon heads: for each h=4,16 a 280->16->128 MLP with GELU, no hidden source outside decoder. Main 1-step head remains original and unmodified mathematically. Model registration and all output semantics versioned; original checkpoints immutable.

## Exposure and experiment
- Original genuine *publisher* Vim9.1 manual with 30 historical FIT chapters and six **previously exposed DEV** chapters; original source SHA256 36/36 and original fit/DEV document identity/split restored from the existing source partition manifest. DO NOT introduce source from new TEST.
- Five seeds 1701–1705; original sampled FIT source and offset schedule exactly 120 updates × batch8 × context128 and same AdamW lr=.001, optimizer wd default and clip norm1 as original; horizon labels use the same 129-byte sampled real windows. Preserve sampling SHA. No extra update/tuning on exposed DEV.
- One precommitted arm (joint loss .15,.15); **no sweeps, seed cherry-picking or hyperparameter adjustment from DEV**. Report all attempted seeds, failed fits, checkpoint/optimizer/RNG and source SHA records.
- Exposed DEV evaluation uses the original eight fixed 129-byte windows from each of 6 DEV documents; report h1,h4,h16 NLL, Brier, top1 and counts separately. Expected totals: 48*(129-h) targets/horizon per seed (6144,6000,5424).
- Comparator on identical h=1 source positions: five frozen original WaveCED500K checkpoints, previously recorded original native mean 2.53744 on exposed DEV; independent rescoring preferable. Both trains have unequal objectives and supervision; this is a development *whole-model* tradeoff, not a matched architecture-superiority result.
- Practical future forecast reference: source-only smoothed 3-gram future-outcome frequency estimators at h=4,16 trained solely on 30 FIT whole chapters, with original source identity and Laplace smoothed FIT global label prior. Report whether the neural direct horizon distribution beats the simple same-task forecast; simple reference uses more FIT positions and does not establish architecture superiority.
- No new CMake/Python first-open or sealed cohort: GRAPH015 and GRAPH016 opened those sources and they remain EXPOSED. Nothing from them is used for selection or training.

## Registered decision
1. Native h1 guardrail: five-seed average NLL must not worsen by >0.01 nats/byte relative to frozen original model, and no more than 1/5 seeds materially regress by >0.03. Brier/accuracy also reported. Failing this bars "overall prediction improvement".
2. H4/H16: require NLL advantage >=0.01 nats/byte against frozen FIT-only predictive 3-gram baseline **at both horizons**, in >=4/5 seeds, before calling these learned forecasting skills (development only). Do not invent an independent confirmation from constructed data or software tests.
3. If either fails, classify **MIXED/NEGATIVE** with original outcome preserved; no further tuning on exposed DEV. A software test pass only certifies tensor causality/finite gradient/valid law.

## Math to check
For time t and horizon h, output `p_h(y|x_0:t)=softmax(W2_h GELU(W1_h z_t))`; z_t computed without future x. Softmax is normalized and positive for finite logits (up to FP precision). Matrix-scan chronology unchanged; h-heads are reads only, they cannot write future-target values into state. Supervision masks t+h within original source window. Verify prefix causality by future-token poisoning and nonzero gradients for both horizon heads. If external loss surgery is added later, it requires a distinct preregistration; here a fixed joint loss is tested to measure gradient interference honestly.

## Compulsory unresolved gates
Real DEV already exposed: not independent improvement. Full forward+backward+optimizer FLOP parity with original is not enforced (different objectives), document measured resource use. Git/LFS original trained checkpoint and lawful clean-checkout replay mandatory for promotion; local ZIP+hash or source-only CI != repo cold replay. PNDS complete Lean/floating-point/autograd mathematical correspondence OPEN. No commercial value or general intelligence claim.
