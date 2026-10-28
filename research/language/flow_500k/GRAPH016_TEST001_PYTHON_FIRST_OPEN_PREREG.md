# GRAPH016-TEST001 — frozen prospective Python 3.13 source for causal online expert mixture

**Registered before opening any target bytes (2026-10-28).** This study is System-1 / language inside the six TAC families, not an architecture replacement. **Do not tune policy from these documents afterward.**

## Fixed question and intervention

Evaluate classical *causal Bayes expert weighting*, three frozen source-known forecast laws per backbone:
(1) native TAC or original SSM; (2) native+exact GRAPH009 3-gram count cache with fixed lambda=.6, tau=.05, capacity512; (3) same exact-priority cache plus frozen GRAPH011 learned 2,816-parameter approximate query, lambda=.1, top8, temperature=.25.

At prediction position t, with expert probabilities p_i(.|h_t), owner-local weights w_i(t) are normalized products of only **previously observed targets** in that same owner. The caller must produce prediction before observing y_t, then the weights update by w_i(t+1) proportional to w_i(t)*p_i(y_t|h_t); never allow current y_t or other owner labels into prediction. Initialize equal weights 1/3 per source window, and reset on every new window. This is a **fixed rule, no model training, no new neural learned parameters**. Test original frozen five 499,290-param WaveCED TAC, five new GRAPH015 499,290-param bounded delta TAC, and five new GRAPH014 498,288-param selective SSM; all from original researcher-owned archives, no refitting. Keep identical source-only cache/query predictions per seed across backbones. All original checkpoint hashes must match existing GRAPH013–015 ledger; source-only user files not redistributed. The Graph011 learned-query model was trained solely on Vim FIT, earlier external Python files and CMake docs are already exposed.

Per-owner exact arithmetic invariant with three fixed experts:
  L_online(owner) <= min_i L_i(owner) + ln(3).
This is a **pathwise no-regret upper bound relative to the best fixed expert in the same owner's actual observed sequence**, not a bound to the SSM, or ability to switch between arbitrary experts for free. Positive normalized expert categorical laws, source authorization and no current-target access required. FP64 implementation checked separately.

## Truly new original Python source selection frozen before first open

Original Python 3.13 stdlib source files installed in /usr/lib/python3.13, separate from GRAPH011's exposed 8 Python source files and GRAPH009's exposed six: choose only these **eight** actual original package-owned files, registered from pathname/size metadata without inspecting bytes or predictions:

1. asyncio/base_events.py (80,245 bytes)
2. asyncio/streams.py (28,481 bytes)
3. email/message.py (48,215 bytes)
4. json/decoder.py (12,872 bytes)
5. email/utils.py (16,615 bytes)
6. urllib/parse.py (45,676 bytes)
7. http/server.py (48,992 bytes)
8. concurrent/futures/_base.py (22,845 bytes)

None is in GRAPH011 eight new files (asyncio/tasks.py, csv.py, email/parser.py, http/client.py, logging/__init__.py, re/_parser.py, textwrap.py, urllib/request.py), or GRAPH009 six previously exposed files (argparse.py, ast.py, inspect.py, random.py, typing.py, doctest.py). Package provenance and rights are public PSF Python standard library; verify version/sha and report. These are *new documents* for these experiments, but domain (Python source code) was seen in GRAPH011, so no claim of wholly new domain/independent institutional data.

## Immutable scoring procedure and success rule

- Exactly eight 129-byte raw-file windows per document at **np.linspace(0,file_length-129,8,dtype=int64)**, no label-based selection.
- Map bytes >=128 to character ID63 as the frozen training pipeline did. Each window's 128 inputs causally predict the next 128 authentic byte labels, one-step ahead. **8,192 real next-byte labels/seed/backbone**; 64 independent state-reset windows per seed, eight original documents as source clusters. No source-model fine-tuning.
- Primary study on **original TAC**: compare online mixture's NLL to the best of the three whole-evaluation *fixed* native/exact/exact+approx policies (report name). Predeclared material win requires ≥0.005 nats/byte NLL improvement on the five-seed average, improved direction in at least 4/5 paired seeds, and upper endpoint of a descriptive 95% two-level bootstrap across whole source documents and whole training seeds below -0.005. Also report accuracy, Brier, any potential drift, run time and memory.
- Secondary: repeat identical scoring for delta TAC and SSM, report per seed and per document; compare to static uniform blend as non-adaptive control; exactly report any negative.
- Independently check per-owner log-loss regret <=ln3 in actual FP64 scoring. The best-fixed baseline in hindsight is not deployable and cannot itself depend on future labels for an online choice.
- After first read, archive local original SHA256 values and score outputs, mark these documents EXPOSED forever; no re-tuning of the online prior/hyperparameters from outcomes.
- Strong scientific promotion still BLOCKED until Git/LFS original checkpoint cold replay, untouched external authority/source, and matched compute controls; local first-open data and mathematical tests cannot establish native architectural superiority, intelligence generally, long-history facts, customer ROI, or production trust.

**Deliberately reserved CMake/Vim evaluation**: previous GRAPH013–015 Vim DEV and CMake first-open sources are EXPOSED; use only as engineering retrospectives, do not pick policy based on them for this frozen test. The rule was selected before this Python TEST opening.
