# LANGCAP102 — Capability-first architecture research (31 October 2026)

**Status: DESIGN + NO-FIT HARNESS. Not a trained experiment, not a new result.**

## Decision and relationship to prior work

LANGMATH101 / [PR #679](https://github.com/FPC-effortless/TAC-OSM/pull/679) is preserved as historical evidence. The 135/138 original FIT passes and three separately time-amended recoveries are **not retroactively merged**. The exploratory mixed-cohort DEV evaluation is useful for showing that a 4-byte local model can win short-byte prediction under tiny (~100k parameter) / ~512-update conditions, **not** that it is a better general architecture. The ~100k, one-block LANGMATH101 Transformer is not a modern, pretrained Transformer and the diagonal selective-SSM control is not a faithful Mamba-2.

We will **not remove simple baselines**: they remain essential negative/shortcut controls under [the standing real-world evidence rule](../REAL-WORLD-EVIDENCE-RULES.md). They are *not* the headline contest or scientific objective. A candidate does not win merely because a window-4 model fails a purpose-designed task.

**Scientific question:** Under a legitimate information boundary, and with sufficiently trained, credible rival systems, which mechanisms acquire, retain, transform and deploy task-relevant information at better accuracy, robustness or resource cost?

### Two experimental lanes — never conflate their claims

**A. Mechanism qualification (new prospective scale v2; ~100K is optional).** A preregistered, source-audited, *genuine real-data* study. Train equally sized (within 1%) and properly optimized multi-layer attention, recurrent/SSM, and TAC variants on the same eligible training cohort. Use learning curves and equal-token/exposure and separately equal-measured-compute comparisons. A model too small or undertrained to learn an easy real-data positive-control task is **inconclusive**, not a failure of its architecture. Mamba-2/Titans labels may be used only for faithful versions with implementation parity evidence; otherwise explicitly name the approximation. No new fits are authorized by this document: preregister a separate `FAIR_COMPARISON.json` with `model_scale_policy: declared-target-v2`, explicit trainable/deployed targets and pass its real-source, size, math and resource gates first.

**B. Modern capability-reference and scaled-training lane (scale amendment approved).** A capable pretrained Transformer and a mature recurrent/SSM with matched context/evidence access are **external empirical references** when not matched to TAC's size and resource budget; larger trained-model comparisons can now use prospectively registered scale-v2 targets. They must be identified with exact model/weights revision, tokenizer, benchmark contamination/exposure, inference scaffold, context ceiling, hardware and costs. A 100M+ model must never be represented as a 100k matched control or slipped into a scratch-training experiment. The fixed 100K rule has been amended in a separate prospective policy; each new larger fit still requires a frozen target size, real-data evidence, measured compute budget, signed parity preflight and dedicated authorization before optimization.

## Primary task families

1. **Real long-document understanding:** annotated human-authored multi-document QA, dialogue history and code-repository comprehension; source provenance, unique labels, owner/time disjointness and adequate context must pass before scoring. Candidate: [LongBench v2](https://longbench2.github.io/) (503 human-curated examples). Public benchmark, so *exposed DEV research only*; unknown licensing/redistribution must be checked. Report by subtask, document length and evidence distance.
2. **Actual code understanding and transfer:** independently sourced repositories and semantics-based function search (e.g. [RepoQA](https://arxiv.org/abs/2406.06025)); measure function identification with semantic cues, unfamiliar repositories and owner-disjoint splits, not copied identifier lookup. Never assume human annotation provenance not independently checked.
3. **Evidence-dependent memory and multi-hop reasoning:** annotate evidentiary spans in *real* documents and keep question/answer independently labeled. Query text must not itself carry the answer. Require validated counterfactual removal of distant evidence to change oracle answerability; quantify shortcut leakage and retrieval cues. Report absence of answerability when evidence missing; do not fabricate clean natural gold answers.
4. **Adaptive compute:** on the *same* authentic task cohort compare fixed and adaptive routes using an accuracy–FLOP/latency/memory Pareto curve and conditional compute accounting; require causal route knockout and unchanged data access. Accuracy alone does not certify efficiency.
5. **Long-horizon agentic work:** real repository changes with pre-fix snapshots, issue timestamp, public test provenance and future-patch contamination audit. Do not use known exposed SWE-bench Verified scores as independent frontier evidence; newer evaluation and independent expert judgment are needed.

### Diagnostic-only benchmark tools

[RULER](https://github.com/NVIDIA/RULER), [BABILong](https://arxiv.org/abs/2406.10149), and [NoLiMa](https://github.com/adobe-research/NoLiMa) motivate long-distance, multi-hop and nonliteral-matching controls. Generated/synthetic examples can test wiring and the *ability of a benchmark to catch shortcuts*, never satisfy the primary real-world promotion gate. Do not copy protected labels into generated prompts or training. Genuine-source benchmark licence and reproducibility receipts must be checked independently.

## Model roster and fairness

- **Headline:** credible decoder-only attention, rigorously identified recurrent/SSM, native TAC-OSM, associative/hierarchical TAC and adaptive-token TAC. A lightweight one-block Transformer is called a lightweight control, **never** a modern-standard Transformer.
- **Diagnostics only:** last byte, last four bytes, shuffled history, query only, no memory, random/untrained, same-model state reset, relevant vs irrelevant history knockout, answer-in-query and label-swap checks.
- **Equal information:** same observable source bytes, event ordering, allowed external memory, query and generation conditions. Distinguish bounded-recurrent state from a Transformer with unbounded context; compare both an accuracy-unconstrained regime and a memory-budgeted regime. State size and external history/cache bytes must be counted.
- **Training fairness:** count active and deployed learned parameters; pretrained weights count; audit data exposure, optimizer schedule, actual FLOPs, training tokens, tuning budget and data contamination. CPU wall time is a cost, not a forced equality gate. Three independent seeds minimum for freshly trained comparisons, more when uncertainty is large. Multiple-testing correction and source-cluster confidence intervals.

## Mandatory *before* first training or primary evaluation

The `PROTOCOL.json` in this directory and `gate.py` provide **no-fit design lint**, not an authorization token. `python research/langcap102/gate.py design` checks the frozen design contract. This is the only implemented subcommand; it does not assess training readiness. Actual fit readiness requires the separate signed parity/source gate below. No original source bytes, gold labels or training data are downloaded or opened by design CI.

A separate preregistered run must supply:
- Source dataset version, license/permission, publisher, raw archive/content hash, real collected event identity, label availability; source owner/time split and zero target leakage.
- Human or independent answerability review; evidence locations, answer-in-query/leakage check, a strong capable positive control on easy slices, and a negative/shortcut control. Mark unanswerable or unable-to-verify cohorts **inconclusive**; no deletion after seeing model outcomes.
- Tokenization and context access equality; valid streaming vs full-prefix conditions; model source/weights hash, exactly counted parameters and resource ledger.
- Fixed cohort and task definitions, primary metrics and success/failure thresholds **before DEV**; protected independent TEST remains inaccessible.
- Recorded experimental uncertainty: paired across same cases/seeds, source-group bootstrap and multiplicity correction, quality–resource Pareto frontier. Do not cherry-pick only length bands or models.
- Failure retention and immutable artifact provenance.

## Precommitted interpretation rules

**A breakthrough** requires a reproducible, independently replicated improvement on a genuinely evidence-dependent real-data task against an adequately trained contemporary rival, with correct matched resources and no hidden outside-memory advantage. **A practical product gain** needs a measured production-relevant benefit (including total cost and reliability) on authentic tasks, not merely local next-byte accuracy. A failed positive control means the *benchmark/training regime is non-diagnostic*; an equal or worse TAC result despite credible controls is negative evidence about TAC on that domain. Report task-specific wins and failures separately.

## Current actions

- [x] Keep original LANGMATH101 data, runs and artifacts untouched.
- [x] Create source-controlled new program with no-fit governance gates.
- [ ] Acquire and independently verify permitted real source bytes, frozen splits and annotations.
- [ ] Demonstrate authentic positive-control solvability and shortcut ablations.
- [x] Amend the model-size rule prospectively through versioned scale-v2 declarations, preserving historical 100K runs.
- [ ] Preregister concrete scale-v2 matched model cohorts with credible Transformer/SSM comparators and measured resource budgets before any new training.
- [ ] Train across multiple budgets until interpretable learning curves, then evaluate under equivalent evidence access and report costs.
- [ ] Obtain an untouched independently scored cohort before claiming any capability or advantage.

**These checkboxes are not experimental results.** No model was trained, no DEV/TEST content was accessed and no model was promoted in LANGCAP102 at this stage.
