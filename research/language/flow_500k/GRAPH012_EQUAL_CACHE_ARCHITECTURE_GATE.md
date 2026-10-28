# GRAPH012 — separate neural backbone value from an identical source cache

**Status (2026-10-28): preregistration/scoring infrastructure, NO executed 500K TAC-versus-SSM comparison.** This belongs to the existing System-1 / language research family. It is not an additional seventh variant. GRAPH009/010/011 negative and positive evidence stays unmodified.

## Why this study must be done

GRAPH009's 3-character count cache improved a frozen 500K TAC's genuine-source log loss; GRAPH011's trained 2,816-weight approximate-query head also beat untrained and exact-only source-retrieval controls on eight newly frozen Python files. **GRAPH011 external scoring did not load the 500K TAC language backbone.** Neither result isolates an advantage of TAC's internal memory over the equally cache-equipped selective diagonal SSM. That is the next falsifiable hypothesis, not a known fact.

The previously reported 500K native-only study on exposed Vim DEV (5 seeds, 120 updates, context128) was: custom selective diagonal SSM 2.39256 NLL, original TAC 2.53744, GRU 2.72739 and a tiny scratch Transformer 2.93553. It was **not FLOP-equal**, does not imply superiority over modern trained Mamba and has no GitHub-hosted 20-model original checkpoint cold replay. Reusing it as new independent confirmation is prohibited.

## Registered primary comparison and prediction law

- Freeze real whole-document FIT, DEV and **new genuinely untouched TEST** identities, licenses/legitimate acquisition, SHA256 manifests, chronological availability and source owner boundaries. Previously exposed Python eight and Vim DEV are diagnostic-only.
- Freeze five paired seeds and model training schedules. Compare trained 500K TAC and the existing trained custom selective diagonal SSM, with 1% active-parameter gap maximum (count the 2,816 common learned query coordinates on **both** when totals are reported). Keep training tokens/sampled sequence schedule, objectives, optimizer updates, tuning effort and full forward+backward+optimizer FLOP budget matched; *record* any mismatch, never call such a result fair. CPU wall time is an efficiency outcome, not a required equality constraint. A modern Mamba-family implementation is an important additional comparator, not interchangeable with the currently reported custom SSM.
- Source cache is instantiated once, from the same lawful observed tokens and frozen source boundaries: order3, 512 admitted events per owner, tau .05, exact hit gate lambda=.6, trained approximate-query lambda=.1 on **exact misses only** with topK8, temperature .25, identical pretrained query weights and prior. Freeze all source/query-model binaries and hashes. The cache cannot access target labels at inference. True authorization is external; `source_ids` strings alone are insufficient.
- Record each frozen model's raw 128-way next-byte probability **for the same observed prefix/target** and the **same** exact and approximate cache posterior arrays per token. The new offline scorer applies the *identical* probabilities/gates to the two native models. It never runs the training job, fills gaps, changes labels or tunes mixture weights.

For backbone `m`, fixed prior `u`, shared exact posterior `c`, approximate posterior `q`, and source-conditioned mutually exclusive indicators `e,a`, the registered intervention is:

```
p_m,exact = (1 - .6 e) p_m + (.6 e) c
p_m,both  = (1 - .1 a) p_m,exact + (.1 a) q
```

The **cache-only** condition replaces `p_m` with the **same FIT prior** `u` and uses the exact same gates. Report all three conditions (native / native+exact / native+exact+approx) for TAC, SSM, and prior-only. Define:

- `Delta_native = NLL(TAC native) − NLL(SSM native)`;
- `Delta_cache = NLL(TAC+same cache) − NLL(SSM+same cache)`;
- `Interaction = Delta_cache − Delta_native` (additional influence of the common cache on the model gap);
- native incremental value for each model = `NLL(FIT prior+shared cache) − NLL(model+shared cache)`.

Only a **negative** `Delta_cache` would establish better predictive NLL for this TAC model against this SSM under the controlled cache, *conditional* on matched training/data and adequate independent evidence. A stronger cache-only score by itself does not mean TAC memory is useful. Even negative `Delta_cache` is not proof that native associative memory specifically caused the gain: additionally run frozen direct associative-memory erasure, wrong-key, owner-disjoint and state-reset interventions on identical real targets, including source ages 64/192/512/2048 where legal. Test task relevance: random long delays or synthetic keyed strings are diagnostics, never evidence of general-world utility.

## Offline scorer and strict boundaries

`equal_cache_score_v1.py` accepts a JSON bundle with:

```
{
  "schema": "graph012-equal-cache-v1",
  "policy": {
    "exact_weight": 0.6, "approx_weight": 0.1,
    "cache_capacity": 512, "cache_order": 3, "tau": 0.05,
    "query_topk": 8, "query_temperature": 0.25,
    "query_checkpoint_sha256": "64 lowercase hex characters",
    "fit_prior_sha256": "SHA256(json.dumps(fit_prior,separators=(',',':')))"
  },
  "fit_prior": "list of 128 strictly positive normalized probabilities",
  "data": {
    "source_manifest_sha256": "64 lowercase hex characters",
    "scoring_protocol_sha256": "64 lowercase hex characters",
    "split": "DEV", "prior_test_exposures": true
  },
  "models": {
    "tac": "architecture, active_parameters, training_updates, training_tokens, training_flops, fit_sample_sha256, checkpoint_sha256",
    "ssm": "exact same fields"
  },
  "rows": [{
    "owner_id":"source-owner", "source_sha256":"64 lowercase hex characters",
    "ordinal":0,"target":0,
    "native":{"tac":"128 positive probabilities","ssm":"128 positive probabilities"},
    "exact_hit":false,"approx_hit":false,
    "exact_posterior":null,"approx_posterior":null
  }]
}
```

When `exact_hit` or `approx_hit` is true, replace the corresponding null posterior with 128 positive normalized probabilities. They **cannot both be true**. The scorer verifies owner/source/ordinal uniqueness, shared cache policy, FIT prior content hash, both model metadata, all probability laws and joint paired target alignment. It emits paired native/exact/both NLLs, source-level slices, interventions, source count, an input digest and visible fairness blockers. It always says `SCORING_ONLY_NOT_ARCHITECTURE_QUALIFIED`, since a supplied JSON alone cannot demonstrate authentic training, checkpoint replays, truly causal cache construction or sealed TEST integrity.

**Do not pass model-specific independently generated cache arrays to this scorer.** The upstream runner must instantiate a single shared cache/query scan and export one posterior/gate per observed source prefix. All raw predictions must be bound to the same source bytes and absolute ordinals. Re-run the exact scorer on sealed source **only after** a registered experimental protocol has been frozen, and never edit thresholds after seeing TEST.

## Promotion/failure decision

Minimum: five independent model-paired training seeds, independent untouched real documents and genuine source-ID memory queries; source-clustered uncertainty intervals, full NLL/accuracy/calibration; raw/exact/learned-query/no-neural controls; wrong-address/owner/state-reset lesions; training FLOPs and end-to-end p50/p95 latency/throughput, checkpoint+source bytes and process RSS. Every checkpoint and legitimate original source must independently cold-replay under CI with immutable hashes. Document all failed or noninferior conditions.

**Current blockers:** exact original 500K paired training-checkpoint binaries/optimizer RNG and authenticated source files aren't in GitHub independent replay; the SSM training protocol is not yet installed as a new paired full-model experiment; the external GRAPH011 eight files have already been examined and cannot become a fresh sealed TEST; actual trained memory lesions at long delays remain negative/incomplete. No model, semantic benchmark, client SDK or commercial superiority should be promoted by merely passing the scorer unit tests.

## Five-seed rollup and source-cluster uncertainty (GRAPH012 implementation increment)

\`graph012_rollup_v1.py\` consumes exactly five \`equal_cache_score_v1.py\` trace documents with distinct integer \`seed\` values, identical \`(owner_id, source_sha256, ordinal, target)\` rows, data split, manifest and FIT prior, and matched within-seed source sample schedules. It refuses unequal target identities, duplicate seeds and incomplete studies. It calls the source scorer for each seed rather than trusting reported losses.

Outputs: all five raw native/equal-cache NLL comparisons, pooled paired differences, source-macro paired differences and a reproducible **descriptive** two-level bootstrap interval, resampling whole source documents and whole seeds (never individual overlapping character positions). No interval is printed for fewer than eight source documents; a computed interval is **not** permission to make a capability or independent TEST claim.

Run (only after obtaining legitimate, originally replayed predictions):

\`\`\`sh
python graph012_rollup_v1.py seed_1701.json seed_1702.json seed_1703.json seed_1704.json seed_1705.json --output five_seed_report.json
\`\`\`

The five-seed rollup inherits the mandatory \`SCORING_ONLY_NOT_ARCHITECTURE_QUALIFIED\` evidence status. No original source archive, query-model checkpoints or 500K TAC/SSM binaries were introduced by this implementation increment. All eight unit tests use constructed distributions strictly for scoring mathematics and validation. \`REPO_REPLAY=BLOCKED\`, \`INDEPENDENT_TEST=NOT_RUN\`, \`ARCHITECTURE_SUPERIORITY=NOT_ESTABLISHED\`.
