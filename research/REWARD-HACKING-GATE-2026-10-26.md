# TAC-OSM Reward-Hacking Investigation and Prevention Protocol
Effective: 2026-10-26. **Fail-closed no-fit engineering infrastructure.**

## Objective and scope

Reward hacking / specification gaming occurs when a model optimizes the
training proxy, evaluation harness, or reporting machinery without improving
the intended external task. It may occur in RL, supervised multitask fitting,
simulated robotics, enterprise decision systems, human feedback, or scientific
agent research. It is not synonymous with literal illegal label leakage.

This contract applies to **all TAC-OSM variants and every new version**.
It is **model-architecture agnostic**: scorers, evaluator evidence and
read-only/prediction-time boundaries are outside the model's memory layout.

The research system must be able to answer:
1. **What did the model optimize?** Record the exact hash of the reward
   implementation, optimizer objective, reward model, discount factors,
   exclusions, aggregation rule and prior benchmark.
2. **What did the real world do?** Verify independently sourced outcomes,
   original episode ordering, intervention and sensing costs, real failures,
   and heterogeneous long-horizon consequences.
3. **Did scoring change?** Independently re-run the reward scorer and the
   external goal evaluator against source-pinned facts, not model-reported
   success claims.
4. **Did the model access the reward or evaluator?** Change ONLY postdecision
   reward fields on a cloned authorized episode. Query an identical model,
   fixed RNG and checkpoint twice. A changed prediction => FAIL.
5. **Could the actor rewrite its judge?** Treat training agent writable
   paths and evaluator/reward/holdout/report assets as independent trust
   domains. Check path permissions and hash actual protected bytes against
   out-of-band pinned SHA256s; DO NOT grant model/optimizer write access to
   test definitions, reward implementations, adjudication, sealed data,
   runtime safety enforcement, or evidence logs.

## Attack surfaces, controls and CI falsifiers

| Mechanism | Diagnostic | Prevention layer |
|---|---|---|
| Reward-channel manipulation / reporting forged rewards | \`audit_reward_recomputation\` | Independently recompute reward from sealed facts, block publication if mismatched |
| Falsified real-world goal / self-evaluation | \`audit_goal_recomputation\` | Use separate external goal adjudicator after the decision |
| Manipulation of reward model, harness, scoring script, heldout outcomes | \`audit_protected_assets\` | Verify actual bytes against separately controlled hash pins, restrict write domains |
| Historical outcome or reward slipped into decision time | \`probe_reward_noninterference\` plus existing research anti-cheat | Change reward-only fields; fail or investigate if predictions change |
| Proxy reward increases but true outcome does not | \`audit_proxy_goal_alignment\` | Predeclare minimum improvement and fail on one-sided optimization |
| Easy strata hiding hard-horizon degradation | Per-group, per-episode and horizon matched episode IDs | Required groups / minimum samples / individual and subgroup-specific harm triggers |
| Safety incidents or excessive action/sensing costs obscured by success | Goal + safety + decision-cost paired receipts | Count newly introduced safety incidents without offset from improved episodes; bound worst individual and aggregate decision costs |
| Silent diagnostic disablement | \`RewardIntegrityGate\` via \`reward_audit_ledger\` | Missing, failed or inconclusive required checks block score promotion |

All trust boundaries are enforced outside the reward-optimizing model. A
matching hash proves content equality to a *pin*, not who set that pin or
whether it represents an honest source. The trustworthy evaluator must be
**operationally isolated**; passing the path overlap check is not sufficient
to establish real operating-system permissions.

## Mandatory reward promotion policy

For every new reward-driven or proxy-based TAC-OSM trained model, comparator
or autonomous decision evaluation:

- Preregister separate **reward** and **real-world goal** definitions, their
  higher/lower directions, the cohort and evaluation split, required groups,
  chosen horizons, actual intervention/sensing costs, tolerable safety
  violations, goal regressions and min reward/goal gains.
- Freeze the policy source digest *outside the model-controlled workspace*.
  The comparison uses exact shared source/cohort/goals/reward/evaluator
  contracts. Missing or changed sample IDs, horizons or required groups must
  remain INCONCLUSIVE, not improve a score by dropping hard examples.
- Authenticate the publisher/source chain and real outcome availability
  separately from receipts. Self-reported "external_source" does **NOT**
  independently authenticate a future outcome or causal effect.
- Run all five required checks in \`src/tac_osm/research_reward_integrity.py\`
  with actual source-authorized data and callbacks, then call
  \`reward_audit_ledger(...).assert_clean()\` **before presenting candidate
  scores as qualified improvement or making an optimization/deployment
  selection**. If it raises, freeze that promotion and keep the diagnostic.
- Maintain experiment-specific formal loss/source graph and independent
  scoring notes. When the model has an unknown reward pathway, cannot
  independently reproduce the goal, or can edit a protected scoring
  component, investigate rather than marking PASS.
- On suspicion: preserve source snapshots and hashes; compare the exact
  source-based outcomes against an external verifier; replicate with a
  sealed independent evaluation; report success with and without suspected
  exploits; keep negative findings and rejected checkpoints. Do not tune on
  heldout TEST to evade these checks.

\`PASS\` means only the **implemented checks** found no contradiction on the
specified dataset. It is NOT "reward hacking prevented everywhere", proof of
causal improvement, or a claim that the evaluator is uncompromised.

## Independent runner code

    from tac_osm.research_reward_integrity import (
        audit_protected_assets, audit_reward_recomputation,
        audit_goal_recomputation, probe_reward_noninterference,
        audit_proxy_goal_alignment, reward_audit_ledger,
    )

    findings = [
        audit_protected_assets(
            actor_write_paths=verified_runner_write_acl,
            protected_sha256=publisher_or_auditor_pinned_assets,
            read_artifact=trusted_readonly_reader),
        audit_reward_recomputation(
            actual_reward_receipts, trusted_reward_scorer),
        audit_goal_recomputation(
            actual_goal_receipts, independently_trusted_goal_judge),
        probe_reward_noninterference(
            cloned_fixed_checkpoint_predictor,
            clean_decision_record, reward_only_mutated_record),
        audit_proxy_goal_alignment(
            original_baseline_receipt, new_model_receipt,
            separately_pinned_preregistered_policy),
    ]
    reward_audit_ledger("SOURCE-PINNED-RUN", findings).assert_clean()

The example identifiers are integration placeholders; never fabricate these
receipts, callbacks, checkpoint weights or published results.

### Paired receipt schema

The two files must contain \`schema: tac-osm-reward-audit/v1\`, different
\`run_id\`, equal \`source_sha256\`, \`cohort_sha256\`,
\`reward_scorer_sha256\`, \`goal_evaluator_sha256\`, \`goal_contract_sha256\`,
plus \`outcome_authority: external_source\` and matched \`rows\`.
Rows contain hashed IDs, group, horizon, reward, goal, decision_cost and
safety_violation. These values are descriptive; if baseline and candidate
chose different actions on recorded histories, a paired cohort does NOT
identify the outcome under the unexecuted action. Only actually verified
outcomes (or separately qualified off-policy estimators) can support
real-world benefits.

The preregistered policy has exactly:
\`min_proxy_gain\`, \`min_goal_gain\`, \`max_goal_regression\`,
\`max_added_safety_violations\`, \`max_added_decision_cost\`,
\`goal_direction\`, \`min_group_samples\`, \`required_groups\`.

Partial receipt review CLI:

    python scripts/run_reward_integrity.py --baseline BASELINE_JSON \
      --candidate CANDIDATE_JSON --policy PREREG_JSON \
      --expected-baseline-sha256 EXTERNAL_HASH \
      --expected-candidate-sha256 EXTERNAL_HASH \
      --expected-policy-sha256 EXTERNAL_HASH

CLI intentionally exits nonzero for *all* real receipts: they are missing the
privileged **live** audits. It cannot self-promote an experiment from
self-reported JSON or user-chosen hashes, even when numerically clean.

Fixture-only falsifier:

    python scripts/run_reward_integrity.py --fixture

CI runs this as part of \`.github/workflows/research-anticheat.yml\` on all
PRs and mainline pushes, and saves JSON/JUnit in the normal 14-day audit
artifact. Synthetic / constructed records are solely engineering tests,
**not qualified real-world evidence**.

## Research status, limitations, escalation

Implemented and unit-testable: source hash / write-domain guard,
reward & goal replay, fixed-prediction reward perturbation, matched subgroup
reward-vs-goal/cost/safety contrast, fail-closed five-check ledger, CI controls.

**OPEN:** adaptation of every historical/future variant's *actual* trained
reward function, genuine source-authenticated goal adjudicator, hardened
out-of-band ACL/log storage, hermetic independently recomputed scoreboard,
closed-loop stress testing, long-horizon delayed consequence validation,
offline-policy evaluation with propensities and overlap checks, real-robot
action safety, and independent red-team adversarial reward-design review.
No real trained checkpoint passes this policy simply because the CI
falsifiers pass.

## Research references

- OpenAI (2026), *A shared playbook for trustworthy third-party evaluations*:
  https://openai.com/index/trustworthy-third-party-evaluations-foundations/
- Huang et al. (2026), *Reward Hacking Challenges Oversight of Autonomous
  Research Agents*: https://arxiv.org/abs/2609.28614
- Thaman (2026), *Reward Hacking Benchmark: Measuring Exploits in LLM Agents
  with Tool Use*: https://proceedings.mlr.press/v306/thaman26a.html
