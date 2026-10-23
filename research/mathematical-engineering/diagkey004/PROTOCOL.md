# DIAGKEY004 v1 — one bounded write/read repair

**Owner-authorized synthetic mechanism diagnostic only. Not a real-world model
improvement, architecture-superiority comparison or product qualification.**

The current owner explicitly requests DIAGKEY004 shared key/query normalization,
decoupled suppressive writes and a gated interference curriculum, and asks us to
consider controls and a versioned longer-budget test. This is authorization for
this one narrow diagnostic; it does not repeal the real-world evidence rule.
This registration is committed before any optimizer update. Original DIAGKEY001–003,
real models, scores, protocols and untouched real cohorts are immutable.

## Corrections incorporated before fitting

- The previous native code ALREADY L2-normalizes write keys and final read
  queries. Normalization alone does not align independently learned projections.
- The DIAGKEY002 oracle uses 8 one-hot keys and **8 values**, four active pairs.
  Its chance is 1/8. The neural task draws four DISTINCT keys from16 initial
  keys (64-key vocabulary including distractors), four DISTINCT values from32,
  and one answer-free query. Its chance is 1/32. The oracle proves no arbitrary
  64-key capacity or learned recall advantage. New independent correctness tests
  also cover 32 continuous value codes at four orthogonal addresses.
- Near-zero mean cosine is suggestive, not proof of a unique primary failure.
  Read norm is not value decode accuracy. Gated writes do not prove calibration,
  epistemic confidence or learned relevance; beta is a learned update strength.

## Integrated candidate

Native bounded 88D fast +88D slow recurrent state and one 8x8 value-by-key matrix.
Learned normalized key-ID addresses are SHARED for WRITE and QUERY. They do not
mix the value or event kind. Learned 8D value codes are tied between writing and
32-class metric decoding, with an additional native fast/slow nonlinear readout.
Alpha retention and beta error-write gates are separate learned functions of
observed event/native state, initially .99 and sigmoid(-2). No hardcoded initial
four-step warmup or distractor-write suppression. QUERY is an explicitly read-only
operation: it neither decays nor overwrites the associative store. Fast/slow still
process the query. Matrix projection has radius4, rather than the inherited
radius1, to avoid shrinking four correctly stored unit associations. This is a
bundle of mathematical/training-interface changes, not isolated attribution.

For a unit key k, bounded value v and writable operation:

    D = alpha M
    M' = projection_radius4(D + beta (v - D k) k^T)
    lookup = M' q

For a QUERY, M'=M exactly. Shared addresses guarantee address agreement for the
same observed ID BY CONSTRUCTION, not successful learned value recall. Neither
this guarantee nor normalization proves correctness under interference.

## Fixed fit and selection budget

Five original seeds75001–75005, five arms: aligned_tac101527, original_tac101554,
no-association_tac101560, GRU101500, Transformer101616. All weights active and
within95k–105k; max relative gap0.115%. No padded/frozen module. Identical original
DIAGKEY003 FIT_ZERO stateless synthetic examples/order, batch16, AdamW lr.001,
weight_decay.0001, clip1. **4096 updates** (65536 query exposures/model). This is
an explicit budget amendment motivated by DIAGKEY003's still-improving curves.
Fresh models; no resuming original artifacts, LR sweep, restart or model selection.
Save ORIGINAL model/optimizer/Torch+NumPy+Python RNG at **640 and4096**, plus live
prefit-to-update-to-checkpoint receipt, exact FIT stream digest, process CPU and
learning curves. This supplies the requested identical640-update controls while
also measuring a longer budget on every model, rather than only benefiting TAC.

The signed partial supported-operator forward/backward FLOP profiles are recorded
before fitting. They are **not complete FLOPs**, differ by arm, and do not pass
1% full-compute parity. CPU is report-only. This owner-requested exception admits
mechanism learning/reference measurements with matched parameters/data/updates;
it does NOT admit a fair architectural winner or real-world advantage. No dummy
work or changed data exposure is used to pretend to match compute.

DEV128 fixed new synthetic queries/seed report curves only. Do not tune against
DEV. All25 models and BOTH checkpoint sets must be Git-owned before any fresh
synthetic TEST512/seed is scored. TEST uses a new fixed DIAGKEY004 tag, shared by
arms; seeds are not independent real cases. Preserve all logits, labels, correct
bits, NLL and accuracy per model/milestone. Evaluate both milestones after all
fits freeze; always use4096 as the final gate, no cherry-picked seed/checkpoint.

## Gate and stop

The aligned native model must reach **>=90% exact recall in EVERY seed** at zero
distractors. The threshold is unchanged. If any seed fails after this one amended
run, stop this synthetic tuning series: redesign write/read outside this fit and
return to authentic-data capability development. If all pass, a separately
registered interference curriculum may be considered. No delay training, extra
budget, hidden follow-up sweeps or256-step memory qualification in this study.

## Independent correctness and replay

Compare the cell against the external Flash Linear Attention naive gated-delta
reference (MIT; Songlin Yang, Yu Zhang, Zhiyuan Li; pinned commit
54f725e5dad4abab3211be202b13ead380ef26da), scale1 and transposed state orientation.
Reference excerpt and MIT notice are preserved in tests/references. Reference
checks exclude our radius projection; separate tests cover projection and exact
read-only semantics. This cannot substitute for a genuinely trained positive
control; the fresh Transformer provides that empirical check on the task.

Prior art: Yang, Kautz and Hatamizadeh, **Gated Delta Networks: Improving Mamba2
with Delta Rule**, ICLR2025; official abstract read before registration:
https://proceedings.iclr.cc/paper_files/paper/2025/hash/4904fad153f6434a7bcf04465d4be2cc-Abstract-Conference.html
Reference implementation inspected at:
https://github.com/fla-org/flash-linear-attention/blob/54f725e5dad4abab3211be202b13ead380ef26da/fla/ops/gated_delta_rule/naive.py
Retention and delta correction are established mathematics. New work here is
engineering their interface with TAC's native recurrent state and testing it.

A separate clean Git checkout must restore ALL original checkpoints, verify
source/registration/schedule/runtime evidence and recompute50 original TEST
score sets. Exact labels/correct bits; logits atol/rtol1e-5; NLL abs1e-5.
Repeat replay is idempotent and never overwrites original evidence. Real-world
capability scorecard stays ENGINEERING_VERIFIED, authentic_real_data=false,
independent_gain=false, regardless of synthetic accuracy. No merge/deployment.
