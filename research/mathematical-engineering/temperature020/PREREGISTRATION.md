# TEMP020: scalar calibration mathematics and source-budget correction

System1; internal_research; 2026-10-28. Engineering-only now: no authentic
checkpoint/source/model scores available, no neural fits, protected TEST, GPU
or promotion. Preregister before numerical fixtures/source construction audit.

Implement positive scalar inverse-temperature a on frozen logits: p=softmax(a*z).
No per-class biases (they can alter argmax). Use fixed a in[.25,4],64 bisections
on mean calibration CE derivative, compare endpoints/root with identity a=1.
Freeze selected scalar and numerical calibration-data digest. Identity fallback
ensures the implemented calibration-sample loss does not worsen; this says
nothing about a different document or held-out population. Count the selected
scalar as an extra deployed learned calibration value when actually fitted.

Later real-data experiment is BLOCKED until original source/checkpoint intake
passes and exact original logits/labels/document memberships are independently
reproduced by the authorized original pipeline. Before that run append immutable
actual cases/source/license/checkpoint hashes; freeze 3 lexical DEV documents
for calibration and remaining3 for scoring using original6-document protocol,
with no document overlap. All6 DEV are already exposed, so findings are only
exploratory internal calibration, never new TEST evidence. Primary compare
same frozen logits with a=1 vs fitteda; mean NLL nats, exp(mean NLL), top1/top5,
case/document denominators; report per-seed/document, adverse changes and total
training exposure (120*8*128=122880 target positions/model; repetitions allowed).
Do not report raw-byte BPB for the historical lossy >=128→? mapping. It reports
bits/mapped-symbol, unless a new separately registered byte-complete codec is
used. Never tune temperature/bias on scoring labels or access protected TEST.

This increment includes the pure-Python calibrator, negative/gradient fixtures,
source-pinned no-inference constructor/parameter audit, evidence report and a
reciprocal PNDS Lean/Python formulation. Whole-model IEEE/AD OPEN; no proof of
unseen gain/convex-optimizer convergence. Source audit reads exact GRAPH019
commit2aeb631300f1465b5b62e7ffc966f8add8566f81 original base model, verifies bytes
and constructor counts; no change to original model source/checkpoints.

Acceptance: fixtures for scalar-order preservation, bias counterexample,
analytic CE derivatives/Hessian, zero-variance/boundary temperatures, unchanged
input logits, calibration/score document separation, honest mapped metrics;
source budget arithmetic verified against live untrained constructors only;
Lean build/axiom and Python reference gates, draft PRs and Issue472 update.

Prior art: Guo etal ICML2017 temperature scaling
https://proceedings.mlr.press/v70/guo17a.html ; Grave/Joulin/Usunier ICLR2017
continuous neural cache https://arxiv.org/abs/1612.04426 . Temperature scaling
is established calibration, not TAC novelty. Cache mixing already exists in
GRAPH009–016; preserve first-open Python failure rather than duplicate it.
No conv/shifted-write/dynamic-evaluation training before measuring this stage.
