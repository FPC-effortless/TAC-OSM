# ME018 — Exact native recurrence execution equivalence and CPU budget audit

**Scientific goal:** keep the successful ME015 native TAC fast/slow recurrence untouched while identifying execution overhead that disadvantages native TAC against PyTorch's optimized GRU. This follows the completed ME017 frozen genuine FIT diagnostic: unchanged TAC neural input-gradient oldest/newest ratios 2.53/3.27/4.01 and slow-state knockout raises FIT Brier in 3/3 seeds. The conjecture that 32-step slow memory is near-inactive is not supported; do not add ME016 inside-tanh identity or ME017 convex beta proposal as a default.

## Hypothesis and falsifier

Compiled scripted execution of the **exact same native model arithmetic** could reduce CPU overhead. Primary engineering criterion: equal logits and backpropagated parameter gradients from pretrained original three TAC models; reject implementation upon any finite/parity failure. Performance criterion is descriptive steady-state median CPU forward AND forward+backward speed-up versus *an identical eager kernel* with the exact same weights/operations. Speed-up >1 is favorable, <=1 is negative. No metric threshold tuned after timing.

No new layers, architectural conversions, random physics worlds, estimated commercial savings, real policy-efficacy claims or user future data openings. Model weights and learned architecture must remain bitwise unmodified. The measured numerical contracts use tight floating-point tolerances rather than bitwise equality, because reduction order may differ.

## Original weight/data provenance

- Immutable ME016 successful original 600-update checkpoint run 39412071932, source SHA e12982eb59a2b820855888b2df0f74d2553abb40, original results.json SHA256 6870e254b65b74e8a55a242a3528aa478bf7bd08c6812ce92bf32a87fd4ba416.
- Exactly **three** original native_chrono_anchor pretrained research models seeds 14001/14002/14003, original 135050 active trainable parameters. Check original 12-model ledger and each of the three selected checkpoint SHA256 hashes BEFORE authentic source access. Compare the legacy PyTorch model, refactored identical eager native arithmetic, and scripted identical arithmetic. No learned new parameters or optimizer updates.
- Real publisher KuaiRec ZIP MD5 261550d472c48eff4990fb13c0e5bcf7 and SHA256 c23f211cf109c7a060a25a4afe9e8c681b1ccbbc816cca921d87ad37dd11c70c.
- Only original user%16==0 original FIT anchors before 2020-08-01. Select original 768 evenly spaced genuine FIT windows and use first 128 for fixed timing/gradient parity. Source reader ignores later reactions before parsing target. Previously used DEV/future and heldout users 1/2/3 not scored.

## Mathematical contract

For given x,z,s:

    proposal_fast = tanh(enc(x) + transition(z))
    gate_fast = sigmoid(gate([x,z]))
    z_new = (1-gate_fast)*z + gate_fast*proposal_fast

    proposal_slow = tanh(slow([z_new,s]))
    gate_slow = sigmoid(slow_gate([z_new,s]))
    s_new = (1-gate_slow)*s + gate_slow*proposal_slow

    logits = head([z_final,s_final,embedding(current_action),duration])
             + logit((1+sum(prior32 real binary outcomes))/34)

All exact original parameters transferred to the eager and scripted arithmetic modules. No state detaches. Equality test requires legacy/model equivalence and same trainable parameter names, counts and per-parameter training gradient equivalence after one real-data BCE backward. Artificial random tensors are preflight-only and cannot constitute performance evidence.

## Timing and fairness

Use a single CPU PyTorch thread, same genuine 128-example tensor batch, same device and dtypes, warmed inference, seven outer repeats of eight forward calls each, report median and raw times; five repeats of three identical forward/backward calls without optimizer, to isolate gradient execution cost. Exclude one-time JIT compile time from **steady-state** results but declare this exclusion and do not extrapolate to one-off inference or cold-start. Legacy original eager includes its own per-call source-shape checks; core-eager versus scripted comparison holds validation placement constant. Record three paired frozen seeds. Timings are environment-specific engineering observations and cannot establish broader performance across GPUs or production distributions.

**Architecture-comparison parity rule:** exact original active trainables, same real input/data, same batch/step/seeds and measured compute are compulsory for any later model-vs-GRU claim. This test uses the SAME TAC weights under two execution runtimes, so it is **not** a new architecture comparison and does not claim GRU superiority or a matched GPU/commercial cost.

## Decision

- If the scripted recurrence achieves numerical parity and lower measured CPU time, retain as an OPTIONAL mathematically equivalent implementation candidate and test real production hardware later.
- If it is slower, keep the original eager model; do not modify the predictive recurrence solely for speed.
- Any independent long-memory or real-world predictive improvement still requires genuinely longer real histories and fresh frozen evaluation, not this FIT-only speed test.

All previous ME001–ME017 outcomes, master mathematical plan, and real-world-first policy remain authoritative. This is an unmerged draft engineering experiment, not a scientific or production promotion.
