# ME053 — genuine DEV forensics: unstable learned TAC feedback confirmed

**Original evidence**: [fully passed authentic publisher DEV run #39741301288](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39741301288).
[SHA-pinned 12-source-episode diagnostic receipt #12409541819](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39741301288/artifacts/12409541819).
Artifact ZIP SHA256 9fba7b9cd388c3eab1cbc7cf06248d8821e9e03efceec8b310b88b6268eb4fdf.
Exactly original trained ME052 checkpoints, original published immutable revision,
12 DEV episode IDs [77,102,103,121,149,166,167,320,339,378,451,454];
no real TEST frames opened, no model optimization or additional fits.

## Original physical DEV next-state MAE, lower is better

| Original frozen arm and seed | Full original feedback | Physical-only feedback | No learned feedback |
|---|---:|---:|---:|
| TAC 52001 | 29.312104 | 48.548435 | 4.061415 |
| TAC 52003 | 55.668643 | 13.173667 | 6.002467 |
| GRU 52001 | 2.144604 | 2.146940 | 2.364031 |
| GRU 52003 | 2.890832 | 2.885592 | 3.056472 |

These are authentic robot measured x_(t+1) outcomes, never generated data.
No-new-training mechanistic ablations; human takeover suppression varies
only whether the postprediction feedback is assimilated, not the genuine
state/action observations or forecasting labels.

Original native TAC maximum recurrent state norm:
- TAC 52001 full feedback **842.206** versus NO feedback **6.706**.
- TAC 52003 full feedback **1053.761** versus NO feedback **6.393**.
- GRU 52001 full feedback **4.499**, no feedback **3.829**.
- GRU 52003 full feedback **4.453**, no feedback **3.821**.

Actual native TAC full-feedback MAE across real history regimes:
- Seed 52001 frames 256–511: 7.6863; 512–1023: 33.4774;
  ≥1024: 71.0625. No-feedback: 4.3595, 3.5984, 5.1637.
- Seed 52003 frames 256–511: 26.7067; 512–1023: 65.1659;
  ≥1024: 95.3718. No-feedback: 6.2130, 5.8352, 6.1201.

## Causal, constrained interpretation

Within these **original frozen model** ablations, enabling the learned
postprediction feedback causes a massive recurrent-state explosion and
progressive physical forecasting damage for TAC. Suppressing that path
dramatically reduces both state norms and forecasting errors. This is
evidence that **the trained original TAC verification mechanism is a
major proximate source of failure**, not proof that the design can
learn long-history physics with no other changes. Even no-feedback TAC
MAE of 4.06/6.00, and GRU of 2.14/2.89, remain far worse than the
observed original task's typical last-state persistence error (~.067
on the full original TEST).

The original feedback update does not guarantee zero belief change
at zero true innovation, and permits repeated additive increments.
That permits drift mathematically. The measured original trajectories
demonstrate it is an important REAL failure mode, not an invented
synthetic benchmark. The exact blame apportioned between physical
and takeover feedback varies by seed and joint nonlinear dynamics:
TAC seed 52001 physical-only worse than full, TAC seed 52003 full
worse than physical-only, so no simplistic "physical head alone"
claim is scientifically warranted.

## Decision: one meaningful correction to whole world-state math

Next model should use existing known physical system-identification
engineering (not an invention):
1. Identity-anchored action-conditional next-state prediction
   `x_hat_(t+1)=x_t+scale_delta*neural_delta(z_t,a_t)`.
   Freeze original FIT-only physical-delta scales before fitting.
   Both native and established GRU comparator get equal access to
   this baseline and the same supervised delta targets.
2. Innovation-dependent bounded recurrent verification with
   `alpha_t→0` when factual outcome residual `r_t→0`,
   `z_new = (1-alpha_t) z_old + alpha_t tanh(proposal)`.
   This replaces unbounded additive drift, preserving the native
   fast/slow relevance/associative architecture rather than replacing
   it with an SSM.
3. More substantial actual matched learning exposure than the ME052
   80×2×8 = **1,280 gradient-bearing factual next-state targets**,
   while preserving compulsory equal parameter counts, training
   data/order, optimizer budget and evaluation conditions.
4. ME052 TEST already inspected; future corrections are hypothesis-
   generating on this exact physical task. A NEW independently
   untouched original robot task/session source is needed to claim
   fresh corroborated generalization, not reuse the old test and
   pretend it was prospective.

Do not run a dozen tiny retention gates; train one integrated
physically stable native candidate and a full known recurrent rival.
Report zero improvements as carefully as any true original gains.
