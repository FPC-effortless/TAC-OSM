# ME-052: ACTUAL real-robot heldout training/evaluation RESULT
Evidence checked 2026-10-22. This is NOT a synthetic task, no simulated performance,
and no after-the-fact modifications to the actual trained checkpoints.

## Original real-world experiment

- [Original 10 completed training fits, 5 native TAC / 5 GRU, 80 steps each](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39714369920)
- [Ten genuine frozen trained checkpoints and optimizer/source/seed receipts](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39714369920/artifacts/12398303623)
- [Independent evaluation-only run 39715423987: SUCCESS](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39715423987)
- [Full original per-seed/per-episode heldout evaluation and checksum evidence](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39715423987/artifacts/12399585506)
- 325 ORIGINAL physical FIT episodes; 80/80 DEV and 100/100 TEST original
  physical episodes evaluated. The original empty first DEV Parquet
  was an acquisition issue; retry with the SAME immutable source revision
  recovered the file. **DEV missing=0, TEST missing=0**.
- Original immutable dataset revision
  `9290f5a96dc91ea58256e81d9ce2dd9621890b51`.
- Models use identical original sampled real history/action/feedback rows and
  80 optimizer updates for each seed. Native 37,985 trained parameters and
  established dual-feedback GRU 38,295 (0.81% difference).
- All 10 original actual model checkpoint SHA verified BEFORE any independent
  evaluation. Evaluation optimizer updates: **0**.
- CPU time was never required to match (user correction PR326). Exact total
  operator FLOP equality is NOT independently demonstrated; disclose this
  rather than claiming a stronger compute-matched result than actually proven.

### Genuine untouched TEST physical next-joint-state MAE

MAE on original next observed robot joint positions, LOWER IS BETTER.
Scores use full original episodes with recurrent belief and postprediction
measured feedback, and every 64th actual physical target from original
frame 255 onward. All values are in original physical joint-state units.

| Fixed paired seed | Native TAC MAE | Trained GRU MAE |
|---|---:|---:|
| 52001 | 29.334864 | 2.068242 |
| 52002 | 15.344459 | 2.710149 |
| 52003 | 63.877079 | 2.757953 |
| 52004 | 12.363735 | 2.828587 |
| 52005 | 31.842294 | 2.701072 |
| MEAN | **30.552486** | **2.613201** |
| Last measured state persistence (untrained control) | 0.066949 | 0.066949 |

TAC beats GRU in ZERO of five paired seeds. It also loses dramatically to
an honest last-measured-state persistence reference in every seed.
Across 100 original test episodes, paired cluster-bootstrap 95% CI for
GRU MAE minus TAC MAE is **[-29.952035, -26.046816]** (1,000
original episode resamples, frozen seed 52052). Negative interval
strongly favors the GRU. **Registered TAC capability success FAILS.**
However this does not establish GRU as a commercially valuable world
model: GRU itself loses to the plain physical persistence baseline by ~39×
mean test MAE; native TAC loses to persistence by ~456×.

### Genuine DEV data (not primary claims)

Original DEV 80/80 physical episodes, mean across five pretrained seeds:
native TAC physical MAE 31.769316; trained GRU 2.671611;
last-measured-state persistence 0.069773. Consistent negative pattern.

### Human takeover metric: not reliable evidence of intelligence

In the evaluated t≥255 and 64-stride original frames, **TEST yielded just
2 next-human-takeover positives** across 1,282 eligible original policy
frame positions; DEV yielded just 1 across 1,083 eligible policy
positions. These counts are shared by ALL seeds and models. Takeover
logloss exists but does not establish meaningful detection/generalization,
PR-AUC or safe autonomous operation. No task success, verified cost/reward,
controlled counterfactual intervention effect or commercial decision
performance is available in this publisher corpus.

### What the failed experiment teaches (without inventing a solution)

A strong, immediate observational baseline (persistence) outperforms both
trained world models by an enormous margin. The measured error cannot be
dismissed as lower computational efficiency, model size mismatch, fake
data, or synthetic task artifacts. It indicates a major failure of the
integrated implementation/training regime to preserve near-identity local
real-world physics, plus especially poor/unstable native TAC full-episode
belief predictions; the specific mechanisms remain HYPOTHESES, not yet
identified by this result. Both models predict the absolute 14D next state
instead of starting with the known physical x_t and learning an action-
conditional residual, which is standard for locally smooth dynamical
systems. Do not call this proposed repair a new discovery. Frozen
observation/feedback ablations and per-history/episode-state diagnostics
on actual DEV physical frames are required to distinguish state drift
from output representation. Do not redo another dozen minor gates.
After choosing a mathematically justified full predictive-state
reformulation, train **both architectures** on the same real sources
and budgets and validate on an independent previously unseen physical
task/cohort. TEST from ME052 is now already observed and cannot be reused
to create a new independent confirmatory claim after tuning a response
to these test metrics.

### Scientific decision

This failed actual experiment is IMPORTANT and must remain unmodified.
It directly contradicts claims of native TAC robot-world capability.
Treat recurrent state stability and action-conditioned physically plausible
next-state prediction as the NEXT major capability target, not CPU runtime.
No "commercial qualification", System-2 intelligence or superior
action-policy claim can be made from these results.
