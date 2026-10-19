# ME-001B recorded-only independent replay and sign-prior diagnostic (PASS)

**Audit GitHub Actions run:** https://github.com/FPC-effortless/TAC-OSM/actions/runs/39336856338
**Archived audit artifact:** `12226529392` (`me001b-independent-recorded-replay`).
**Original model-training run:** https://github.com/FPC-effortless/TAC-OSM/actions/runs/39334730465, immutable nine-checkpoint artifact `12225908024`.

### Evidence integrity
- Authentic source: official SCANIA Component X v3 publisher **TRAIN only**, uncompressed CSV SHA-256 `e01cb0bd87dfab4c9dbad215d51d81282fc0d413be96d6f819ea872fb7a3c715`.
- Original source container ZIP SHA-256: `6fdf63d15805f4ff429e6cd3172c4bd1634ca5c6d293406a79553c56c07db483`.
- Independent replay container ZIP SHA-256: `b0463ced99ef3c08fc35685e1467d45763b08dbc58f0079227dec412f597f645`.
- The source **member** was byte-identical on both downloads; ZIP container bytes differed; the cause of the archive discrepancy has not been established.
- All 9 original model checkpoint SHA-256 checks passed. Scores, source split, TRAIN-only change thresholds, persistence, and midpoint state-reset metrics reproduced within the audit contract. **Exactly 0 model refits**.
- Publisher VALIDATION/TEST and any untouched SEQ reserved tests remained unopened in this work.

### Newly identified label degeneracy
On the original **6,379 selected heldout sensor changes**, defined using TRAIN-only absolute-change thresholds:
- **6,379 positive changes** (100%).
- **0 negative changes**.
- A constant `predict increase` sign classifier achieves **100% accuracy** without sequential modeling.
- A per-feature dominant-sign classifier trained only from original TRAIN histories also reaches **100%** on this heldout subset.
- The original `persistence change_direction_accuracy=0.0` is a zero-residual sign convention, not a proper comparison with a constant-positive sign classifier.

Thus, the recorded GRU ~98.7% and native TAC ~83.0% directional scores **do not demonstrate change-direction discrimination or reasoning**. Signed-log magnitude MAE remains an authentic recorded result, but the task's dynamics are structurally limited. Do not claim direction generalization.

**Scientific decision:** Original ME-001B native TAC promotion FAILED: mean macro MAE 0.042881 vs persistence 0.041950 and global GRU 0.041281. Audit PASSED. Existing original model checkpoints and failure evidence preserved.

**Next research:** Audit new authenticated multivariate operational signals for both positive and negative changes and genuine events; do not choose a new architecture by repeated tuning on this SCANIA heldout cohort. MetroPT-3 and Backblaze have different roles: non-monotone signal evolution and logged drive failures respectively. No counterfactual repair utility claim until authentic action-effect data justify it.
