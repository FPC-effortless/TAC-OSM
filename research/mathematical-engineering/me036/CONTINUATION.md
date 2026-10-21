# ME036B — prospective training continuation, test still closed

Registered after ME036 validation learning curves, before any final test scoring.
All280 original100-update fits remain intact. Their late learning curves motivate
training the SAME integrated models longer, rather than another mechanism study.
No architecture or hyperparameter trial is selected from test outcomes.

Continue EVERY successful model from its u100 checkpoint including Adam/RNG,
through total updates250,500,1000, batch4 and identical sampler schedule per seed.
Cosine learning rate from.001 at update101 toward.0001 at1000. Same wd/clip.
500 additional process-CPU-second ceiling per fit; failures/partial checkpoints
retained, no replacements. Keep all six endpoints25/50/100/250/500/1000 eligible
for identical lowest-validation-MSE selection. Same sparse validation anchors,
full stride1 test evaluation, two seeds, five datasets and four horizons.
Freeze all continued selections and hashes before opening final test metrics.
Original100-update model test results are not scored separately; final score
may select an original endpoint. Original training + continued training + all
validation/checkpoint/evaluation costs must be separately accounted.

Add20 unmatched practical shared-channel ridge references (one per dataset/H):
all training anchors stride24; normalized96-value input plus intercept;
full-H locally normalized targets; alpha100, intercept unpenalized, float64
normal-equation accumulation/solve. No extra or future data. Save float64
coefficients; deploy float32 coefficients and score that exact deployed model.
No initial float64/float32 forecast equivalence claim. This is not a capacity,
exposure or compute matched architecture arm and receives no validation tuning.
All ridge fits finish and coefficients freeze before final test scoring.

ME036 success gate and all original integrity rules unchanged. Actual total
spend parity is not established. No architectural-superiority/paper-parity claim.
