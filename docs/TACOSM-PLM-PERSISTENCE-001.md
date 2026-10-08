# TACOSM-PLM-PERSISTENCE-001

This experiment isolates whether persistent state is causally necessary.

At t0, a memory bit m is presented. After an optional delay, the current
observation presents an independent bit c. The q2 target is m XOR c.

For every c there are paired histories with identical current text/image/audio
but opposite m and therefore opposite q2 targets. Thus any model that discards
the earlier state has a conditional accuracy ceiling of exactly 0.5.

The delays tested are 0, 1, 4 and 8 steps. The persistent arm retains a learned
residual state from t0. The reset arm discards that state immediately before q2.
A shuffled-history control pairs states with the wrong current observations.

The primary criterion is delay-4 persistent accuracy >= 0.90 in aggregate and
>=0.75 for every seed, with reset <=0.60 and a paired persistent-minus-reset
gap >=0.30 whose bootstrap lower confidence bound is positive.

A positive result establishes bounded causal persistence on this synthetic
task. It does not establish long-horizon real-world memory, multimodal
competence, or scaling.
