# DEV-025: frozen-checkpoint cross-evaluation

Adaptive diagnostic registered after the DEV-024 audit. Known diagonals: DEV-023
masked-local H512 short98.54%/long93.75%; DEV-024 joint99.17%/74.58%.
Initialization, training cases and evaluation cases changed together. Before
redesign or retraining, exchange the held-out evaluation grids for these frozen
models. No off-diagonal outcomes have been opened when this protocol is frozen.

Twenty bound checkpoints (two cohorts, five seeds, two regimes) each run on both
H512 grids:40 cells. Seeds43/59/71/89/107. Evaluation offsets23000/24000,
four repetitions, all roles/deadlines/delays and both partners:288 rows/cell,
96 primary rows. No optimizer, training or component changes. The executable
protocol lives in development_protocols because the training-contract schema
requires positive training steps, whereas this experiment has exactly zero.

Require exact diagonal reproduction against archived raw rows and a separate
process replay of both grids. Input report/checkpoint/parameter hashes are
registered in research/autoresearch/dev025-inputs.json. Save those original
reports and checkpoints with every output. The stdlib auditor binds the inputs,
regenerates both grids, validates every prediction/physical outcome/cost, checks
all40 cells and paired contrasts. Its public-case validator is frozen DEV-024.

Primary diagnostic: DEV-024 minus DEV-023 frozen-model accuracy on each common
grid. Secondary: new minus old evaluation grid within each cohort, and their
interaction. Use matched-seed exploratory95% t(df4) intervals; a meaningful
separation needs magnitude>=5pp with interval excluding zero. Report all means,
per-seed failures, reset/swap and physical costs. Keep accuracy>=.8/swap drop>=.1
and unchanged failure/unnecessary/cost gates. Contrasts are diagnostic, not a
retrospective rescue of DEV-024 attribution. If worse models remain worse on
both grids, investigate trained-model variation; if scores track the grid,
investigate case sensitivity. Mixed results remain mixed.

This distinguishes model-cohort variation from evaluation-sample variation. It
cannot isolate initialization from training data, exposure or optimization.
Original diagonal results were already known; this is adaptive, synthetic,
fixed-eight-entity research. Preserve both prior studies and their limits.
