# ME-055 pre-training result

Status: **architecture implemented; comparative training blocked**.

ME-055 reserves an independent authentic robot task and implements a new
multi-step TAC world-model candidate. It does not report model accuracy or an
intelligence gain. The compulsory resource preflight failed before the first
comparative optimizer fit.

## Candidate architecture

The candidate combines:

- a FIT-derived bounded servo prior shared with the GRU control;
- a 32-dimensional fast state and 32-dimensional slow state;
- an 8-dimensional relevance-conditioned associative trace;
- synchronous memory-write and slow-state updates;
- one bounded cross-modal physical/takeover correction;
- observation-conditioned episode initialization;
- 16-step open-loop physical and takeover-onset forecasts.

The design removes the unstable additive feedback implicated by ME-053. Every
physical residual is bounded, and factual supervision stops after human
intervention. These are implemented properties, not evidence of learned robot
capability.

## Independent real source

The selected source task is `clean_stains_off_tabletop` from the official
`BodenAI/RW-RL-HIL-Dataset` at immutable revision
`9290f5a96dc91ea58256e81d9ce2dd9621890b51`. The task was selected by a
predeclared minimum-SHA256 rule before reading episode outcomes. Metadata fixes
228 FIT, 76 DEV, and 80 TEST episodes. No DEV or TEST trajectory was opened.

One FIT episode was subsequently opened solely for the registered authentic
resource preflight. Its SHA-256 is
`6fdfd71bd635a97319831fbe7b7e6583260e52113f209ef6a37955014e94b6cb`.

## Compulsory preflight

Pinned one-core AVX2 CPU, batch 2, 256 observed frames, 16 open-loop frames,
full backward pass and AdamW update:

| Architecture | Parameters | Profiled FLOPs/update | Median CPU/update |
|---|---:|---:|---:|
| ME-055 TAC | 38,266 | 42,724,495 | 0.039856 s |
| GRU world control | 38,295 | 42,546,619 | 0.034604 s |

- parameter gap: **0.076%** — pass;
- profiled FLOP gap: **0.416%** — pass;
- measured steady-state CPU gap: **13.175%** — **fail** against the 5% limit;
- comparative model fits started: **0**.

Compilation cost also differed materially (5.90 CPU seconds for TAC and 0.54
for GRU) and is reported separately rather than hidden in steady-state timing.

The correct outcome is a blocked experiment. The independent source remains
reserved and the candidate remains untrained. Training may proceed only after
a frozen implementation passes the same strict gate; the limit will not be
relaxed after observing this result.

## Evidence

- `research/me055-independent-robot-task-source.json`
- `research/me055-prefit-profile.json`
- `scripts/me055_reserve_independent_robot_task.py`
- `scripts/me055_profile_prefit.py`
- `src/tac_osm/real_robot_me055_rollout.py`

