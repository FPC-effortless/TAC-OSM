# ME-057 protocol

The experiment uses the metadata-reserved `clean_stains_off_tabletop` task at
publisher revision `9290f5a96dc91ea58256e81d9ce2dd9621890b51`.  The frozen
block split is 228 FIT, 76 DEV and 80 TEST episodes.

Five paired seeds train native TAC and the 40-wide GRU for 600 AdamW updates,
batch two, from identical metadata-determined windows.  Each window contains
256 factual history transitions and a 16-step action-conditioned forecast.
Physical error after logged human takeover is censored. TAC's anchor and
physical-feedback losses use 32 transitions already present in the same
history window; they receive no additional episode or future data.
The final factual burn-in transition remains in the gradient graph for both
architectures, so each model's feedback dynamics are trained; the preceding
255 transitions provide no-gradient recurrent context.

Transition kernels execute in PyTorch eager mode. A compiled-kernel attempt
aborted before its first optimizer update, so it exposed no heldout trajectory
and produced no fitted model. Parameter, data, optimizer-update and profiled
FLOP parity remain compulsory. CPU time is recorded independently as an
efficiency result, consistent with the portfolio-aligned master plan.

All ten checkpoints freeze before any DEV or TEST trajectory is downloaded.
The final evaluation uses one deterministic robot-controlled origin per heldout
episode and reports original-unit physical MAE through the first takeover.

Qualification requires mean TEST TAC MAE at least 2% below both the matched
GRU and persistence, plus an episode-cluster bootstrap interval for TAC minus
GRU wholly below zero.  Failure remains a result and blocks promotion.
