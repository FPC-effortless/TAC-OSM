# REAL004 diagnostic forecaster card

Status: **validation rejected; not promoted**. Two exported models are the first
registered recipe0/seed97400 pilots, not the validation-selected recipe or best
seed. They run but have no operational reliability or economic qualification.

Task: estimate reported post-rehabilitation seconds needed to fill a20L bucket
given actually recorded technical fields. Output is seconds per20L, not a repair
success probability, stroke count, liters/second or a maintenance action.

Architectures: two16-wide GELU layers with exponential residual output, initialized
at training median, float64 inference, training-only centering/scaling and explicit
missing/unseen flags. `prestate` takes pre-intervention functionality only;
`repair` adds recorded replacement/chlorination fields. No IDs, dates, coordinates,
photos, text or post-outcome field is an inference feature. Inference needs only
Python/PyTorch; metadata and demos reproduce the exact input transform.

Training is fully numerical supervised factual-outcome learning. It does not learn
causal effects or discover interventions. Data schema, field selection, grouped
split, missing-value representation and positive target definition are supplied
priors. Replacement entries may be incomplete; performed actions must not be
treated as authorized pre-action plans. Predictions for novel hypothetical bundles
are unsupported. There is no stop controller, Bayesian planner or physical repair
rule in these forecasters because they do not select actions at all.

Real source: openwashdata, Mulanje Malawi2019–2020,178training and70validation
targets in this study. Later60assigned rows/39groups remain unopened as targets.
The selected candidate recipe has mean validation MAE6.8898seconds vs6.5788for
the simple locked reference; all three initialization outcomes are preserved.

No real price record, independent collection/follow-up timing, causal action
identification, new-country calibration, long-term durability or factory test is
established. The exports are for diagnosis/reproduction, not repair execution or
commercial claims. Raw survey geographic/personal fields are not redistributed
inside prepared model inputs. Data attribution: Emmanuel Mhango and contributors,
floodchlorinationsurvey0.1.0, CC BY4.0, DOI10.5281/zenodo.15846578. TAC-OSM code
metadata declares Apache-2.0.
