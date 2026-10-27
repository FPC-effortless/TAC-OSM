# MODEL01-REFIT-SOURCE-QUAL-001 — source/target compatibility, no evaluation

2026-10-27. Evidence tier **PUBLISHER_METADATA_ONLY**, no downloaded REFIT appliance time series, no fit, no inference run, no protected source test exposure.

Publisher: University of Strathclyde REFIT, David Murray, Lina Stankovic and collaborators. Cleaned dataset https://zenodo.org/records/5063428, DOI 10.5281/zenodo.5063428. Public publisher confirms CC BY 4.0, original appliance/aggregate recording of 20 households at ~8-second intervals, multi-year. Cite Murray et al. (2017), Scientific Data DOI 10.1038/sdata.2016.122. Source original research paper/cleaning description https://pmc.ncbi.nlm.nih.gov/articles/PMC5315495/ .

**Crucial target mismatch:** UCI374 trained model predicts appliances energy from the original UCI building; its scalar `Appliances` is not automatically the same as REFIT partial submetered monitored appliances, nor the same as REFIT whole-house aggregate. Each house has a limited set of IAM channels, and appliances can be moved or remapped. A fixed selection of IAM channels must be declared using the publisher's appliance metadata and actual matching physical target definition; do not call their sum `all appliances`. The cleaned REFIT publication forward-fills gaps under 2 minutes and inserts 0 W under longer gaps, so cleaned samples are not unambiguously original observations. Never use those synthetic-fill values as independent truth for an as-of study without a native missingness trace.

## Gate for any future evaluation

1. Independently attest version, license, original ZIP/CSV SHA hashes, site/appliance metadata (which devices each channel measures and when), and which sensor samples are originally observed vs cleaning/interpolation.
2. Either obtain genuine **same measured target** at a second site, or preregister a *new* partial-submetered task with a separate model and equally fitted practical controls. Different targets cannot qualify UCI374 original checkpoint as generalizing.
3. Original per-IAM 8-second power readings can produce exactly one 10-minute accumulated appliance Wh only from 75 contiguous, actually observed power samples (`W*8/3600`). Any missing sample blocks that window; do not interpolate for scientific original-source claims.
4. Record UTC mapping and actual `available_at` telemetry. The measured sample time alone is not proof of the earliest accessible time to a deployed prediction system.
5. Freeze train/cal/DEV/independently untouched TEST by *whole household* and time (do not mix the same house into 'new household' tests), original fit and checkpoint hashes, source split exposure ledger and effect threshold before scoring. Compare persistence, seasonal, historical-feature and source-appropriate matched architecture controls. Preserve original model and source replay materials in Git.

## Repository work

`specialist_models/model01_refit_metadata_and_target_gate.py`: fetches at most 750 KB of original *publisher metadata only*, verifies Zenodo record identity/DOI/files, checks license metadata if provided (does not assume it), and defines a strict 75-value W-to-Wh integration with explicit `actually_observed=True`. It has no download of 490 MB source archives and never executes model inference. A failed metadata audit remains BLOCKED, not a model failure. `tests/test_model01_refit_metadata_and_target_gate.py` uses constructed publisher-like fixtures and is not real skill evidence.

## Commercial implication

REFIT makes actual appliance-level household variation testable **after qualified acquisition**; the historical UCI235 negative transfer already showed non-generalization under target shift. No customer permission, pilot, paid adoption, edge or cloud serving success, source freshness guarantee, or TAC-OSM neural advantage is established here.