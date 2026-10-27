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

## New publisher-original raw README discovery and stronger qualification block — 2026-10-27

Publisher-original RAW dataset: University of Strathclyde [REFIT Electrical Load Measurements](https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements/), **distinct from cleaned Zenodo**. Original RAW dataset DOI **10.15129/31da3ece-f902-4e95-a093-e0a9536983c4**; public portal lists CC BY 4.0, a ~666MB REFIT_RAW_081116.7z archive, a small original README, and physical-appliance metadata. The source can be suitable for *a new appliance-domain task*, but **raw does not mean every 8-second value is a fresh physical sample**.

[Actual original raw publisher README audited in CI #40485327371](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40485327371): **6 fixture/format tests PASS**, real publisher README SHA256 `2bdcb4be06c45df67eb25ab11cb603482b5eb8820ecb18aa616d47921bd30e9c`. Publisher explicitly states readings were recorded on changes of electrical load, intermediate values filled, sensors sampled asynchronously under polling, update may lag actual physical sensing, and appliance ownership/channel mapping could change. Original README thus **does NOT establish a per-sample actually-observed mask, a timestamp-at-observation, unchanged monitored appliance identity or live data available-at timestamp**.

Do **not** set `actually_observed=True` in `checked_wh` just because a row appears at a nominal 8-second interval in REFIT CSV. That unit helper proves arithmetic only if independently sourced native event/sensor-available receipts establish every sample's origin; otherwise it is constructed-input diagnostics, not original real-world verified sampling. Even original RAW 7z must be independently inspected for native change-event source bits, IAM identity history, DST/wall-clock correction and publisher data rights. Publisher READMEs and cleaned metadata have distinct source lineages; never merge their hashes as if interchangeable.

Complete permanent source-only original README digest and warnings are tracked in `research/source-evidence/MODEL01-REFIT-ORIGINAL-RAW-README-OBSERVABILITY.json`. The massive raw or cleaned datasets have not been acquired, no REFIT forecast/evaluation has been trained, and original UCI374 `Appliances` versus a subset of REFIT IAM targets remains **NOT EQUIVALENT**.

**Next registered gate:** qualify an actual source event-level or independent observation-origin history, choose one objectively identified monitored appliance and 10-min Wh accumulator, and freeze appliance/household target and splits before opening outcome rows. Stop at `SOURCE_OBSERVABILITY_BLOCKED` if native observed/filled provenance is unavailable; do not manufacture flags to unblock metrics. Real customer-data rights and historical `available_at` receipt remain unresolved.
