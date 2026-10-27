# MODEL01-UKDALE-KETTLE-ADMISSION-001 — publisher-original source results

**Date 2026-10-27; evidence tier `PUBLISHER_METADATA_VERIFIED_ONLY_NOT_POWER_RECORDS`**. This reports authentic *original publisher metadata*, not actual appliance forecast accuracy. UK-DALE, Jack Kelly & William Knottenbelt, 2015 article DOI `10.1038/sdata.2015.7`, 2017 UKERC/CEDA disaggregated DOI `10.5286/UKERC.EDC.000004`, author open metadata repository Git revision `0d168958ea2e69b8bcb0dccc8a732adbfe085985`. License **CC BY 4.0** with publisher attribution.

## Per-house official metadata outcome

| Real household | Author kettle IAM channel | Source label | Decision |
|---|---:|---|---|
| House 2 | 8 | Dedicated kettle-only active power plug | **Future FIT + chronological CAL**, data unopened |
| House 3 | 2 | Dedicated kettle-only active power plug | **Future wholly distinct-house DEV**, data unopened |
| House 5 | 18 | Dedicated kettle-only active power plug | **SEALED future independent TEST**, its power rows unopened |
| House 1 | 10 | Kettle shares IAM with food processor and toasted sandwich maker | EXCLUDE from single-kettle ground truth |
| House 4 | 3 | Kettle shares IAM with radio | EXCLUDE from single-kettle ground truth |

Original individual plug device `EcoManagerTxPlug`, logged power unit W, source preprocessing clamp at 4000W. Publisher author files detail the exact appliance identities; these are **not** equivalent to UCI374 original scalar `Appliances` (whole-appliance category energy Wh) nor to REFIT sum-of-partial-meters, so **not** an independent validation of the old UCI374 trained checkpoint.

## Original Git and CI evidence

- [Actual original-author live metadata Actions run #40506937959](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40506937959) passed **14 constructed software source-integrity tests** and read the author-owned *six* YAML documents at the exact frozen Git commit; **no meter data or protected House 5 power records were read**.
- Source receipt committed in Git at `research/source-evidence/MODEL01-UKDALE-2017-ORIGINAL-AUTHOR-METADATA-001.json` with all original source file SHA256 and official raw URLs. Original author `dataset.yaml` digest `0758f67d5e05d8e6e0ba9c53f89c56f0c48d39b0671cf0e268d6c9439b88f4f4`; House 2 `building2.yaml` digest `5c55daa763e7cc76041c9e1a8964b30b69c73b1d0314781774be3a4b8e523469`, House 3 `building3.yaml` digest `9475d9967f47fd9ac8c32e7d4ec2fe44360d3b884b1e7e649a863297c91da61f`, House 5 `building5.yaml` digest `5ef515f15a1be9d4c0c9771ad07e0e0122c3d7344cd9e2ef3a0dce1c9e192b0a`.
- Source operator: `specialist_models/model01_ukdale_kettle_source_admission.py`. Offline opt-in original House 2/3 single `.dat` parser validates source-supplied SHA, strictly ascending UNIX timestamps, watt units, publisher 4000W clip, observed-logger snapshots with no fabricated fill, and >90 sample near-complete 10-minute buckets. One bucket's output is the arithmetic mean *logged sampled watts*, not provable integrated Wh nor raw physical sensor instant. Any gap/no qualifying buckets/source corruption blocks. A supplied checksum is insufficient for *publisher* source authenticity without an archive/official file manifest digest.
- House 5 data access is prohibited by main CLI and parser separately. Separate one-time final evaluation under a new preregistration will be required to open it after frozen trained model+metric+source-specific as-of contract.

## Identified acquisition shortcut (still unverified for this study)

The official [UKERC UK-DALE 2015 archive](https://data.ceda.ac.uk/edc/d1/b3baa0aa-05ac-4ec8-b608-da5751758698) advertises separate individual `.dat` files, whereas the final 2017 archive is published as large whole ZIP/HDF5 bundles. These are **different publisher releases**; their source hashes and measurement identity/eligibility MUST be checked before considering an older version for the newly frozen 2017 protocol. No 2015 meter data or dataset release was downloaded or used here, and no 2017→2015 mapping equivalence is implied.

## Failure and next steps

**What passed:** original metadata source, categorical source identities, privacy/protected-site open guard, W source-unit handling and fixture gap checks. **What remains blocked:** genuine publisher-authorized 2017 per-channel time series, attributable archive hashes, history completeness / clipping / timestamps, confirmed prediction-time availability, real trained new-task TAC/GRU checkpoints, source-matched practical baseline, and genuinely untouched house TEST evaluation. The real publisher says power in a polled RF plug packet can be several seconds old, and receiver packets may be missing; logged timestamps cannot prove live remote sensor availability. Do not call observed-only numeric power proxies real energy Wh.

Next substantive task is **original lawful data acquisition and source-boundary qualification for House 2 FIT/CAL only**. Do not open House 3 labels to select a model until a full native-vs-GRU new-task protocol is frozen, and NEVER open House 5 now. Retrain both 95k–105k models from original real kettle same source, equal examples, params within 1%, budget + proper zero/persistence/clock-feature baselines, preserve original full Git checkpoints+independent replay. A successful kettle forecast study is **scientific cross-household evidence**, not automatically a commercially useful product (human kettle switching may be intrinsically unpredictable). Customer actions/cost and paid usage remain UNKNOWN.
