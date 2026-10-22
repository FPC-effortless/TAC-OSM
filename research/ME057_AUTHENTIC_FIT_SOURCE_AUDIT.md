# ME-057 authentic original FIT audit — verified source evidence

**October 22, 2026.** GitHub Actions run [39836744099](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39836744099) completed successfully on commit `d2d0a935df3c184f9dd3525cd967faff0c44da91`.

Archived artifact: `me057-authentic-FIT-qualification-receipt`, ID `12451242831`; GitHub ZIP digest `sha256:15ee894beef1dc2c86a54e01d299bb0cd8240be6bb40f4cebd5db82f6a9110e5`. The artifact was downloaded and its JSON receipt independently parsed. The reported canonical hash of its original per-episode evidence list was independently recomputed and matched.

## Source-backed findings

| Registered measure | Verified result |
|---|---:|
| Official dataset | BodenAI/RW-RL-HIL-Dataset |
| Immutable revision | `9290f5a96dc91ea58256e81d9ce2dd9621890b51` |
| Independent task | `clean_stains_off_tabletop` |
| Exact authentic FIT episodes audited | **228 / 228** |
| Authentic original FIT physical frames | **329,306** |
| Robot-controlled potential sample origins | **210,192** |
| Recorded FIT human takeover onsets | **375** |
| Original DEV/TEST trajectory files opened | **0** |
| Model optimizer training updates | **0** |

Canonical per-episode FIT physical source manifest SHA256:
`c4064651e91cced7aafc11337c32a27266f470ea10388bb0dd2562ae955f88fd`.

The first audited original episode, ID 12, had SHA256 `6fdfd71bd635a97319831fbe7b7e6583260e52113f209ef6a37955014e94b6cb`, independently agreeing with the immutable ME-055/056 profiler receipt. Each FIT episode was checked for expected length, actual 14-dimensional state/action, finite values and genuine boolean control-authority flags.

Dataset publisher license [CC BY 4.0](https://huggingface.co/datasets/BodenAI/RW-RL-HIL-Dataset). Published metadata groups four adjacent episodes when allocating cohorts, but independent operators/sessions cannot be verified.

## Scientific boundary

This is **source qualification only**. It unlocks the possibility of freezing a genuine matched training registration but is **not itself authorization to train** under the current ME-057 diagnostic manifest and is **not** a model capability result.

The previously discovered dense auxiliary target leak must be corrected (pre-feedback snapshots) and the updated causal computation freshly profiled. Scientific head-to-head claims require matched full training budget/compute, data/order, parameters, selection privileges, immutable test cohort, source-authenticated checkpoints, and strong persistence reference; no claim is qualified yet.
