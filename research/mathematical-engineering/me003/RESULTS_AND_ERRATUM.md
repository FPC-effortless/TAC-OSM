# ME-003: real-source suitability results and recorded-only index correction

**Original real-source audit:** [GitHub Actions run 39338118254](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39338118254), artifact `12227571362`.
**Independent recorded-only source-ledger correction:** [run 39338381208](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39338381208), artifact `12228036168`.
**Source:** Official UCI MetroPT-3 DOI 10.24432/C5VW3R. ZIP SHA-256 `aab991a970e58210de853bb8078ce0e63abb4d9412fdc5c79792dae3d8e1721a`.

## Original chronological train-era audit
- Opened **only first 30 days**, 2020-02-01 00:00 to 2020-03-01 23:59:55; strict exclusive cutoff 2020-03-02.
- **222,113** authentic recorded rows, **222,076** eligible short-interval transitions (gaps/duplicate-time pairs 36).
- Original parser's numeric column count **16** included a monotonically increasing **blank-named CSV index** that is not a physical sensor. Original original counts are **preserved unchanged**.
- The recorded-only correction discarded blank key `""` from the originally archived JSON, yielding exactly **15 genuine signals** and **12** with >=100 positive *and* >=100 negative observed changes. No additional data read and no neural fit.

| Physical sensor | Positive delta count | Negative delta count |
|---|---:|---:|
| TP2 | 65,637 | 49,562 |
| DV_pressure | 40,864 | 41,014 |
| Motor_current | 57,112 | 57,703 |
| Oil_temperature | 76,550 | 113,922 |
| COMP | 1,224 | 1,220 |
| Towers | 1,659 | 1,654 |

The user-specified real-world-first **source suitability** gate **PASSES**. Real compressor data contain bidirectional transitions, unlike the selected first-24-channel SCANIA directional evaluation (6,379 positive / 0 negative).

**Limits:** Qualifying real observations are not a prediction result. This is a single air compressor, no autonomous semantic entity discovery, no causally identified actions, no observed decision costs, no commercial claim. The UCI description gives conflicting nominal rates (1Hz vs 0.1Hz), so any future model must use audited timestamps and actual observation intervals. No future period beyond the first 30 days was parsed as part of this audit.

**Next preregistered test:** genuine multi-horizon state prediction and observed-change discrimination on a NEW chronological model-selection split, with persistence, simple drift/LightGBM, GRU, and native TAC recurrence controls. Only after this base model meets real gates should add-ons like CDL-PHR, executable memory, local BDH-inspired plasticity, learned assemblies, and oscillatory gating be promoted. Keep all tracks open for experimentation.

**Decision:** Corrected 12/15 real sensor source gate PASS. No architecture promotion, no model training, no merge. Preserve both original and correction evidence.
