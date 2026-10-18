# Public maintenance sources inspected on2026-10-18

| Source | Verified content | Suitable use and material gap |
|---|---|---|
| [MetroPT-3, UCI](https://archive.ics.uci.edu/dataset/791/metropt+3+dataset), CC BY4.0 | Downloaded complete archive, verified CRC/SHA256; operational compressor sensors and four company-reported failure intervals | Selected for external sensor forecasting/anomaly diagnostics. No measured inspection prices, alternative interventions or independently verified repair outcomes. A published maintenance date is inconsistent; preserve it. |
| [FMUCD, Mendeley](https://data.mendeley.com/datasets/cb8d2nsjss/1), CC BY4.0 dataset | Inspected primary API metadata and a64KiB range containing38-column CSV schema: work-order dates, labor/material/total costs | Real cost/work-order research source. No explicit repair-success, inspection-outcome or independent verification column. Work-order closure is not proof of physical repair. Full dataset SHA is publisher metadata, not claimed independently verified from a header sample. |
| [SCANIA Component X, Researchdata](https://researchdata.se/en/catalogue/dataset/2024-34), CC BY4.0 | Primary catalogue and paper describe real truck operational histories and repaired/censored records; latest release includes test labels | Strong later survival/risk benchmark. Endpoint downloads timed out during this audit; no fake imported dataset or completed evaluation claimed. Censoring is not healthy status; replacements are not randomized causal repair-success trials. |
| [NGAFID, author paper](https://arxiv.org/abs/2210.07317) and [repository](https://github.com/hyang0129/NGAFIDDATASET) | Real flight/maintenance records with before/after-maintenance annotations; full archive multipleGB | Candidate future external sequence benchmark. Not downloaded or licensed for use on inference from paper licence alone; no actual cost or causal repair claim. |
| [Open Repair Alliance](https://openrepair.org/open-data/downloads/), CC BY-SA4.0 | Primary page describes actual community repair-attempt records | Candidate recorded repair-outcome benchmark, not industrial sensor trajectories. No download completed, no inspection-cost/counterfactual evidence inferred. |

Use each source within its actual information contract. Never join unrelated
machines across datasets into fictitious action/outcome/cost trajectories.
REAL-MAINT-001 is an observational external test, not a commercial maintenance
trial. Research code/weights keep their own licence; source data attribution and
licence are recorded independently.
