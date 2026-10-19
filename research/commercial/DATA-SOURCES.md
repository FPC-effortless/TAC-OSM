# Public maintenance sources inspected on2026-10-18

| Source | Verified content | Suitable use and material gap |
|---|---|---|
| [MetroPT-3, UCI](https://archive.ics.uci.edu/dataset/791/metropt+3+dataset), CC BY4.0 | Downloaded complete archive, verified CRC/SHA256; operational compressor sensors and four company-reported failure intervals | Selected for external sensor forecasting/anomaly diagnostics. No measured inspection prices, alternative interventions or independently verified repair outcomes. A published maintenance date is inconsistent; preserve it. |
| [FMUCD, Mendeley](https://data.mendeley.com/datasets/cb8d2nsjss/1), CC BY4.0 dataset | Inspected primary API metadata and a64KiB range containing38-column CSV schema: work-order dates, labor/material/total costs | Real cost/work-order research source. No explicit repair-success, inspection-outcome or independent verification column. Work-order closure is not proof of physical repair. Full dataset SHA is publisher metadata, not claimed independently verified from a header sample. |
| [SCANIA Component X, Researchdata](https://researchdata.se/en/catalogue/dataset/2024-34), CC BY4.0 | Primary catalogue and paper describe real truck operational histories and repaired/censored records; latest release includes test labels | Strong later survival/risk benchmark. Endpoint downloads timed out during this audit; no fake imported dataset or completed evaluation claimed. Censoring is not healthy status; replacements are not randomized causal repair-success trials. |
| [NGAFID, author paper](https://arxiv.org/abs/2210.07317) and [repository](https://github.com/hyang0129/NGAFIDDATASET) | Real flight/maintenance records with before/after-maintenance annotations; full archive multipleGB | Candidate future external sequence benchmark. Not downloaded or licensed for use on inference from paper licence alone; no actual cost or causal repair claim. |
| [Open Repair Alliance](https://openrepair.org/open-data/downloads/), CC BY-SA4.0 | Downloaded complete80,391,622-byte electronics CSV from pinned primary Git LFS object; SHA256 checked; actual repair-status reports, schema and licence inspected | Source audit in `openrepair-source-audit.json`. Community repair outcomes, not industrial sensor trajectories or independent physical follow-up. `problem` includes solutions and leaks labels. No intervention/inspection costs or causal alternative-action data. No model fitted. |

Additional primary source acquired for coverage on2026-10-18: **FHWA LTPP**
([catalogue](https://catalog.data.gov/dataset/long-term-pavement-performance-ltpp),
[table export](https://infopave.fhwa.dot.gov/Data/TableExport/),
[official FAQ](https://infopave.fhwa.dot.gov/Page/Type/faq)). The normal public
preview API returned638MNT_COST rows,5067MNT_IMP rows and156193longitudinal-condition
rows.360cost records report a positive total cost;334have nonmissing measured MRI
both before and after treatment, across238sections. Four duplicate selected-field profile
projections are reported; no rows silently removed. Five profile records have statusD,
the rest statusE. The [official user guide](https://infopave.fhwa.dot.gov/InfoPave_Repository/files/FHWA-HRT-21-038.pdf)
(pages11–12) warns that E is not inherently more reliable than non-E. Page44
warns that highway agencies compute maintenance costs differently; no error-free
quality or uniformly comparable economics is inferred.

`ltpp-source-audit.json` preserves every source page's SHA256, coverage counts and
matched treatment identifiers/dates. `scripts/acquire_ltpp_maintenance.py` reproduces
read-only acquisition with no credentials, SQL, fitting or outcome-based selection.
The catalogue links the US government-works policy; reuse terms and attribution
must be checked for the exact tables before redistributing raw data or commercial
packaging. Raw source pages currently remain in the execution workspace.

This is a stronger **maintenance** data candidate than unrelated-cost joins:
actual dated work and reported costs link by state/section to physical roughness
measurements. It is road pavement, not factory machinery. Measurement dates can
be far from treatment; MRI is not binary repair success. Costs are historical
reported dollars, missing costs remain missing, inspection prices are absent,
and observational before/after measurements do not identify causal treatment
benefit. No fitted benchmark, sealed performance score or commercial qualification
is claimed from this source audit.

Use each source within its actual information contract. Never join unrelated
machines across datasets into fictitious action/outcome/cost trajectories.
REAL-MAINT-001 is an observational external test, not a commercial maintenance
trial. Research code/weights keep their own licence; source data attribution and
licence are recorded independently.

Follow-up eligibility audit (2026-10-19): `ltpp-followup-audit.json` adds2,386
rehabilitation records, hashes every additional public page and checks the original
source hashes. Of334positive **component-total** cost events with before/after
roughness,95have other recorded maintenance and56have recorded rehabilitation
between nearest measurements (sets overlap).197have neither recorded category;
only4also have both measurements within90days. Zero have both within30days.
These are posthoc date/coverage counts, not preregistered sample selection or an
effect estimate. Missing interventions, same-day treatment bundles and inspection
prices remain unresolved; absence of a record is not absence of a treatment.

Additional candidates screened on2026-10-19:

- [Nepal medical-equipment maintenance study](https://link.springer.com/article/10.1186/s12913-022-08392-6):
  public cost calculations exist, but the publisher explicitly excludes raw
  maintenance job cards from the downloadable data. Several comparison scenarios
  use modeled assumptions. Not an accessible action/outcome training cohort; no
  author contact or restricted data requested.
- [Medical equipment maintenance, Hugging Face](https://huggingface.co/datasets/electricsheepafrica/medical-equipment-maintenance):
  the dataset card describes30,000 **simulated** observations. Excluded from real
  repair validation despite its convenient cost/outcome fields.

No newly qualified factory inspection/action/cost/independent-outcome cohort has
been found in this screening. The real-outcome gate remains unmet; no negative
result is relabeled as commercial evidence.


Additional primary-source screening during DEV067 (2026-10-19):

- [EPA MSOD](https://19january2021snapshot.epa.gov/moves/mobile-source-observation-database-msod_.html)
  documents repair descriptions, links from earlier repairs to subsequent test
  results, parts costs, repair minutes and emissions-component inspection minutes.
  The costs/times are PROCURE fields and can aggregate several repairs; they are
  not automatically prices for a single REPAIR record. The official page offers
  database copies by contacting EPA. Only its public guide was acquired, not
  populated tables. No request was sent and no cost/outcome coverage is verified.
  `msod-source-screening.json` preserves the guide hash and access limitation.
- [ORNL FRP measured HVAC faults, OEDI909](https://data.openei.org/submissions/909),
  DOI10.25984/1862682, CC BY4.0: downloaded and SHA256/CRC-checked the3.77MB
  experiment archive plus1.81MB description. These are measured physical
  fault-injection experiments, not the separate EnergyPlus simulation dataset.
  `ornl-source-inventory.json` and `scripts/audit_ornl_source_inventory.py`
  preserve all member names, sizes and first-line schemas, without parsing sensor
  values or fitting a model. The publisher describes49scenarios; the current zip
  contains63equipment CSVs and5weather CSVs. Reconcile cohort membership before a
  registered benchmark. Fault identity/intensity in filenames must never become
  input features. Weather first lines are logger metadata. No actual repair or
  inspection prices, independent field-repair follow-up or commercial qualification
  are established. This can support an external physical-sensor experiment, not
  replace the missing operational action/cost/outcome cohort.

The source search remains open. Public source documentation, an injected fault
and a measured baseline are not evidence of profitable real repairs.


[Malawi flood-response handpump rehabilitation](https://openwashdata.github.io/floodchlorinationsurvey/),
DOI10.5281/zenodo.15846578, CC BY4.0, was acquired from publisher commit
`a2755213c841fdfc7b65aaaff0b91016296f2985` during this source search.
`handpump-source-coverage.json` hashes the CSV, dictionary, licence and package
metadata. It has328survey records,223normalized waterpoint names (82names recur),
reported pre-intervention functionality and56replacement-related columns.
325records have a recorded20L bucket-fill time,327have post-intervention
functionality and327have strokes-to-water. These are **presence counts only**;
no outcome distribution, predictive score or model has been inspected/fitted.
No cost/price column exists. Missing replacement entries remain missing, not
zero; report dates are not verified physical measurement dates. Collector
independence, follow-up timing, action completeness, ambiguous/repeated names
and possible post-treatment leakage require further source audit before fitting.
Names, coordinates, linked photographs and postrepair fields must not become
shortcuts for a pre-action model; no linked photos were requested. Real component
replacements plus measured flow are useful factual-outcome candidates, but do
not identify counterfactual repair benefit, actual economics or factory deployment.

## 2026-10-19 research-priority change: operational fleet first

The standing [real-world evidence policy](../REAL-WORLD-EVIDENCE-RULES.md)
now governs all new experiments and model-promotion claims. **SCANIA Component X
is the next primary real-world prediction benchmark**; its source was already
catalogued in the 2026-10-18 audit, so this is a priority decision, NOT a
newly discovered or downloaded raw dataset. Current publisher version 3:
https://researchdata.se/en/catalogue/dataset/2024-34/3 ;
DOI https://doi.org/10.5878/bnh5-ka77 ; CC BY 4.0 ; 11 files / 1.54 GiB.
The version-3 publisher provides test labels. Keep them unopened for
selection, verify actual acquisition plus file hashes before claiming
import, and protect the official vehicle-disjoint splits. Recorded replacement
is not a randomized intervention or independent post-repair physical success.
The expert asymmetric cost matrix is a challenge proxy, not measured invoices.
Use Backblaze as an independent forward-calendar-time operational failure task;
retain LTPP / handpump real action-outcome research as separate incomplete
commercial-utility gates. Freeze synthetic-only pilots as historical diagnostics.
