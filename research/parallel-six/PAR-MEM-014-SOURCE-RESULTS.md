# PAR-MEM-014 — first independent public original EPA AQS longitudinal source admission

**27 October 2026 | completed original publisher data acquisition, metadata-only
identity/clock examination, source byte SHA freeze and independent cold replay.
No train/dev/test PM2.5 concentration fields were parsed as numerical model
targets, and no neural training or physical outcomes were scored.**

This study continues genuine TAC source event memory research after
PAR-MEM-013's short-delay gains on repeatedly viewed original Beijing DEV
and prevents falsely calling the older cohort an untouched new TEST.

## Official source provenance

Publisher: US EPA Air Quality System (AQS), 2024 hourly PM2.5 FRM/FEM Mass
parameter **88101**. Official
[pre-generated source catalog](https://aqs.epa.gov/aqsweb/airdata/download_files.html)
and [EPA hourly CSV schema](https://aqs.epa.gov/aqsweb/airdata/FileFormats.html).

Publisher original source URL:
`https://aqs.epa.gov/aqsweb/airdata/hourly_88101_2024.zip`.

**Original publisher ZIP SHA256 (measured during actual GitHub CI source
download, not from web text):**

`d5cae2dc878c6b97c2d80e7583bc6c930562012ff129df9db604262fda465c38`

ZIP size **66,347,589 bytes**; original inside hourly CSV uncompressed
**2,365,678,973 bytes**; **8,139,551 original publisher rows** (matches
EPA publisher listing as of June 25, 2026). Original ZIP file is not
copied into the Git repository because of its approximately 66-MB size;
the exact original SHA and publisher retrieval URI are frozen. If EPA
changes its archived ZIP, the original replay will **fail** rather than
silently accept a different dataset; long-lived byte-level archival
outside the publisher remains an operational source reproducibility task.

[Original exact source identity + monitor split + ZIP member/CRC frozen
receipt](../source-bundles/PAR-MEM-014/EPA2024-METADATA-RECEIPT.json) |
[Original preregistered intake and data visibility protocol](PAR-MEM-014-PROTOCOL.json) |
[Original metadata-only intake code](../../scripts/parmem014_epa_metadata_admission.py) |
[Successful real publisher source download, freeze and independent
replay](https://github.com/FPC-effortless/TAC-OSM/actions/runs/40548213196).

## Real independent monitor streams and source-only split

Each monitor is identified by original source-reported
`State Code-County Code-Site Num-88101-POC`, **not** arbitrary
row index or site name. A SHA256 monitor hash modulo 100 registers
disjoint monitor cohorts (train slots 0–69, DEV slots 70–84, sealed
TEST slots 85–99); UTC source-observed clock windows are disjoint:
January–August 2024 train, September–October 2024 DEV, and
November–December 2024 sealed TEST. A monitor is eligible by
**at least 160 unique source-reported GMT hourly timestamps**
in its originally assigned calendar window, permitting a real
129-event historical-input study if clean-label availability is later
verified. Duplicate original owner+hour timestamps are excluded from
metadata eligibility, never averaged, and hour gaps cannot be filled
by synthetic measured values.

| Registered source partition | Real eligible monitor IDs | Original unique owner-clock observations, metadata only |
|---|---:|---:|
| Train | **706** | **3,781,649** |
| Development | **128** | **180,618** |
| Sealed independent source TEST | **158** | **222,784** |

Across the three selected periods, **no duplicate monitored
owner/hour clock collisions** were observed in metadata. The
source-header parser counted **22,390 rows** with out-of-scope or
noncanonical owner/GMT clock metadata, but did not inspect their
pollution values or select tests using these counts. This count does
not establish invalid physical measurement: reporting in late
December 2024 local time can cross GMT calendar boundaries.

The entire source's 2024 hourly data is NOT the 4.19-million
selected metadata-only owner/time rows; other monitors/hours are
excluded by preregistered partitions. These source-monitored counts
are NOT guaranteed valid concentration-label counts.

## Source quality fields verified WITHOUT scoring physical values

All original rows falling within the three preregistered monitor/time
partitions use EPA physical units **Micrograms/cubic meter (LC)**
and **FEM** method-type metadata. Some original reported rows
carry nonempty qualifiers; the receipt preserves the original
distribution (including empty, code 1/2/3/6, IF, IT, QX etc.)
without assuming their numerics are valid. For any subsequent
training/source label read, the quality policy must be fixed
BEFORE parsing source concentrations: only unqualified, properly
sourced FEM, standardized PM2.5 rows shall be admitted by default,
and owner-method changes need explicit documented treatment.

The metadata-only intake deliberately NEVER indexes the EPA CSV
`Sample Measurement`, `MDL`, `Uncertainty`, `Latitude` or
`Longitude` fields. The CSV bytes necessarily pass through a
streaming reader that sees full text records; **no concentration
numeric value is parsed into a source model, summarized, thresholded,
scored or used for source monitor selection.** Claiming that the CSV
file physically contains no target bytes would be inaccurate.

## Existing TAC-OSM comparisons and scientific boundary

PAR-MEM-013 trained ten original matched-source-shaped TAC old-value
models (~98,326 active weights each) on real Beijing FIT:
associative source memory versus independently trained
**extra-matrix-blind** contextual TAC. Original source held DEV showed
paired old-value MAE improvements by ~4.60% at lag16, ~3.93% at lag64,
and only ~0.36% at lag128; the correct-vs-swapped event value margin
remained very small. These are **previously viewed Beijing cases** and
are NOT new-source generalization.

This EPA stage has **ZERO neural optimizer steps**, no 2024 EPA PM2.5
target scores and no source-action/control consequence data. New-source
performance or cross-source generalization must NOT be claimed.

EPA has only the single measured PM2.5 quantity in this selected
archive, while previously fitted Beijing architecture received 13
physical channels + availability. It is scientifically invalid to
invent the other 12 physical channels as if measured, or to claim
zero-shot qualified transfer with a mismatched encoder. A new
univariate **actual observed source-channel** adapter with explicit
missingness, original GMT clock, source-specific train-only
normalization and matched **actually trained** 100k-parameter
controls is the next implementation gate.

Pre-register a complete one-channel source train objective, original
owner/time/quality filter, 16/64/128 genuine observed-event old-value
pairs, selective write eligibility, and actual next-hour measured
target availability. Keep 158 monitor-ID sealed TEST concentration
records unparsed until optimizer/checkpoints/DEV models frozen and
independent test protocol authorized. Test real source distinct
site/period; compare trained aligned and trained context-only/zero
read, seasonal/recent source priors, genuine stateless, GRU and SSM;
full forward+backward+optimizer FLOP parity and 95–105k live learned
parameters within 1% mandatory. Native CPU runtime not required to
match; all checkpoint, source, case order and cold-replay materials
committed before TEST scoring. PNDS full FP32/AD/probability
refinement remains an open separate proof gate.

**Decision:** original independent sensor source now exists and its
metadata/source identity is frozen; the long-event retrieval hypothesis
is NOT yet validated on this new EPA source. Research remains within
the original six TAC families; no original trained checkpoint, prior
negative experiment, or official source numerical heldout target
has been rewritten or scored.
