# Real observations and exposure record

New source: UCI849, DOI10.24432/C5B034, *Power Consumption of Tetouan City*,
Abdulwahed Salam and Abdelaaziz El Hibaoui. Publisher metadata and license page,
original ZIP and original CSV are retained. UCI declares CC BY4.0; the license
link was verified on the preserved publisher page. Source URLs/hashes are in
sources.json. This is publisher-provided operational power/weather data, not an
independent audit of its instruments, timestamps or collection latency.

CSV: `Tetuan City power consumption.csv`, 4,222,390 bytes.
SHA-256: `ca8b2f30cf07dd872cb7f2aa45dac110ee38dbe5fc5910a6ff4bcc530dae3a40`.
Actual numeric rows: 52,416; API says 52,417 instances, which differs by one
from the parsed row count. No rows are removed. Cadence is 600 seconds throughout,
from 2017-01-01 00:00:00 to 2017-12-30 23:50:00; timezone unspecified.
No nonfinite/missing numeric values. Original row order and bytes preserved.

Train/validation/test boundaries: [36691, 41933, 52416]. The first 70% fit all
population statistics. Last 20% is the reserved temporal period, opened only after
all selections were frozen at 75d4dce0. All forecast targets stay in their role;
observed history may precede the role. Sixteen hours of observed history at this
cadence, up to 120 hours of direct future outputs. No future weather or calendar
input, fabricated labels, external warm-start representations or simulated data.

All eight numeric columns are inputs and outputs:

- Temperature
- Humidity
- Wind Speed
- general diffuse flows
- diffuse flows
- Zone 1 Power Consumption
- Zone 2  Power Consumption
- Zone 3  Power Consumption

The three power columns are the registered primary application metric; all-channel
scores also include five historical weather measurements. Three zones of one
city do not provide independent multi-city or unseen-site transfer evidence.
Power units are not specified in the UCI variable metadata; no Wh/kW conversion,
energy saving, price, intervention or action utility is inferred. Zone MAE can be
reported in original publisher numeric units by train-std rescaling.

Five legacy sources retain ME036 original bytes and processing. Their existing
exposure, shared ETT equipment/resolutions and Weather missingness/sentinel/cadence
caveats remain. The completed household/KuaiRec tests are not reopened.

A metadata/header-only acquisition of Beijing UCI501 was screened and rejected
for fresh qualification because ME013 already used that source. Its ignored local
bytes under results/me038-source are not used for any fit, score or capability
claim and are not included as ME038 primary training data. All primary source
and model attempts are counted in the final manifests.
