# Raw-F room-temperature forecaster

Working research component: a64-unit MLP with RevIN and a learned residual above
the last reading. It predicts all ten rooms at 30/60 minutes from eight past
15-minute records. There is no maintenance-action script, Bayesian filter, simulator
or CASM program in this model. All training targets are measured future readings.

**Units:** all eleven temperature inputs and outputs are **°F**; the final power
input is **W**. The frozen protocol's Celsius wording is wrong. The corrected
runtime and metadata use `temperature_F`; the original label error and unchanged
weight/prediction proof are documented in UNIT-ERRATUM.md. Do not use the preserved
pre-correction export. For a temperature in°C use `(F−32)×5/9`; for MAE use only 5/9.

Input order: room 102,103,104,105,106,202,203,204,205,206 temperatures, outside
temperature, Net_Total_Power. Each JSON line supplies `values` (8×12) and
`minute_of_day` (8 corresponding logger times). Metadata names every channel/unit.
Output names the channels, horizons[30,60] and two point-forecast vectors. It is
not a health probability, confidence interval, fault diagnosis or repair authority.

Candidate and LR were chosen from validation before20 fresh initializations. On
935 held-out windows from 11 physical2019 experiment days, all 20 candidate models
avoid point-MAE regression in every cell; mean relative MAE is.6790, versus 1 for
persistence. Simultaneous20-cell and locked-ridge comparative gates pass. Adjacent
days in one building do not establish independent-site reliability. Linear RevIN
fails some cells; no general architectural or normalization superiority is claimed.

Exports use the predeclared FIRST seed 97301 for all four arms. All80 fresh fits,
eight pilots and eight references remain in evidence. No best-seed selection,
test-tuned fallback or intermediate-checkpoint choice. Each final model is EMA,
float64 at inference. The released candidate has8532 learned parameters. Warm
batch1 forward CPU median is.0705ms on the recorded host; tensor storage is68KB,
not the memory footprint of the Torch process or a production latency guarantee.

Runtime imports only its bundled model and Torch plus standard-library modules.
No TAC-OSM training harness, Veritas or NumPy is needed by the exported code.
CPU PyTorch is required. Example requests and expected outputs are bundled; their
public measurements retain source attribution. The4 corrected exports replay
exactly locally; verify remote final-head status separately.

Intended use: offline/read-only thermal forecasting and independently evaluated
research pilots using the exact unit/schema contract. Unproven: new buildings,
different sensor/channel identities, other sampling periods/horizons, calibrated
uncertainty, missing-input handling, actual repair-policy utility and ROI. This
component does not qualify the broader TAC-OSM commercial goal or its memory/CASM
novelty claims. Prior negative compressor studies and reserved July B stay intact.

Source: Kim, Janghyun; Leach, Matt; Im, Piljae, Curated Test Fault Data Set, NREL2019,
DOI10.25984/1862682, CC BY4.0. Project metadata declares Apache-2.0; data attribution
and licensing are separate. See RESULTS.md, PROTOCOL.md and UNIT-ERRATUM.md.
