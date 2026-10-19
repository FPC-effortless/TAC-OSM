# Unit erratum: published temperatures are Fahrenheit

The frozen protocol incorrectly calls the targets Celsius. The publisher's
measurement table specifies **°F** for all room temperatures and `T_out_Avg`,
and **W** for `Net_Total_Power`. Training and evaluation read the published numbers
without conversion. The fitted model predicts Fahrenheit, and every archived
raw MAE/difference is Fahrenheit. This was a labeling error, not a converted-Celsius
training run. The original protocol, fitting source, weights and all scores remain
unchanged and available.

Relative MAE, TRAIN-scaled MASE and the MASE reference contrast are unchanged by
positive affine temperature conversion. Raw-error differences and bootstrap bounds
convert by multiplication with 5/9, preserving the registered zero-difference gate.
For temperatures themselves use `(F−32)×5/9`; **do not subtract32from an error**.
No numerical threshold, candidate, training seed or test selection changed.

The initial standalone export incorrectly named its raw output `temperature_C`.
That export and its numeric-only audit are preserved as
`pre-unit-correction-export.tar.gz` and `pre-unit-correction-audit.json`; they are
superseded and must not be used as the release interface. The corrected adapter
returns `temperature_F`, declares every input/output unit in metadata and loads
exactly the same checkpoint bytes. Corrected standalone replay is reported in
audit.json. No real equipment was controlled.

Primary source: OEDI909 description PDF, measurement table in section1.3,
SHA256 `fca4ed23265c244434ee40ba45127fe8a9d166a246fc86e1d90b6781521755a4`.
This erratum is a qualification caveat, not a new experiment or a claim that the
failed MetroPT study has passed. Any commercial or independent-site replication
must use a correct, explicit unit contract from registration onward.
