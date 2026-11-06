# Mamba-2 reference provenance

`src/tac_osm/decision_sys1_ssd_reference_003.py` adapts the stable segment-sum
and chunk-factorized SSD reference from Albert Gu and Tri Dao (2024),
state-spaces/mamba commit `6b72c122713bb769cc82c6b8e6d019c53d27d6a1`,
`mamba_ssm/modules/ssd_minimal.py` (Git blob
`9632ebd4350fa18ddc977c2bdedb0bab1fd82646`). Apache License 2.0 is included.
Changes: removed CUDA imports/test; replaced einops rearrangements with torch
reshape/permute; added explicit shape/dtype errors and a batch-state interface.

The Mamba-2 block reference uses equations and parameter geometry from
`mamba_ssm/modules/mamba2.py` at that commit (Git blob
`36b16d4715ed2d753697167d65da5e6abba22017`), with restricted one-group,
full-SSM, no-projection-bias geometry and out-of-place state for differentiation.
The gated RMS equation is checked against that source's `rms_norm_ref`.
FP64 reference arithmetic preserves FP64 rather than emulating Triton FP32.
This is not a claim that official kernels were installed or executed.
