# Additional final-metric verification contract

Registered while training is in progress, before selection freeze or any ME038
held-out metric scoring. No changes to prediction/training code, fits, selection,
thresholds or gate. Saved validation-probe replay is distinct from test-metric
recalculation; the report must distinguish them.

After first final scoring and frozen source/weights, independently recompute ALL
Tetouan selected-model and persistence test metrics, using NumPy float64 error
aggregation instead of Torch float64 reductions. Preserve batch16 and float32
subtraction so that differing batch-dependent float32 matrix kernels are not
silently attributed to the accumulator. Same real test rows, all horizons and
origins, every channel. This second numerical verification cannot trigger
reselections, architecture/optimization changes or a new predictive claim.

Accumulator tolerance is declared now, without observing a discrepancy:
`4*gamma_(n+8)*max(1,abs(metric))`, where n is the number of scalar forecast
errors per channel, unit roundoff u=2^-53, gamma_n=nu/(1-nu). Both positive-sum
reductions/normalization are covered by the factor4. This is a float64 reduction
bound, not the old ME035B float32/ridge criterion; that failed check stays intact.
No tolerance changes after measurement. Publish maximum differences, the declared
bound and failures. This is aggregate final-metric verification, not bit-exact
replay of all final forecast arrays or independent data collection.

Also record supported Torch profiler forward/backward FLOP subtotal on actual
ETTh1 TRAIN windows at H96 for each arm. List operator coverage limitations;
linear/bmm counts omit unsupported nonlinearities, optimizer, preprocessing,
validation and parent training. It is not a complete FLOP or energy parity proof.
