# What the two real-data tests establish

REAL001's retained normalized MSE is191.8306316. Oil_level, DV_pressure and
Caudal_impulses contribute96.65% of its normalized squared error. Two are digital
channels. Five of seven original analog channels have lower raw MAE than
persistence, but DV_pressure is4.80x and Oil_temperature1.57x worse. This is neither
a clean architecture verdict nor a normalization-only excuse. Separate measurement
weighting, type-appropriate objectives, distribution shift and actual tracking error.
The retrospective source/metric code and all twenty fits per arm remain auditable.

The new2022 experiment selects MLP+RevIN using validation only. On June A, seven
channels have model/persistence MAE ratios0.994–0.998. Flowmeter is constant in A:
persistence error0, trained forecast error about3.1e-8. The ratio is undefined,
not epsilon-floored into a favorable number. The strict no-worse gate still fails;
Oil_temperature's simultaneous upper difference also exceeds zero. No practical5%
improvement is established. All validation arms have0/20 all-channel passes.
These are the fixed registered outcomes, not grounds to loosen the gate or choose
an arm from A. July B remains unopened and was excluded from the published payload.

The new sensor forecasts use future public measurements as targets, not simulator
fault labels or posterior teachers. That alone is not discovery of hidden physics
or an action-conditioned real repair model: this source has no action/inspection
choice/cost/independent post-repair outcome contract. It is one APU recording,
not twenty independent machines. No commercial or foundation-model claim follows.

The belief-only prototype is separately UNTRAINED. Its12-state distribution also
represents dependency structure, preventing an information-mismatched health-only
controller. Property tests establish a real bottleneck and declared temporal
symmetries; fresh training, causal interventions and H1024 reliability remain open.

Subsequent work: the original prototype status above is historical. The separately
registered [DEV063 trained study](../dev063-belief-integrity/RESULTS.md) now has
140fresh fits and14pilots. Its failed calibration gates do not qualify the real
forecast study or authorize opening July B.
