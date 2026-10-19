# Active research goal

Build a reliable learned maintenance model that earns its complexity on real
sensing and verified action outcomes. **ACTIVE; NOT COMMERCIAL-QUALIFIED.**

Acceptance: [active goal](research/commercial/ACTIVE-GOAL.md).
Progress: [goal state](research/commercial/goal-state.json).

[REAL-MAINT-003](research/real-maint-003/RESULTS.md) now passes every local numerical
room-forecast gate. The validation-selected RevIN MLP reduces mean relative MAE
by 32.1% versus persistence; all 20 seeds avoid regression on every room/horizon
cell, with simultaneous bounds and a locked-ridge comparison passing. All 88
neural checkpoints and 8 references replay with zero local drift. Remote checks
for the new final head remain pending.

**Unit erratum:** published temperatures are Fahrenheit, despite the frozen
protocol's incorrect Celsius wording. The corrected portable model explicitly
returns Fahrenheit; weights, predictions and dimensionless gates are unchanged.
See the [erratum](research/real-maint-003/UNIT-ERRATUM.md). This is one building's
point-forecast component, not a repair policy, full analog-sensor qualification,
calibrated health belief or profitable commercial agent.

[DEV067](research/dev067-stable-interface/RESULTS.md) remains NO PROMOTION: all six
arms complete every required world across 20 seeds with zero final faults, but
only 16 TAC and 9 preselected GRU encoders pass both calibration tails. No CASM or
freezing advantage is established; log coordinates increase cost. PR228's ten
remote checks pass at417930c. Prior studies and weights stay intact.

Next work requires independent-site/unit-verified sensing and actual public
inspection/action/price/physical-outcome evidence. Learned state/dynamics and CASM
attribution remain open. [REAL-MAINT-002](research/real-maint-002/RESULTS.md) stays
failed and MetroPT July B unopened. A good forecasting component is progress,
not completion of the broader maintenance goal.
