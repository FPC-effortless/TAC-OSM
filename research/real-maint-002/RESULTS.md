# REAL-MAINT-002 — persistence-start external sensor qualification

**NO_PROMOTION; B remains sealed**. 80 fresh fits plus 8 validation-only pilots; 86.71 fitting/preprocessing CPU seconds. Sealed B remains unopened.

Primary arm selected from validation: **mlp_revin**. No arm was selected from A outcomes.

| Arm | Validation seed passes | Mean relative MAE on A | All-channel simultaneous gate |
|---|---:|---:|---|
| linear_static | 0/20 | None | False |
| linear_revin | 0/20 | None | False |
| mlp_static | 0/20 | None | False |
| mlp_revin | 0/20 | None | False |

| Primary channel | Raw MAE | Persistence MAE | Relative MAE | Simultaneous upper difference |
|---|---:|---:|---:|---:|
| TP2 | 1.05666 | 1.0585 | 0.9982668764078944 | -0.00134105 |
| TP3 | 0.505549 | 0.508549 | 0.9941003361063772 | -0.0019599 |
| H1 | 1.17648 | 1.18357 | 0.9940160050895348 | -0.00408471 |
| DV_pressure | 0.00545641 | 0.00547082 | 0.997366375241648 | -9.71845e-06 |
| Reservoirs | 0.505242 | 0.508241 | 0.9940989266872751 | -0.00195844 |
| Oil_temperature | 0.970633 | 0.973843 | 0.9967037691466188 | 0.00349526 |
| Flowmeter | 3.06723e-08 | 0 | None | 9.61748e-08 |
| Motor_current | 1.74057 | 1.74647 | 0.996623047228741 | -0.00482762 |

MASE uses TRAIN persistence error as scale; full per-channel MASE, undefined scales, every seed and raw errors remain in the [statistics](statistics.json) and indexed evidence. Ratios here divide by A persistence MAE for descriptive comparison; this denominator is never used for training. The simultaneous gate uses 2000 paired day-block bootstrap resamples with Bonferroni8.

Data: [MetroPT2, Zenodo](https://zenodo.org/records/7766691), CC BY4.0, creators Veloso/Gama/Ribeiro/Pereira. This 2022 recording is distinct from the opened 2020 MetroPT-3 study. A is June15–30; B is July onward and stays unopened. Source metadata dates are public; numerical B evaluation remains prohibited by this registration.

No advanced memory comparison, commercial repair policy, cost saving, foundation-model or universal hardware claim follows. Only eight predeclared analog channels are evaluated; digital/GPS were excluded before fitting. Original REAL001 failures remain in the [retrospective diagnostic](retrospective-001.json).
