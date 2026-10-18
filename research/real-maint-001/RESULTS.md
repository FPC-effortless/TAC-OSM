# REAL-MAINT-001 — external operational sensor forecasts

**Real data evaluated; commercial repair-policy validation remains unavailable.**

Fitted 80 fresh models and 8 validation-only pilot models. All twenty training seeds retained per arm. CPU fitting/preparation: 4.81 minutes.

| Arm | Mean normalized MSE | Mean Gaussian NLL |
|---|---:|---:|
| tac | 191.830632 | 2532.107525 |
| no_casm | 191.963655 | 2624.269002 |
| gru | 192.110837 | 869.555987 |
| mlp | 208.482555 | 37263.864897 |

Persistence normalized MSE: 7.464566. Training-mean MSE: 192.569026.

## Registered gates

- no_worse_than_ridge: **False**
- better_than_persistence: **False**
- better_than_GRU_MLP: **False**
- physical_repair_validation: **NOT_AVAILABLE**
- commercial_cost_validation: **NOT_AVAILABLE**

One real compressor and four reported fault windows do not provide twenty independent fleets or repair trials. Outside reported fault windows, health is unknown. A May maintenance-date inconsistency is preserved. Models train only on next observed sensors; no simulator teacher or fault labels are used for fitting. Gaussian marginal assumptions include binary sensors.

The exported seed92701 was selected before outcomes. The archive includes a PyTorch-only runtime with train-only normalization and next-sensor mean/std output. It does not issue maintenance actions or certify repair success. No cost saving is inferred from unmeasured inspection prices or counterfactual interventions.

See [registration](PROTOCOL.md), [complete statistics](statistics.json), [strict replay and CPU measurements](audit.json), [source download hash](source-download.json), and [complete evidence index](evidence-index.json). Source: [MetroPT-3, UCI](https://archive.ics.uci.edu/dataset/791/metropt+3+dataset), DOI10.24432/C5VW3R, CC BY4.0.
