# ME017 — Original frozen real-history mathematical diagnosis (2026-10-20)

**Original successful** [GitHub Actions run 39427184346](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39427184346) (first attempt 39426757288 was cancelled by PR synchronization and is NOT an extra result). Original results artifact [12267525406](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39427184346/artifacts/12267525406). Source/checkpoint hashes checked before parsing real FIT. Exact forward parity for 12 original 600-step frozen models; 384 real early FIT histories, no fit/optimizer steps and no development or new heldout user1/2/3 outcomes examined. All 135050 trainables each. The archived original full gradient curves remain the audit source of truth.

## Measured per-seed results (positive values under knockout indicate worse real FIT Brier)

| Native model | Seed | oldest/newest neural history input logit-gradient ratio | measured mean fast gate | measured mean slow gate | zero-slow Brier change | zero-fast Brier change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| unchanged engineered TAC | 14001 | 2.531110 | 0.13475 | 0.04605 | +0.00089662 | +0.0002453 |
| unchanged engineered TAC | 14002 | 3.274811 | 0.13186 | 0.04508 | +0.00064074 | +0.0003089 |
| unchanged engineered TAC | 14003 | 4.008499 | 0.13633 | 0.04759 | +0.00194669 | -0.0000083 |
| negative ME016 identity TAC | 14001 | 2.516404 | 0.13349 | 0.04728 | +0.00157511 | +0.0000385 |
| negative ME016 identity TAC | 14002 | 3.473163 | 0.13171 | 0.04403 | +0.00090569 | +0.0002629 |
| negative ME016 identity TAC | 14003 | 3.389980 | 0.13787 | 0.04742 | +0.00215197 | +0.0000222 |

| GRU control | Seed | oldest/newest neural history input logit-gradient ratio |
| --- | ---: | ---: |
| anchored GRU | 14001 | 0.088144 |
| anchored GRU | 14002 | 0.280141 |
| anchored GRU | 14003 | 0.00001099 |
| original unanchored GRU | 14001 | 0.059949 |
| original unanchored GRU | 14002 | 0.065831 |
| original unanchored GRU | 14003 | 0.074960 |

The original code computes **neural-only** history-input gradients by treating the embedding-plus-metadata sequence as a separate autograd leaf; it deliberately excludes the analytic user prior32 logit shortcut from this derivative. These input derivative ratios and internal knockout Brier measurements are on real historical **FIT**, not independent scientific performance evidence. Grads on model internal states do not bound the recurrent transition's singular values or prove longer-horizon memory. In particular, the ratio >1 could reflect nonlinear readout properties and input scaling as well as useful sensitivity.

## Formal decision on preregistered gates

- Rule A (unchanged native 32-step oldest/newest gradient ratio <0.10 in >=2/3 seeds): **NOT MET**, all three >2.5. Evidence does NOT justify the proposed slow-memory rescue as an answer to an observed 32-step vanishing-gradient problem.
- Rule B (slow zeroed FIT Brier delta <=0 in >=2/3 native seeds): **NOT MET**, all three positive. Original slow state meaningfully contributes to current predicted logit in this in-sample attribution diagnostic.
- Rule C: identity path did not earn real-user heldout ME016F promotion, regardless of seemingly favorable knockout effects or gradients. **REJECT the ME016 identity path as default**.
- Rule D: no fresh heldout/out-of-sample advancement, no 128+/512+ event retention guarantee, no physical counterfactual or causal utility, no fully compute-matched GRU superiority.

## Correct next decision

**Do not train beta=.25 convex slow proposal as a presumed memory fix** on the strength of ME017; that would be an under-motivated architectural addition. The useful minimal direction is to maintain the successful original native recurrence and reduce its measured computational overhead, then study real longer-horizon capability on previously unexposed data and explicit equal parameter, data and training compute constraints.

Because all 3 original native seeds show nontrivial slow-state contribution, a deeper probe of real long-history retention should use genuinely longer histories rather than re-litigating 32-step boundedness. No real future cohort or fresh dataset was opened in ME017. The original result artifact, both positive and negative controls, and all ME001–ME016 studies remain retained.
