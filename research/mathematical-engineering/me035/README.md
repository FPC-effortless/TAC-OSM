# ME035 — Native forecasting architecture development

[Results](RESULTS.md) · [Architecture mathematics](MATHEMATICS.md) ·
[First registration](PROTOCOL.md) · [Backbone refinement](../me035b/PROTOCOL.md) ·
[Original models and offline replay](ARTIFACT.md)

15new neural fits and1ridge fit produce a working selected native forecast
model: development MSE0.369106, versus0.497194for the native reference and
0.370001for ridge. The result is selected development evidence; the final
model uses more capacity and warm-start training. One strict initial
float32/float64equivalence receipt fails and remains explicitly recorded.

Use the3selected raw-input TorchScript models for source-unit forecasts, without
Python package imports. See ARTIFACT.md for paths, field order, shape and loading.
No merge or model promotion occurred. KuaiRec's closed research recipe stays closed.
