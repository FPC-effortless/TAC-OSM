# Input-to-output execution graph

```mermaid
flowchart TD
  X[Recorded station sensors, masks, clocks and station ID] --> V[Causal validation and masked FIT normalization]
  V --> E[Shared encoder for every event]
  E --> A[Observed-only learned admission]
  A --> S[Parallel diagonal sum/mass scan]
  E --> B[Parallel evidence basis and prefix sum]
  S --> C[Exclusive source context]
  B --> C
  C --> G[Fast/slow scheduling maps]
  E --> G
  G --> M[Ordered affine associative-memory scan]
  M --> R[Final causal query and read]
  G --> R
  R --> D[Shared bounded state refinement]
  E --> D
  S --> P[Observed or retained source anchor]
  V --> P
  P --> F[Six coherent censored-lognormal forecasts]
  D --> F
  F --> Q[Median, intervals and proper likelihood]
```

The two prefix computations can run alongside each other, then schedule the matrix scan. Their order is a dependency, not a claimed single universal scan. The last query and decoder run once per requested forecast, reusing the encoded observations and final states. Streaming uses the same equations with explicit owner/clock/version state. No fixed wall-clock speedup is inferred from this graph.
