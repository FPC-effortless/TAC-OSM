# TACOSM-PLM-JOINT-SCALING-005

This is the first explicit fusion of the two selective-computation surfaces:
persistent-state addressing and computational/operator addressing.

P1 measured memory admission as M increased. P3 measured operator admission
and indexed-vs-scan cost as E increased. P5 measures the interaction.

## Question

Can the PLM loop retain access to both the required memory record and the
required computation while M and E grow independently, under a single
explicit end-to-end budget?

## Loop

state address -> relevance rank -> operator address -> exact execution ->
outcome

The execution is constant-cost and exact so the controlled variable is the
addressing budget. The reranker is an exact agreement control, not a learned
capability claim.

## Grid

M = 64, 256, 1024, 4096, 16384

E = 16, 64, 256, 1024, 4096, 16384

Seeds = 0..4

State K = 16
Operator K = 8
Query noise = 0.10

## Cost

State address = probes + raw candidates + admitted candidates

Operator address = probes + raw candidates + admitted candidates

Execution = 1 exact operation

Full scan control = M + E

## Promotion constraint

A cost reduction is not a capability result unless both target rank tests pass.
The exact reranker isolates the address budget and does not establish learned
semantic routing.

No asymptotic law, semantic-language claim, named-SSM equivalence or
Transformer replacement is inferred from this phase.
