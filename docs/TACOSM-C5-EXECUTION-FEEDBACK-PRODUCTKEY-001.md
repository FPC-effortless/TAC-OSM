# TACOSM-C5-EXECUTION-FEEDBACK-PRODUCTKEY-001

## Objective

Bridge the new verifier-driven representation learning signal into the existing
factorized product-key routing boundary.

## Design

The candidate population is fixed within each M level. Sixteen target state
codes are held out from a 48-code training population used only for product-key
codebook fitting. Queries are noisy versions of held-out target codes.

The dense control trains on execution outcomes. The verifier arm adds a
post-execution repair positive and the rejected selected candidate as a hard
negative. The product-key arm uses the same verifier-trained representation,
but routes through the existing three-factor factor-size-16 beam-7 index with
a maximum shortlist of 32.

The index is refreshed every 32 learner updates. Query-time factor scoring and
state reranking are measured separately from index-build work.

## Required measurements

For each M and seed:
- held-out top-1 retrieval;
- mean target rank;
- proposal admission recall;
- conditional selection given admission;
- candidates scored;
- states_scored_over_M;
- factor scoring MACs;
- pair generation operations;
- amortized index-build MACs.

## Interpretation

A capability gain from verifier feedback is a routing result. A bounded
shortlist is a sparsity result. A sublinear C5 claim requires routing
computation itself to grow slower than M; selecting only a small fraction is
not sufficient.

The phase is intentionally separated from the earlier C5 teacher experiments:
the dense teacher remains a historical diagnostic, not the new learning target.
