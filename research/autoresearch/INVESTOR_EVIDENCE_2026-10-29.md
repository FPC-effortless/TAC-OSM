# TAC-OSM stage and investor evidence — 29 October 2026

Assessment, not a valuation or a claim that investors will fund the project.
The supplied results and original repository records support a reproducible
small-model architecture research prototype. They do not yet establish a
superior language architecture, persistent learned retrieval, device-level
quality/cost advantage, product-market fit or safety certification.

## Position relative to Transformer/SSM development

The original [Transformer paper](https://arxiv.org/abs/1706.03762) reported quality
and parallelism/training benefits on established translation tasks. The original
[Mamba paper](https://arxiv.org/abs/2312.00752) reported hardware-aware sequence
scaling and model comparisons at materially larger scales. These are the kinds
of milestones TAC must establish in its chosen niche; component combinations
or parameter counts alone are not equivalent evidence. This comparison is a
stage assessment, not an estimate of how many years TAC is behind.

TAC has passed implementation, genuine small-scale fitting, reproducible frozen
checkpoint replay and informative ablations. A convincing capability/quality-
cost result, independent replication/generalization and a usable narrow product
remain unshown here. Text likelihood does not demonstrate instruction following,
planning, causal intervention skill or general intelligence. CPython-only short
training cannot establish broad LLM scalability.

## Claim ledger

| Claim | Current support | What is missing |
|---|---|---|
| Genuine trainable byte predictor | Five-seed source-identified original fits/replay in035–040 | Larger task/data/seed coverage |
| Better budget allocation | 040 graph-free48D best mean within this exposed DEV study | Individual factorial CIs include0; decoder allocation confounds effects |
| Useful old-history retrieval | Not demonstrated by041 | More adequately sampled genuine history-dependent tasks/labels and fresh confirmation |
| Small weight tensors | 499623 FP32 values =1998492 bytes, about1.91MiB, arithmetic only | Full model/runtime/cache/working-set measurements at useful quality |
| Constant-memory streaming | Not established; windows reset | Implemented carried-state API, restart/ownership and long-history validation |
| Better RAM/latency/energy than alternatives | Not established | Matched-quality/device/input comparisons against simple strong baselines |
| Full formal safety/IEEE/AD | Scoped operator/proof checks only | Complete source/execution/proof and independent qualification |
| Dual cheap-memory verification/reconstruction | Research hypothesis | Implemented authentic-source retrieval with error and full cost evidence |

P3 remains the stronger practical byte-prediction baseline on the original
paired4096-target study. The new04132-target cohort is different; do not compare
its2.0077 NLL directly to040's2.08534 as a new model gain. Repeated seeds share
source documents/targets, not five independently sampled datasets. No customer
adoption, revenue, team/IP rights or physical-phone traction is established by
these reports; those may exist elsewhere but must be documented separately.

## Fundraising position

It is reasonable to apply now to research-oriented angels, accelerators and
pre-seed funds as a research-stage company. Eligibility is not the same as a
strong investment case or a high probability of funding. [YC's FAQ](https://www.ycombinator.com/faq/)
explicitly welcomes very early companies; its [seed guide](https://www.ycombinator.com/blog/this-brief-guide-is-a-summary-of-what-startup-founders-need-to-know-about-raising-the-seed-funds-critical-to-getting-their-company-off-the-ground)
stresses team, market and customer evidence as well as the idea.

An honest positioning statement: "We are developing offline, source-grounded
predictive memory for a specific constrained-device workflow. We have genuine
reproducible small-model experiments and an instrumented research loop. We are
raising to establish a quality/cost advantage and validate a customer use case."
Do not pitch proven Transformer replacement, general intelligence, square-root
neural memory, guaranteed exact recall or certified safety.

Before a stronger architecture investment claim, define one customer/task and
predeclare a practically meaningful quality/cost threshold. Measure the same
real inputs/targets on an actual target device, against P3/simple retrieval and
matched neural controls. Report NLL/BPB, historical identity recovery/wrong-match
rates, p50/p95 end-to-end latency, peak total RAM, energy and retained-source bytes.
Demonstrate reproducibility on fresh authorized data and a customer-visible
working workflow. Multiple approved model/data scales and learning curves are
needed before projecting training requirements or economic scaling. No GPU
fleet/valuation forecast follows from these tiny CPU studies.

## Mathematical interpretation of the dual-memory proposal

Two fixed-size lossy memories cannot guarantee recovery of arbitrary histories
merely by agreement or checksum. If b bits represent an arbitrary N-byte history
and b<8N, pigeonhole collisions exist. Exact arbitrary recall requires sufficient
retained information or recoverable external source; count its storage and I/O.
Different memory encodings may have correlated errors. Checksums test particular
identity/integrity conditions, not source truth or neural semantic correctness.

With verification trigger rate p and an audit fraction a on untriggered cases,
verification frequency is r=p+(1-p)a. Cost includes both cheap paths, routing,
updates, verification/reconstruction, retained source and runtime overhead.
Measure all of them; a conditional path is not automatically cheaper overall.

[Williams' STOC2025 result](https://doi.org/10.1145/3717823.3718225) concerns general
computation simulation, not recovery from lossy neural states at low latency.
It does not supply a TAC algorithm or a neural memory guarantee. The transcript
mentioned by the operator was not attached in this task; primary sources were
consulted instead. Dual memories, bounded candidate verification and recomputation
remain sensible hypotheses whose value must be measured, not assumed.
