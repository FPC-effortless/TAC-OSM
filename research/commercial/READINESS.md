# Commercial acceptance contract — not a product-readiness claim

Research objective: export a neural state estimator/controller whose measured
quality and total operating cost justify it for a specified operational workload.
The current maintenance worlds are synthetic. Their cost units are not currency,
Veritas replay uses the same effect tables, and simulation ground truth is not an
external outcome-verification service. Checkpoint export is not commercial proof.

## Concrete reviewable deliverables

* Torch-only persistent JSON-lines public interfaces, candidate scores, learned
  completion probabilities and parallel typed state judgments.
* Twenty-seed training distributions, all unsuccessful fits, matched retrained
  controls, source/data/checkpoint hashes, strict replay and native execution audit.
* Actual CPU inference latency, parameter and state bytes; retain UNKNOWN for
  physical memory traffic/FLOPs until hardware counters establish them.
* Economic comparisons charging observation acquisition, paid inspections,
  interventions, residual faults and compute separately. Synthetic research
  accounting remains separate from any customer currency conversion.

## External workload gate

Select a concrete application and an untouched, permissioned operational trace
corpus before a commercial claim. Every trace needs public observation timestamps,
identity/relationship schema, task/choice definitions, actual available actions,
inspection prices, execution receipts, independently measured final outcomes,
action authorization and failure costs. Keep an out-of-time/site/customer split.
Do not expose final labels to deployment or allow a completion score to certify
physical success. Ambiguous or unmeasurable outcomes must remain UNKNOWN.

Run the learned agent, optimized public rule/belief controller, matched learned
GRU/MLP, and any available Jev/LLM agent under identical information, tool access,
limits and prices. Record data preparation and integration costs; compare total
cost per successfully verified task alongside residual-fault risk and latency.
No mandatory win over a perfect synthetic rule is manufactured by removing it.
If a simpler controller meets the quality target more cheaply, select that control
for deployment and retain TAC as research. CASM novelty is independently unproven.

A customer-specific revenue/pricing model, measured workload volume, independent
outcome instrumentation and authorized shadow trial are still required. None is
provided by this repository's synthetic benchmark. Until they exist, commercial
viability and competitive real-world capability are NOT ESTABLISHED. No external
systems are changed and no live equipment is controlled by this research work.

## Portable deterministic inference qualification

The exported eager CPU runtimes use float64 and one Torch thread. They omit
`torch.use_deterministic_algorithms(True)` because in pinned Torch2.14 that public
configuration API imports NumPy via compiler initialization. Known eager CPU
operations replay against the flagged audit model with strict1e-8 telemetry and
exact choices. Import-guard tests block absolute NumPy/TAC/Veritas imports while
allowing unrelated relative module names. This is tested local parity, not a proof
of universal hardware determinism; remote replay remains an independent gate.
