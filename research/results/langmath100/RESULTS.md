# LANGMATH100: trained-state attribution results

Executed in draft PR #678. All three original exact10M models, seeds9901/9902/9903,
were loaded unchanged. Dedicated run41164962393 SUCCESS for all three jobs.
Each job passed71 tests (original scaled/probe checks plus38 new constructed
intervention checks), then reproduced original initial/trained language and
all five episodic condition outputs.432 acquisition states and270,336 language
byte targets were replayed. All recorded parameter/optimizer/source identities
were checked. Zero optimizer steps, protected TEST, promotion or merge.

138 new prediction sets =46 conditions ×3 seeds;9,936 source-case-condition
rows. The32 factorial conditions reset every combination of recurrent context,
three-byte embedding cache, previous key, fast matrix+mass and slow matrix+mass.
Additional legal-donor, address-shuffle, continuous bank-clamp, no-write,
query-only, every-byte-state-reset and raw-history-replay controls are separate.
Clock and owner identity are preserved except the explicit fresh-start control.
Bank matrix and mass are always coupled; independent cell corruption is not
misrepresented as a legitimate normalized-bank state.

## Results

Lower first-byte NLL is better. Exact results below list seeds9901/9902/9903.

| Domain | Condition | First-byte NLL | Exact task successes per seed |
|---|---|---:|---|
| lexical | reset_00 | 3.283519 | 0/32, 0/32, 0/32 |
| lexical | reset_08 | 3.304336 | 0/32, 0/32, 0/32 |
| lexical | reset_16 | 3.321854 | 0/32, 0/32, 0/32 |
| lexical | reset_24 | 3.342178 | 0/32, 0/32, 0/32 |
| lexical | off_fast | 3.087114 | 0/32, 0/32, 0/32 |
| lexical | off_slow | 3.449472 | 0/32, 0/32, 0/32 |
| lexical | off_both | 3.243763 | 0/32, 0/32, 0/32 |
| lexical | reset_31 | 3.356036 | 0/32, 0/32, 0/32 |
| lexical | stateless_every_byte | 4.010649 | 0/32, 0/32, 0/32 |
| lexical | direct_history | 3.283519 | 0/32, 0/32, 0/32 |
| arc | reset_00 | 3.157151 | 0/8, 0/8, 0/8 |
| arc | reset_08 | 3.384969 | 0/8, 0/8, 0/8 |
| arc | reset_16 | 4.322217 | 0/8, 0/8, 0/8 |
| arc | reset_24 | 4.525608 | 0/8, 0/8, 0/8 |
| arc | off_fast | 3.124026 | 0/8, 0/8, 0/8 |
| arc | off_slow | 4.474779 | 0/8, 0/8, 0/8 |
| arc | off_both | 4.898336 | 0/8, 0/8, 0/8 |
| arc | reset_31 | 3.818525 | 0/8, 0/8, 0/8 |
| arc | stateless_every_byte | 5.944980 | 0/8, 0/8, 0/8 |
| arc | direct_history | 3.157151 | 0/8, 0/8, 0/8 |

Continuously disabling fast memory improved positive lexical first-byte NLL
by0.196404 nats; disabling slow memory worsened it by0.165954. Both directions
held in all three seeds. These are frozen-model inference interventions, not
trained replacement architectures. Slow reset once and continuous slow clamp
are different: one allows subsequent query writes, the other suppresses every
matrix/mass update and read throughout query/generation. Removing both banks
can produce cancellation and is not equivalent to removing all state.

All46 conditions still produced0/32 positive lexical completions and0/8 exact
ARC solutions in every seed. The12 registered bank-capability gates all FAIL.
No positive integrated architecture or scale-up gate is earned by these scores.
Uniform lexical code completions remain separate; they do not rescue the
positive reuse endpoint. ARC first-byte probabilities are format-sensitive.

The full results include all case-group means, cluster bootstrap intervals,
factorial pair interactions, donor unavailability and Holm corrections. Only4
lexical source clusters and8 ARC tasks exist. All registered exploratory
sign-flip effects have Holm-adjusted p=1.0. Bootstrap intervals excluding zero
must not be reported as multiple-comparison-corrected statistical discoveries.
Some direct-replay deltas are around1e-7 from FP32 scan ordering, far below the
3e-5 correspondence tolerance; no capability claim is drawn from them.

## Information availability

All32 positive lexical four-byte targets occur in the acquisition input; two
also occur in the query. There are31 distinct lexical queries, one duplicate
group and no identical-query/conflicting-target pair. All8 ARC queries are
unique. This finite-cohort audit does not establish an information-theoretic
stateless ceiling. A new delayed-cue and oracle-query solvability protocol is
still required to determine whether failure is acquisition, addressing,
readout, spelling/format or insufficient training. Do not infer the cause from
the zero-success endpoint alone.

## Slow memory and Transformer behavior

Suppressing slow memory leaves recurrent context, the recent-byte cache and
fast memory. Suppressing both banks still leaves context/cache/key state.
Resetting ALL state between episodes still allows recurrent computation within
the query. Resetting ALL state before EVERY byte removes cross-byte history
from this inference condition. None of these operations adds self-attention,
query-to-sequence-token access or Transformer blocks. This is not conversion
into a Transformer and it is not a matched trained Transformer comparison.

Current099 lexical state is3 learned byte embeddings. Earlier work has at
least two further cache types:
- GRAPH009 PR533, bayesian_cache_scan_v5.py: a bounded512-event source-count
  cache, distinct from learned neural state. Its probability gains and costs
  belong to that original external-cache study, not automatically to099.
- LANGDECODE064 PR603, scripts/langdecode064_cached_cap.py: one cached7-number
  capped positional vector per decoder, an inference optimization without new
  learned memory. Later guards/replications remain separate.
- LANGSPEED044 PR573: a KV cache in the Transformer comparator. This is not a
  native099 KV cache. Sharing the word cache does not make these equivalent.

## Same-host cost measurement

See TIMING.json: original trained seed9901, FP32 CPU, two threads, batch one,
3 warmups/15 samples, prefix lengths64/256/1024. At1024 bytes, median native
step2.400ms, slow-clamp2.420ms, both-clamp2.432ms, fresh single byte2.438ms;
raw-history replay plus query173.638ms. Runtime clamps retain original dense
computation and allocation; these are NOT a physically pruned implementation
or production/mobile timings. Native output vocabulary is bytes, so one token
is one source byte and tokens/sec equals source bytes/sec for this model.

Nominal dense arithmetic9,921,852 MACs per byte, before scan/elementwise work.
The supported ATen profiler reports19,856,817 FLOPs per single step and
20,351,997,717 for1025-byte replay; uncounted operators are listed. These are
partial operation counts, not hardware instruction totals or energy. All
conditions still allocate11,104 tensor-state bytes. Dropping tensors from a
future pruned implementation has not been benchmarked here. No production
speed/quality superiority or energy result follows.

## Program status

001: baseline reproduction complete.
002: factorized current-benchmark attribution complete; new delayed/oracle task pending.
003: new matched trained memory-free/enabled candidates pending preflight.
004: repeated-depth candidates pending exact graph/math and matched-budget design.
005: learned hierarchy pending exact graph/math and functional tests.
006: adaptive byte/token/patch integration pending causal-token baseline.
007: new structure/composition benchmark pending identifiability/solvability.
008: frozen-reference costs complete; trained A1-A6 comparisons pending models.
009: independent protected replication requires an access-separated assessor;
     no protected source is opened or self-certification substituted.
010: integrated candidate remains conditional; no mechanism has passed the
     required exact-task/quality-compute gates in this battery.

This is the first completed frozen-model campaign in the unified program,
not completion of all ten stages. No new training occurred. The original
model computational graph, historical failures and PNDS proofs remain unchanged;
whole-model IEEE/AD refinement and intervention-wide formal proofs are OPEN.
