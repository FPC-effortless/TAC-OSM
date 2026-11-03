# Post-null candidate repairs — hypotheses, not COLLECT016 results

These proposals follow the first connected-state models failing even local copy
queries. They do not change the frozen fits or claim the matrix has reached its
capacity. COLLECT017 adds an explicitly post-null attention learnability control
before attributing failure to the phase lane. Neither study establishes convergence.

## 1. Make the first uncertain answer the explicit retrieval target

The current four-answer-byte loss averages all four positions. The first byte
receives one quarter of that term; later positions also see preceding true answer
bytes through ordinary teacher forcing. This is legal causal training, but
improving word completion is not equivalent to learning the initial retrieval.
The first answer term still has 96 times the weight of one natural byte at this
384-byte length, so a shortage of per-token supervision is not established.
Measure these terms separately, then preregister a first-byte-weighted curriculum
that establishes short-delay binding before increasing delay. Keep ordinary
language loss separate, and give controls the same targets/exposure/compute budget.
This is a proposed optimization repair, not proof that its current weighting is
the cause or that reweighting will work.

## 2. Contextualize the address path before increasing the state

Source inspection and the fixed-checkpoint address probe show the first phase
layer's address at the final query space depends only on that byte embedding. It
cannot directly encode the preceding multi-byte identifier in that address.
Later layers and query-driven state updates are contextual, so this is not a
proof that the whole model is query-independent. In fact later addresses change
across queries; sensitivity alone does not show correct binding.

A source-specific candidate is to feed already computed native fast/slow/rotary
context into the phase content/role/router path while keeping value construction
separate. This targets where to write/read without increasing archive size or
adding full token attention. It must preserve causality, owners, chunk boundaries
and all live parameter counts, and needs an ablation before any benefit claim.
An alternative is a small causal shift/convolution or learned key-span compositor
that produces the same identity representation in record and query roles. Bank
selection must align as well as the continuous phase vector. A context-free
value-byte router is not automatically an identifier router.

Local mixing before recurrent Q/K/V is established prior art. Gated DeltaNet
explicitly uses short convolution on these paths [1]; BASED also studies local
mixing alongside linear attention [2]. Adding a convolution is therefore not a
novelty claim. The tested question would be whether this specific native-context
reuse repairs our byte-level binding at matched resources.

## 3. Separate immediate prediction from admission for future use

For the unfloored normalized write and the same unit read/write key,
read_after=(1-g)*read_before+g*v. Thus admission can improve the current local
output while also changing persistent storage. A better immediate predictor need
not preserve the association needed later. A candidate is a separate immediate
value branch plus a historical read whose write gate is trained for delayed use.
This is not implemented or benchmarked here, and a clean trained control must
isolate it from extra parameters or hidden state.

Also distinguish balanced token routes from balanced useful writes. COLLECT016's
Switch auxiliary balances all token selections, even when their relevance to
future fact queries differs. An admission-weighted or role-conditioned balancing
candidate could reduce filler pressure, but can cheat by admitting nothing.
Delayed-query success, coverage and explicit storage/compute costs must constrain
that experiment. There is no basis for prescribing a universal 95% discard rate.

## 4. Treat higher-order state and exact archives as later capacity tests

The September 2026 Triadic Linear Attention preprint introduces a third-order
recurrent tensor with two address axes [3]. This is relevant prior art for
conjunctive identity/role binding. Its second-key dimension E multiplies state
size by E; extra capacity is not free. Author-reported improvements were not
replicated here. Any future comparison must match state bytes/search work as
well as parameters and training exposure, and test capacity after basic binding
works. Merely using two keys cannot support a novelty claim.

For an exact side store, retain the COLLECT015 falsifiers: current residual is
neither future importance nor a guarantee of future loss. Scale normalization,
age-stratified reserves or a small reservoir hedge can be tested under fixed
bytes and admission/search work. Their utility remains untested in this LM.
Typed coverage/count/belief state is useful only for its specified questions;
an archive used for arbitrary exact recall must be counted as retained memory.

## Primary sources inspected

1. Gated Delta Networks, section 3.4 and short-convolution ablation:
   https://arxiv.org/html/2412.06464v3
2. BASED, ICML 2024:
   https://proceedings.mlr.press/v235/arora24a.html
3. Triadic Linear Attention, submitted 29 September 2026:
   https://arxiv.org/abs/2609.36529 and https://arxiv.org/html/2609.36529v1

No extra model fit, new benchmark outcome or global-attention equivalence follows
from this design note. The completed native-attention control also fails primary copy under the same
512-update budget. This makes local learnability the immediate checkpoint before
capacity scaling; it does not identify a unique cause or prove a repair works.
