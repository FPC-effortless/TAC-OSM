# PLM-018A — Amortized target-blind probe policy

Status: PREREGISTERED / IMPLEMENTATION.

018A is the first learned CDL experiment. It does not replace the exact
G-CASM teachers. The exact greedy budget-aware teacher is the reference policy;
the learned network is successful only if it approaches that teacher on untouched
test seeds.

The learner sees compact per-action statistics of the current six-bit
activation-trace belief. Exact U_B and exact Shannon entropy are deliberately not
passed as features. The sketch can be updated incrementally when candidates are
removed from the current belief, making persistent belief maintenance part of the
runtime rather than recomputing every candidate partition from scratch.

## Split

Training uses seeds 0-19, checkpoint selection uses validation seeds 20-24, and
the confirmatory test uses seeds 25-29. Test teacher labels are generated only
after the checkpoint is frozen.

## Baselines

The exact U_B teacher is the primary ceiling for the learned policy. Exact
Shannon-information selection is a classical information baseline. Generalized
binary search is the conceptual scalar-output special case; it is not treated as
an exact theorem for arbitrary six-bit trace channels.

This follows the active program-learning/OGIS literature: information-theoretic
question selection has been used for interactive program synthesis, while
LearnSy shows that efficient question selection is itself a distinct optimization
problem. Deep Adaptive Design provides the amortization template for turning an
expensive adaptive design computation into a learned one-pass policy.

## Required scientific comparison

The primary unit is test seed, not individual task. Teacher agreement is
secondary because matching a teacher's action does not itself establish useful
downstream capability.

The learner is evaluated on:
1. U_8 regret;
2. verified B=8 success;
3. teacher agreement;
4. explicit belief/policy computation work.

No submodularity guarantee is assumed.

## External track

A later independent evaluation will use public CodeARC data and its hidden-function
oracle interface. The external benchmark will not contribute training examples
or teacher labels to the 018A model.
