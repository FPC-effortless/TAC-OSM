# G-CASM-016C — Trace-to-Capability Bridge

**Status: PREREGISTERED / IMPLEMENTATION**

The confirmed G-CASM-015 measurement established an observation-channel result: on the registered finite graph workload, the activation trace carried substantially more one-step information per environment work than scalar output.

016C asks the next causal question: does that information advantage translate into exact selective computation under a fixed terminal candidate budget?

## Primary test

At M=512 and B=8, each seed contributes 32 paired scalar/trace task outcomes. Within each seed, verified success is averaged across tasks. The primary effect is:

    mean(trace success) - mean(scalar success)

A 4,000-resample bootstrap resamples the five seeds and recomputes the paired seed-level effect.

## Probe selection

Both arms use deterministic maximum information-per-environment-work selection over the same 16 input-row actions. Candidate-predicted evidence is public; target identity and realized target evidence are not available until after selection.

## Terminal computation

After the target probe evidence is revealed, candidates with exactly matching predicted evidence form the compatible bucket. At most eight candidates are retained in fixed public library order. Exact CASM verification then checks the full 16-row truth table for each retained candidate.

There is no target-truth shortcut, learned ranker, or privileged verifier label in the selection path.

## Interpretation

A positive primary result demonstrates the observation-to-capability bridge on this synthetic finite workload.

A non-positive result is equally useful: it would locate a remaining bottleneck between evidence representation and terminal computation rather than justify immediately adding another router.

## Scope

016C does not establish learned probing, sequential optimality, asymptotic scaling, multimodal capability, or general access to internal state.

## Registered prediction and cost interpretation

The exact 015-derived M=512/B=8 partition utilities are preregistered before the
016C result: scalar U_8 = 0.03125 and activation-trace U_8 = 0.848046875.
These are expectation ceilings under the uniform target prior, not observed
016C success rates.

The report separates two axes. Both arms acquire one graph execution per probe,
which isolates evidence-channel capacity. The activation trace additionally
incurs six trace-read units, so explicit total-work accounting is reported as a
secondary cost view. The primary endpoint remains verified capability at the
same one-probe terminal budget.

## Confirmatory trigger
[run-graph-casm-trace-capability-bridge-016C-full]
