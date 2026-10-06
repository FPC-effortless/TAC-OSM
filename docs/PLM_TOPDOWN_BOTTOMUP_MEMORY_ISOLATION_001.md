# PLM Top-Down Assembly / Bottom-Up Memory Isolation — 001

Status: preregistered. Successor to the bounded multi-family temporal-state results.

## Question

The earlier benchmark required the model to use persistent state while implicitly
handling three temporal relation families. Because the relation family was not an
input, that benchmark jointly tested temporal memory and hidden family inference.

This lane removes that confound.

The relation is fixed to one temporal contrast for the entire experiment:
exponential filters with alphas 0.45 and 0.985. The model still receives only the
current observation and persistent state. The current observation is always 0.0.

## Top-down scaffold

environment -> history observations -> persistent state -> conventional tanh MLP
-> fixed binary executor -> outcome -> independent verifier.

The only architectural mechanism under test is the persistent state update.
The policy, executor, outcome and verifier are standard.

## Bottom-up ladder

`no_state -> single_timescale -> two_timescale -> mtsk`.

Every stateful arm addresses the same temporal relation. The primary MTSK versus
single-timescale comparison therefore removes the hidden relation-family task.

## Benchmark integrity

Training and test RNG streams are disjoint. The test stream is new and is not copied
from the earlier MTSK or distributed-transfer experiments.

Each test history is paired with its exact negation. Both members have current
observation 0.0 and opposite correct actions.

No pair identity, filter identity, gold action, verifier output or benchmark
descriptor enters the model.

## Primary decision

At H=256, MTSK must beat single-timescale by more than 0.10 with a seed-bootstrap
95% interval excluding zero, and also beat no_state.

## Scientific boundaries

A positive result would establish that the persistent state mechanism can solve an
isolated temporal relation without hidden relation-family variation.

A null MTSK-vs-single result with all stateful arms above no_state would show that the
earlier multi-family ceiling was not solely caused by hidden family inference.

This experiment does not establish relation-family inference, multimodal competence,
general intelligence, biological equivalence, asymptotic scaling, or hardware
efficiency.

The universal G0-G10/P0-P7 governance protocol remains mandatory.
