# TACOSM-C5-MULTIFACTOR-001

Status: PRE-REGISTERED.

## Purpose

C5-004 and the independent ten-seed robustness experiment show that the
two-factor product-key mechanism can operate at roughly 0.15 M state
reranking cost on this workload, but the candidate union is still a substantial
fraction of the state population.

This experiment asks whether increasing factorization depth from two factors to
three can improve cell resolution without increasing the final shortlist or
changing the teacher and task stream.

## Registered arms

- two_factor_16_beam6: existing two-factor reference; 16 codewords per factor,
  beam 6, 36 admitted cells.
- three_factor_8_beam3: three factors, 8 codewords per factor, beam 3, 27
  admitted cells.
- three_factor_16_beam3: three factors, 16 codewords per factor, beam 3, 27
  admitted cells.

For width 16, three-factor addressing splits the embedding into factor widths
6, 5, and 5.

All arms use:

- H=256
- M={128,256,512}
- K=32
- seeds={10,...,19}
- 100 evaluation steps per seed/M/arm
- eight cosine k-means iterations
- width-16 raw-trained cosine teacher
- training-only codebook fitting
- the same state-distinct downstream executor
- the same task stream

## Primary measurements

The experiment records:

1. exhaustive target recall;
2. selective target recall;
3. proposal target retention;
4. states scored/M;
5. absolute states scored;
6. end-to-end success;
7. arithmetic;
8. nonempty cells and maximum cell occupancy.

## Interpretation

A three-factor arm supports the depth hypothesis only if it reduces states
scored/M relative to the two-factor reference without a material retrieval
regression on the registered workload.

No hardware speedup or universal asymptotic claim is licensed by this experiment.

## Reproduction

Smoke: python scripts/run_c5_multifactor_001.py --smoke

Full: python scripts/run_c5_multifactor_001.py

Artifact: artifacts/TACOSM-C5-MULTIFACTOR-001.json

Full-run marker: [run-c5-multifactor-001-full]
