# Autoresearch loop, adapted to this maintenance experiment

Reference: [Andrej Karpathy autoresearch](https://github.com/karpathy/autoresearch)
and its program.md. Reuse the principle of bounded source-controlled experiments,
immutable data/evaluation, explicit measured keep/discard decisions and continued
iteration. Do not copy its single-GPU language-loss benchmark or claim to run
that training program here.

TAC-OSM adaptation: local CPU caps, fixed update/seed matrices, reliability guards,
training-only recipe selection, sealed fresh test worlds and independent replay.
All model/data/fit dependencies must freeze before fitting. The readout is the
only trainable model part in this experiment. Record failures/negative results
and source commits in an auditable TSV; discard means reject the candidate,
never delete the old evidence or reset away inconvenient results. User's existing
authorization covers this bounded research, so no repeated confirmation loop.

Do not optimize sealed test results, alter safety/science thresholds, add an
analytical controller, buy cloud compute or deploy to equipment. If a pilot fails,
retain it and register the next distinct hypothesis. Engineering reproducibility,
component qualification, whole-agent capability and commercial usefulness are
separate decisions.
