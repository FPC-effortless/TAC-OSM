# TACOSM-C5-LEARNED-STATE-001

Status: PREREGISTERED — result pending.

## Purpose

This experiment removes the hand-designed Hamming state index from
TACOSM-C5-END-TO-END-001 while keeping the workload, persistent state,
candidate population, exact candidate index, and executor fixed.

The new boundary is:

`noisy query -> learned semantic encoder -> compact binary index -> state value`.

Training and evaluation use disjoint semantic code subsets. Evaluation
addresses and evaluation gold targets never enter the learned index training
loop.

## Registered protocol

- M=64 persistent state items.
- H=64, 128, 256 candidate programs.
- R=4 relevant programs per evaluation state.
- 5 seeds (0..4).
- 32 training epochs.
- 100 evaluation queries per seed/H cell.
- binary embedding width = 8 bits.
- Hamming probe radius = 1.
- state shortlist K=1.
- training codes = CODEBOOK[16:64].
- evaluation target codes = CODEBOOK[:16].
- exact candidate index and fixed 12-work-unit executor unchanged.

## Arms

### Exhaustive

Scan all 64 state items, recover the nearest target code, match all H
candidate programs, then execute all H programs.

### Learned selective

Build the learned state index from persistent state values. For each query:

1. encode the noisy query;
2. quantize the first 8 latent coordinates to a binary bucket code;
3. probe the registered radius-1 bucket neighborhood;
4. retain at most one state address;
5. use the unchanged exact candidate index;
6. execute only the retained four candidates.

## Capability rule

The registered comparison is one-sided:

`learned_selective >= exhaustive - 0.05`.

This prevents the experiment from being rejected merely because selective
retrieval improves the synthetic baseline. A positive capability delta is
reported separately and is not used to change the rule.

## Cost accounting

The learned state path reports three distinct terms:

- query encoder MACs;
- one-time state-index build encoder MACs, amortized over the 100-query cell;
- dictionary probe count and returned bucket population.

These are not silently converted into the synthetic executor work units.

For the registered encoder, query arithmetic is:

`8 * 10 = 80 MACs/query`.

State-index build arithmetic is:

`64 * 80 = 5,120 MACs/build`.

The binary bucket operation and Hamming probes are reported as operation counts.

## Interpretation boundary

A passing result supports only the learned state-addressing mechanism on this
held-out synthetic codebook. It does not establish:

- broad semantic memory;
- long-horizon persistence;
- general ANN retrieval;
- hardware-level speedup;
- natural-language program selection;
- the broad L4 C5 computation claim.

The next gate, if this passes, is to replace the exact candidate index with a
learned semantic candidate index and include candidate-index build/query cost
in the same end-to-end ledger.

## Reproduction

Measurement command:

`python scripts/run_c5_learned_state_001.py`

Contract:

`contracts/TACOSM-C5-LEARNED-STATE-001.json`
