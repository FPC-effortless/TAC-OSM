# PAR-MEM-008 — real-source trained TAC writer-key address collapse

**Observed result: TAC's trained 8D WRITE keys occupy only ~2–3 effective
dimensions over 16–256 real recorded events.** This is a genuine input/
checkpoint-based address geometry diagnostic. It does NOT establish
improved recall, independent generalization, or source-authorized
deployment of another address algorithm.

## Original real source and frozen models

[Preregistered protocol](https://github.com/FPC-effortless/TAC-OSM/blob/research/parmem008-real-writer-address-rank-2026-10-27/research/parallel-six/PAR-MEM-008-PROTOCOL.json) |
[Complete original five-seed source results](https://github.com/FPC-effortless/TAC-OSM/blob/research/parmem008-real-writer-address-rank-2026-10-27/research/parallel-six/PAR-MEM-008-RESULTS.json) |
[Readonly diagnostics](https://github.com/FPC-effortless/TAC-OSM/blob/research/parmem008-real-writer-address-rank-2026-10-27/scripts/parmem008_real_writer_key_rank.py).

Five existing **unchanged, original source-trained** 96,639-parameter
TimeAwareParallelResidualWorld checkpoints from PAR-TIME-005. Their
real UCI Beijing PRSA2017 12-station source ZIP is exact SHA256
`d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8`.
Source input rows were genuine and previously observed, including
chronological irregular gaps. Source numerical years 2015–2017 were
never parsed. Original source March–April 2014 FIT and May–June 2014
**previously exposed development** were evaluated separately.
No source target labels were reconstructed or scored in this rank study.

The study obtained each actual source writer key at the real prior
admission using the frozen `core.address(embedding_t)` computation.
Recording individual old keys is a **privileged source trace** and
does NOT increase the actual deployed 154-FP32 TAC state.
No new neural model was trained.

For each of the five seeds, 12 station streams, and original time
anchors 256/384/512/640/767, we inspected 8D normalized key histories
of 16/64/128/256 genuinely observed events. Measured:
- normalized Gram/covariance spectrum \(G=K^\top K / T\);
- spectral Shannon effective rank \(r_{eff}=\exp(-\sum_i p_i\log p_i)\),
  with \(p_i=\lambda_i/\sum_j\lambda_j\), a quantity between 1 and 8;
- mean absolute pairwise key cosine within each true history window;
- true source-writer key cosine over fixed chronological lags
  1/16/64/128/256 source observations.

An input-only 8D known-at-write/read Fourier ordinal reference was also
evaluated using registered periods 7/13/29/59 source-observed event
positions, with normalized sine/cosine features. **This reference is
NOT an independently trained memory model and has NO predictive score.**
It measures how much address diversity is available in the same
eight-dimensional space with input-only, source-legal positions.

## Results from real formerly exposed 2014 development source

| Events in address window | Original trained TAC key effective rank | Source-index Fourier reference effective rank |
| --- | ---: | ---: |
| 16 | **1.8111 / 8** | 5.3758 / 8 |
| 64 | **2.3196 / 8** | 7.9294 / 8 |
| 128 | **2.5153 / 8** | 7.9589 / 8 |
| 256 | **2.6710 / 8** | 7.9865 / 8 |

Mean absolute off-diagonal source writer-key cosine is 0.8225
within 16-event windows and 0.7111 within 256-event windows;
the source-index reference is 0.2860 and 0.2822, respectively.

| True observed event distance | Trained TAC original source write-key mean cosine |
| --- | ---: |
| 1 | 0.93543 |
| 16 | 0.71132 |
| 64 | 0.66186 |
| 128 | 0.64784 |
| 256 | 0.68869 |

The key correlation therefore extends well beyond adjacent genuinely
recorded observations. The source model's keys have very low effective
rank compared with an eight-dimensional source-known ordinal reference.

Original real FIT source effective ranks were likewise low:
1.7079 at 16, 2.2012 at 64, 2.4469 at 128, 2.6117 at 256 events.
This supports a descriptive geometry problem on BOTH real original
training and previously examined development streams.

## Interpretation and explicit limits

PAR-MEM-006 found that actual 16/64/128 source historical reading
with a timestamp key does not beat matrix-blind or current-value
baselines. PAR-MEM-007 found even the original correct privileged
content-dependent writer key does NOT fix retrieval. This new
PAR-MEM-008 study identifies a plausible *underlying contributor*:
**the real trained writer keys use a narrow subspace and mix many
observations into similarly addressed 8x8 memory updates.**

However, low effective spectral rank is NOT a mathematical proof that
all historical information is destroyed; the true rank of the 8D
keys may be eight even when spectral effective rank is ~2.67.
Nor does a high-rank untrained Fourier reference establish source
observations would remain decodable after 256 recurrent writes.
Actual downstream memory target retention still requires testing.

This is NOT a trained eight-dimensional key architecture comparison:
there is no equal-compute learned Fourier-TAC and no independent TEST.
No architecture superior performance, action policy, calibrated
probability, autonomous safety or long-term recall capability is
claimed. Original source-trained weights/checkpoint bytes are unchanged.

## Next preregistered-capability research direction

1. Create a within-six-family **parallel legal-address candidate**
   where the same compact source-known identity/ordinal code drives
   WRITE and READ. Preserve input-only gated affine associative scan
   and constant-size state; do not store any per-event oracle keys.
2. Keep observed content vectors separate from source-known addresses.
   Evaluate **write-vector effective rank, retained content information
   and interference**. Fixed 8x8 state (64 floats) is too small to
   promise lossless access to arbitrary independent 128/256 readings.
3. Add selective admission/overwrite with source-known flags and
   persistence-cost constraints, rather than blindly writing every
   source event, and ablate age/position separately.
4. Formally model source-time address and associative affine
   composition in PNDS Python + Lean; prove exact-real
   chronology/correct-key properties first, then finite-precision,
   autodiff, and probability refinement separately. Do not infer
   full-model correctness from a local address lemma.
5. Run source-authorized true-reconstruction objectives on new
   independently held-out real cohorts. Match **all trained parameter
   counts within 1%**, source data/splits, optimizer and complete
   forward+backward+optimizer FLOP budgets. CPU time need not match.
   The untrained Fourier rank here does not pass the training fairness
   gate. Keep old 2014 exposed DEV separate from sealed TEST.
6. Keep every trained checkpoint, optimizer state, RNG/source order,
   stream states, and clean replay; preserve negative outcomes.

The study is an unmerged stacked draft. It does not add a seventh
TAC research family. Original trained/source evidence is unchanged.
