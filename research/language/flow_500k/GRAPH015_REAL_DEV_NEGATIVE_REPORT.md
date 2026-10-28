# GRAPH015 — delta-corrected 500K TAC, authentic exposed DEV (negative gate)

**2026-10-28 | ENGINEERING TESTS PASS; NATIVE PREDICTIVE UPLIFT FAIL.** Do not merge or promote. Existing System-1/language family, not a seventh model. All original GRAPH009–014 records untouched.

## Finished genuine-text experiments
- Newly trained prospective GRAPH015 TAC: the original WaveCED500K architecture except its three associative banks use an error-correcting delta write with learned retention, still exactly **499,290 learned parameters**. Retained all original alpha, erase, write parameter rows and active gradients.
- All five seeds 1701–1705 fitted on the original **30 authentic Vim9.1 FIT** publisher chapters, with identical 120×8 training sample schedules, batch8/context128, next-character cross entropy, AdamW lr=.001, clipped norm1. Each new complete optimizer/RNG/sampler checkpoint restored and rescored without fitting.
- Authentic old DEV six Vim9.1 documents, already examined in GRAPH013–014. **6,144 scored real targets per seed**; 36 original source document hashes verified; source/labels never called independent.
- Compared original frozen WaveCED TAC (499290) and new GRAPH014 selective diagonal SSM (498288); both keep original weights and local replay records. All three models scored with *the same exact count cache and the same frozen learned approximate-query distribution* per token, not model-tailored policies.
- Seven original constructed numerical tests passed: active count, gradients, gate bounds, future poisoning, spectral bound, exact corrective keyed error reduction and parallel/serial scan AD equality. Tests validate algebra, not prediction.

| Five-seed mean next-character NLL ↓ | New delta TAC | Original TAC | New SSM |
| --- | ---: | ---: | ---: |
| No external retrieval | **2.563868** | **2.537438** | **2.518730** |
| + same GRAPH009 exact cache | **2.467237** | **2.442860** | **2.430226** |
| + exact cache and same GRAPH011 query | **2.469636** | **2.445901** | **2.430577** |

The delta candidate is **worse than original TAC in all five seeds**, mean +0.026430 nats/character native and +0.023735 post-common-cache. The learned approximate query again hurts the average delta candidate after exact cache on this exposed Vim corpus (+0.002400 NLL). Do not import its positive Python external-source outcome as native TAC ability.

| Seed | New delta TAC native | Delta + exact | Delta + exact + query |
| --- | ---: | ---: | ---: |
| 1701 | 2.601436 | 2.499915 | 2.497420 |
| 1702 | 2.558135 | 2.456719 | 2.463072 |
| 1703 | 2.544116 | 2.452251 | 2.457761 |
| 1704 | 2.537525 | 2.446547 | 2.446217 |
| 1705 | 2.578127 | 2.480752 | 2.483712 |

Exactly 809 exact hits, 5,143 approximate eligible misses and 192 no-history positions per seed. All five pairs agree on cache hit masks; all probability distributions normalized. Each new checkpoint's stored native DEV NLL replayed within tolerance 1e-6.

## Mathematical interpretation
The input-only affine recurrence has right-action form:
`M_t = M_(t-1) P_t + Q_t,  P_t=rho_t I - beta_t k_t k_t^T, Q_t=beta_t v_t k_t^T`.
For normalized `||k||_2=1`, `0<beta_t<=rho_t<1`, eigenvalues are `rho_t-beta_t` along `k_t` and `rho_t` perpendicular; hence `||P_t||_2=rho_t` for key dimension≥2. Consequently `||M_t||_F <= rho_t ||M_(t-1)||_F + beta_t ||v_t||_2`. With `rho_t<=.999`, `beta_t<=.999` and bounded value, a loose uniform bound exists for admissible inputs, but the bound is NOT proof of useful factual retention, stronger language prediction or a complete IEEE/Lean/AD refinement.

The strict higher retention floor may preserve interfering content as well as useful old information. The data support only **this candidate failed**. They do not isolate the precise mechanism of the failure or imply every delta rule fails. Earlier original vs new SSM comparison still has unverified FLOP parity and different optimization histories.

## Compute accounting (one warm-up style profiled optimizer step, same [8,128] shape, CPU eager PyTorch2.10)
| Model | PyTorch *counted* per-step operations |
| --- | ---: |
| Original TAC | 3,219,040,346 |
| New delta TAC | 3,218,895,962 |
| New selective SSM | 2,931,397,378 |

**These are NOT certified end-to-end FLOPs**. PyTorch leaves many operator categories without measured FLOP values (TAC: 220; SSM: 140). Old/new TAC are almost identical on *counted* work, but exact parity is not established; new SSM counted fewer. No architecture-superiority claim.

## Open parallel gates
1. **Native recurrence:** implementation and five seeds complete, **DEV UPLIFT FAIL**; keep failed checkpoints.
2. **Strict training-compute/FLOPs:** **OPEN** (incomplete op coverage and no independently certified full original training budget).
3. **Git/LFS original trained binaries:** **BLOCKED** (original 5×~5.8MB delta checkpoints held in downloadable archive, not in repo; original TAC/SSM also external). SHA-only manifest, zip, CI, or state dict reconstruction do not qualify.
4. **Untouched genuine TEST:** **NOT RUN**. Previously exposed Vim and GRAPH011 Python remain ineligible for new independent confirmation; no unconsented sources accessed.
5. **Complete PNDS Lean/PyTorch/IEEE/AD refinement:** **OPEN**; only algebraic conditions and constructed numerical checks established.

Next is not to tune the delta gates against exposed DEV to manufacture a win. Keep this negative, investigate objective-versus-retention tension with observed grad/corruption probes, instrument complete training compute and obtain lawful sealed real documents before final interpretation.
