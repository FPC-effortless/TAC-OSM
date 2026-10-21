# ME045 original verified results — routed multiscale TAC does not clear performance gates

**2026-10-21, immutable original FIT and historical DEV records; no retuning, refitting, target switching or held-out final-test opening.** [Draft PR310](https://github.com/FPC-effortless/TAC-OSM/pull/310), pinned AVX2 original source commit **2fa77a9cdd0262633290d1631e8f75fb000758f1**, fully successful [GitHub Actions run39664196205](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39664196205).

## Executed original research

- Original exact UCI Tetouan publisher and ETTh1 real measurement hashes, numeric outcomes parsed ONLY FIT + repeatedly exposed historical DEV. Two actual domains, H96/720, seeds45001 and45002, and two registered full training lanes.
- Eight original source-only 500-update AR parent fits and eight original **1000-update real-data-trained common AR/context backbones**, then F0 cheap weights frozen for every comparison. Source parent selection endpoints inspect **prior-exposed DEV**, hence NO genuine novel independent qualification.
- **48 independently trained complete-model expert/router fits:** native TAC fast32/slow32 + 8x8 error-correcting associative memory, matched-parameter GRU40, smaller practical GRU16. At equal 1000 expert/router updates, training anchors and true target windows exactly matched within seed/domain/horizon. Another lane charges 10 optimizer CPU seconds (not identical training exposure). All models preserve same strong pretrained context and multiscale QR16 spline/seasonal residual basis; router inference truly skips unselected channel/windows.
- Every trained checkpoint scored three physical operating modes without retraining: original **adaptive hard route threshold 0.5**, expert **always**, and **cheap-only**. Actual dispatch flags, gate probabilities, true measured outcomes, predictions, training CPU, optimizer/RNG, latency, parent CPU, source/seed/splits stored.

## Exact historical DEV results — primary standardized MSE (lower better)

### Equal 1000 genuine training updates

| Trained system / inference mode | ETTh1 oil MSE | Tetouan power MSE |
|---|---:|---:|
| frozen context, cheap only (identical across all three arms) | **0.154665281** | 0.111156733 |
| smaller GRU16, adaptive | 0.158181180 | 0.110240650 |
| matched GRU40, adaptive | 0.158431153 | **0.109387800** |
| native TAC, adaptive | 0.155126769 | 0.110615245 |
| native TAC, expert always | 0.15549161 | 0.11076552 |

Native vs matched GRU40 improves ETTh1 oil MSE by 0.003304384 at equal updates, **but loses to the frozen context by +0.000461488**. On Tetouan the matched GRU40 beats native by 0.001227446. Thus **no cross-domain wins**.

ETTh1 H720 oil MSE under this new frozen-backbone protocol: native adaptive **0.19939062** vs cheap **0.19781345**, registered target **<0.190500 FAILED**. This is also worse than the previous **ME044** separately trained native .191914, but because the training objective/backbone are different that comparison is NOT an equal-training architecture ablation.

### Equal ten seconds of actual optimizer CPU

| Trained system / inference mode | ETTh1 oil MSE | Tetouan power MSE |
|---|---:|---:|
| frozen context, cheap | **0.154665281** | 0.111156733 |
| GRU16, adaptive | 0.15535457 | 0.11062236 |
| GRU40, adaptive | 0.16232476 | **0.10965967** |
| native TAC, adaptive | 0.155528615 | 0.110146489 |

Tetouan fixedCPU native gate MSE **<0.101500 FAILED**. The native is poorer than GRU40 on Tetouan and cheaper frozen context on ETTh1 at both budgets.

### Actual hard routing and complete batch-one CPU latency

| System at 1000 updates | ETTh1 executed fraction | ETTh1 average ms | Tetouan executed fraction | Tetouan average ms |
|---|---:|---:|---:|---:|
| native TAC adaptive | 0.5739 | **1.8537** | 0.5517 | **2.3525** |
| native TAC always | 1.0000 | 2.3591 | 1.0000 | 2.3834 |
| native TAC cheap only | 0.0000 | 0.4303 | 0.0000 | 0.4362 |
| GRU40 adaptive | 0.5806 | 0.7435 | 0.4341 | 0.8668 |
| GRU16 adaptive | 0.5792 | 0.7331 | 0.4127 | 0.8274 |

The runtime includes window normalization, full AR/context encoding, score/router, selected-channel index copies and selected recurrent expert, not merely expert kernels. ME044 original optional TorchScript 1.414x improvement is **NOT** activated in these ME045 eager routed models, and must not be silently credited. Merely routing half the channel/windows does not meet the native **0.45ms** target.

Exact active fitted expert/router trainables per model: native **15,724**, GRU40 **15,746** (+0.14%), GRU16 **8,882** (different smaller capacity; unmatched). These are **expert/router parameters**, not full total parameters including frozen context. Native/GRU40 see equal real training records at fixed1000 updates, but actual CPU/FLOPs differ; equal10sec lane sees different numbers of records. Full simultaneous parameter/data/compute parity remains **UNQUALIFIED**.

## Registered acceptance (immutable)

| Locked requirement | Outcome |
|---|---|
| ETTh1 720h fixed-update oil MSE <0.190500 | FAIL (0.199391) |
| Tetouan fixed10CPU-sec power mean MSE <0.101500 | FAIL (0.110146) |
| Native average full forecast inference <0.450ms | FAIL (~2.1ms overall, eager) |
| Native learned routing selects a nontrivial fraction | **PASS** |
| Native beats both matched GRU40 and cheap in both lanes ETTh1 | FAIL |
| Native beats both matched GRU40 and cheap in both lanes Tetouan | FAIL |
| ALL-GATES | **FAIL** |

Thus **no model promoted or merged**, no state utility or latent System2 claim. User's higher-level *sealed standard-protocol* memory STOP RULE **was not executed**: ETTh1/Tetouan DEV are not sealed independent test sets. Do not stop the entire program as if a sealed test had failed.

## Post-fit paired uncertainty (descriptive ONLY, never gate retuning)

From original saved FIT-trained genuine DEV predictions, a separately computed 2,000-replicate **moving observed-origin block bootstrap**, with original forecast origin stride48, block length max(2,ceil(H/48)), using the **same original measured origins and seeds within each pair**:

| Native adaptive minus GRU40 adaptive primary MSE | Paired point | Origin-block 95% interval |
|---|---:|---:|
| ETTh1, equal 1000 updates | -0.0033044 | [-0.0050571,-0.0005292] |
| ETTh1, equal 10CPU-sec | -0.0067961 | [-0.0114519,-0.0020369] |
| Tetouan, equal 1000 updates | +0.0012274 | [+0.0003456,+0.0021649] |
| Tetouan, equal 10CPU-sec | +0.0004868 | [-0.0011514,+0.0021591] |

**Interpretation warning:** these descriptive CIs hold the two trained seed models fixed, do NOT estimate training-run variation or account for model selection/repeated historical DEV usage, and H720 ETTh1 has only FOUR independent-ish contiguous origin blocks. No honest claim of new-world significance or general superiority. Native ETTh1 was not consistently better than cheap under the same paired historic analysis.

## Original evidence and cross-ISA boundary

- Original two-stage GitHub CI [39664196205](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39664196205) SUCCESS.
- **1046 regression tests, 12 subtests**, exact causal-hard-skip, matched-frozen context, bounded delta, real-gradient, and active-parameter numerical software checks.
- All **48 original model weight/optimizer/RNG records** restored; **32,458,752 scalar genuine development forecasts replayed BIT EXACT in the pinned AVX2 profile across the original adaptive/always/cheap modes**, including measured true labels, q probabilities, routing flags and original cheap forecasts.
- [Original 303,620,884-byte GitHub Actions artifact](https://github.com/FPC-effortless/TAC-OSM/actions/runs/39664196205/artifacts/12377682384), SHA256 **662d3bbed169f380d4ea7808c37bccb70fcff79f363be47417bbf459eb469c9d**, downloaded independently and its full original ZIP SHA verified.
- Original ME044 AVX512 vs AVX2 strict replay **failure preserved**; ME044 had maximum original forecast drift **4.76837158203125e-7**, rather than the 1.14e-6 approximated in the user's synthesized plan. Current AVX2 exact replay is within one pinned numerical variant only.
- These binary checkpoints are currently in a GitHub Actions artifact with limited retention (90 days), **not yet repository-owned permanent binary model shards**. Do not claim perpetual checkpoint persistence without uploading those original bytes. Code, preregistration and this immutable report are in GitHub.

## Architecture-first next decisions

1. ME045 proved hard execution skipping but has **noncompetitive accuracy and latency**. The trained dense expert + same-step labels model often routes ~55-58% of channels; this alone is not a useful compute-efficient native forecast.
2. The useful component remains original ME044 multiscale decoder, which can be carried into a faster lower-overhead GRU16/GRU40 or newly stabilized weight-tied recurrent expert; do not assume all components help when stacked.
3. A future new authentic sealed task with genuine standard multivariate protocol and registered equal-latency treatment is needed to invoke the owner's explicit stop rule. Previous exposed development results cannot stand in.
4. DeepLoop tied-residual scaling was derived for Post-LN looped Transformers; don't blindly apply its alpha/beta constants to TAC's bounded gated fast/slow physics. DeLoop negative repeated tool-call actions require actual action/outcome/cost logs, not ETTh1/Tetouan readings. Source publications are arXiv:2607.13491 and arXiv:2610.19376.
