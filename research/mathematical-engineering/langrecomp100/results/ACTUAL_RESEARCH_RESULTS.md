# LANGRECOMP100 actual activation-recomputation results

**Gradient/update preservation PASS. Registered useful-CPU-memory gate FAIL.** Whole-forward recomputation dramatically reduced logical forward save requests, but increased actual peak CPU memory for the272-slot episodic training batches and made arithmetic updates slower. The tested CPU configuration should retain native backpropagation as its default. No architecture, intelligence or GPU-VRAM advantage is established.

## Actual trained-model trial

Accepted corrected source: `f2f8b616ab7eda5a44635d6a26f0062d103061fe`.
[Corrected Actions run41160473629](https://github.com/FPC-effortless/TAC-OSM/actions/runs/41160473629) completed all three seed jobs, all six actual policy continuations, fresh-process FIT replays, the paired comparison and final actual-size/resource gate successfully. A green execution/parity workflow is not a positive physical-memory result.

Each of the three authentic trained099 parent models has exactly10,000,000 learned parameters and originally completed2,048 AdamW updates. Native and standard PyTorch non-reentrant whole-forward checkpoint policies each resumed16 identical next FIT batches from exactly the same parent weights, all17 original optimizer moment records and restored RNG. Each derived checkpoint therefore has2,064 absolute optimizer steps. The neural model, source data, loss, masks, state semantics, hyperparameters and sampled batches are unchanged between policies. There is no state detachment in model forward or truncated gradient across acquisition within an episode.

At every one of the16 steps in every seed pair, original receipts match loss, complete logits/native-state hashes, all10M raw gradient bytes, all17 gradient norms, global clipping norm, every updated parameter/moment/step byte and Torch RNG. Separate read-only audits loaded the actual original archives and directly compared all final model tensors, all17 AdamW slots and Python/NumPy/Torch RNG; all are identical between native and recompute. All six accepted cold-process actual-FIT model/state replays match bitwise. All archive SHA256 values were verified. No local neural forward or optimizer step was used by the archive/arithmetic audits.

## Logical save requests and physical memory are different

Forward save-hook accounting on representative zero-update parent FIT batches:

| Batch mode | Native logical requested bytes | Recompute logical requested bytes | Reduction |
|---|---:|---:|---:|
|Language B4×T128|109,183,116|1,101,700|98.991%|
|Episodic B4×T272|187,361,848|2,293,392|98.776%|

These are summed forward autograd save requests, not unique live activation bytes, total RAM or CUDA VRAM. Aliases and parameter-save requests can appear repeatedly. Distinct nonparameter requested storage identifiers are separately retained in raw receipts and also do not constitute a live peak. Hooks were absent from all actual timed/RSS update steps.

Accepted Linux whole-process peak RSS DURING measured updates, after resetting the current process VmHWM at each step, with model/AdamW/gradient/runtime/allocator memory included:

| Seed | Mode | Native peak MiB | Recompute peak MiB | Native median step s | Recompute median step s | Recompute slowdown |
|---|---|---:|---:|---:|---:|---:|
|9901|language|650.88|663.24|0.188457|0.239605|27.14%|
|9901|episodic|710.79|774.45|0.372644|0.501829|34.67%|
|9902|language|639.94|667.93|0.255006|0.333955|30.96%|
|9902|episodic|707.88|781.13|0.508030|0.695958|36.99%|
|9903|language|663.14|645.85|0.270837|0.341921|26.25%|
|9903|episodic|706.59|775.33|0.533912|0.714257|33.78%|

On the larger episodic batches, physical peaks increased **8.96%,10.35%,9.73%** across seeds9901/9902/9903. Language peaks were mixed: increases1.90% and4.37%, decrease2.61%. The preregistered useful CPU-memory condition required lower272-slot physical peak in all three seeds; it **failed**. No favorable subset replaces that decision.

Median measured arithmetic updates became26.25–36.99% slower across modes/seeds. Each seed/mode has12 language or4 episodic measurements per policy, so these are small descriptive samples, not deployment p95 estimates. Timings cover forward/backward/clipping/optimizer arithmetic; checkpoint IO, source authentication, digests and cold-import costs are separately reflected in full worker elapsed receipts. Separate policy workers ran serially on the same host within a seed; order was reversed for9902. Hardware can differ across seeds and is preserved. FP32, two Torch CPU threads, PyTorch2.5.1+cpu, no GPU.

Whole-forward checkpointing avoids retaining most forward differentiation saves but recreates intermediate work during backward. That does not bound the backward working set, original outputs, parameter/gradient/AdamW storage or allocator retention. The result is consistent with this distinction; the study does not causally attribute the observed peak to a specific allocation. Every source-model output/state was retained for the paired output audit in the measured worker. RSS figures are training-step phase peaks, not whole-service lifetime, energy, mobile-device or GPU allocator measurements. Process baselines and observed lifetime high-water values are preserved separately.

## Original measurement correction and complete history

[Attempt1 run41159811557](https://github.com/FPC-effortless/TAC-OSM/actions/runs/41159811557), source061dc749674680139600e9895cf5d104e24fa5c8, also completed96 genuine continuation updates and exact gradient/optimizer parity. Its warmup measurement pack hook returned tensors with their autograd references attached, which can retain reference cycles. Its physical-memory figures are therefore **unqualified**, preserved verbatim in ATTEMPT1_RESULT_UNQUALIFIED_PHYSICAL_MEMORY.json and the original archives. They are not used to claim a benefit.

The corrected pack hook returns a detached DATA VIEW, sharing storage while excluding autograd references from the packed representation. This is confined to measurement hooks; it does not detach model-state or learning paths. A new constructed backward regression verifies that the hook preserves gradients. Actual timed updates have no saved-tensor measurement hooks. The correction and one bounded replay were registered before accepting new physical-memory results, with no model, data, hyperparameter or sample-order tuning. Both attempts independently restart from original099 step2,048; they are not sequential32-step continuations. There were192 actual new optimizer steps across the12 derived checkpoints in both attempts, and no third attempt.

The initial replay proof build also failed because prefix was a reserved Lean argument name. That failure/source/run and the corrected names remain in the reciprocal registry and Git history. It occurred before any actual100 continuation. No failure or negative outcome was erased.

## Mathematics and scope

The source wrapper uses torch.utils.checkpoint.checkpoint with use_reentrant=False and preserve_rng_state=True, explicit byte/mask/state inputs and unchanged owner/clock metadata. Integer input bytes do not require gradients; non-reentrant checkpointing preserves actual parameter and incoming-state gradient paths. This is the training adaptation related to LANGRECOMP053, inspired by GRAPH011's time–space tradeoff. It does not directly integrate GRAPH011's inference source-replay engine, nor prove Ryan Williams' universal time/space theorem.

66 source/reference/probe/resource/constructed-optimizer tests pass in Actions. The PNDS companion#101 passes seven independent FULL graph/replay reference tests and four generic exact-real retained-input replay/conditional derivative theorem checks. The original099 bounded native-state/bank/scan proofs are inherited unchanged. Generic replay equality and conditional derivative transfer do not certify PyTorch's checkpoint engine, full IEEE/AD behavior, allocator peaks or arbitrary long training. Those full-source guarantees remain OPEN beyond measured cases. Normal PNDS Lean and reciprocal registry gates also passed on the corrected source. No new neural architecture was introduced.

Genuine original099 FIT-only publisher artifact, licenses and per-file SHA256s were reused; EVAL artifacts were never delivered to continuation workers. Complete original source checks, parameter counts, trained parent hashes and the first2,048 sampler entries were authenticated before the next16 batches. No DEV/TEST model evaluation, checkpoint promotion, new capability claim, architecture-superiority claim, GPU quota or merge occurred. Earlier099 lexical/ARC negative evidence remains unchanged; this trial measured implementation behavior, not improved intelligence.

## Reproduction and originals

Use ORIGINAL_ARTIFACTS.json for both runs' eight ZIP archives and their SHA256 values. They contain all12 original full model/optimizer/RNG continuation checkpoints, step receipts, sampling journals, logical profiles, per-step physical RSS/timing and original cold/replay predictions. All originals are also retained locally; GitHub artifact expiry is90days. Parent checkpoints remain in run41153068368. Small source-controlled summaries here do not replace original evidence.

Check out accepted sourcef2f8b616ab7eda5a44635d6a26f0062d103061fe for exact worker reproduction. Extract artifacts by run ID then artifact name. The optional audit_original_archives.py performs read-only tensor and receipt arithmetic, without any neural model execution:

```sh
python research/mathematical-engineering/langrecomp100/results/audit_original_archives.py --artifacts-root /path/to/originals --run 41160473629 --output /path/to/audit.json
```

The opt-in implementation remains available, while native CPU training remains the supported default for this tested configuration. Segmented/selected checkpoint placement, larger contexts, mixed precision and GPU peak-memory behavior were not tested here; the whole-forward result must not be generalized to them. Internally measured, not independently qualified.
