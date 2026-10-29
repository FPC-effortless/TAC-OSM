# LANGKERNEL061 — frozen vectorized ordinal code microkernel experiment

**Preregistered BEFORE numerical results.** Motivation: measured original LANGPROFILE060 source lines 117–118 consumed ~89 µs and ~80 µs per traced original call (instrumented inclusive times only) repeating `ordinal_code` over four periods 7/13/29/59 at current and prior positions. No full neural model changes are authorized.

## Exact mathematical candidate
Original per period p: `angle_p = remainder(t,p).to(dtype) * (2π/p)` then interleave cos/sin in original PERIODS order. Candidate vectorizes the four remainder/scale/trig calls using one 1D period tensor and broadcasted clock; returns exactly same interleaved shape. Two alternative *microkernels*:
- `vectorized_two_calls`: one vectorized kernel at t and one at max(t-1,0).
- `vectorized_pair`: stack both clock values and compute all periods once, in one tensor kernel. No precomputed cache/retained state; both allocate period and scale tensors per call.

Test FP32 and FP64 across **all clocks 0..8192** plus 65535 and 262144, enforce bytewise exact `torch.equal`, record source SHA and max abs difference without hiding failures. At clocks 128,512,8192 evaluate exactly same current/previous inputs for all arms. CPU FP32 eager batch1, two intraop/one interop thread, two warmups plus 7×64 per-arm same-input timing samples; preserve raw ms/us and p95. The baseline and each kernel perform exactly the original source-defined ordinal task. Do not optimize based on favorable trial only; report failures.

**Accept only as a prospective decoder candidate** if bitwise across both dtypes and all clocks and repeatable kernel median/p95 better without extra persistent state. Otherwise preserve the negative. This experiment **cannot** claim actual TAC token/s, full growing-prefix logits parity, checkpoint compatibility, safe masking, mobile energy or neural quality: source `_advance` remains **unchanged**. Future graph change needs PNDS Lean/Python, strict original 499623p checkpoint/FP32 source-to-output equivalence, 3904B state/snapshots and matched p95 repeated CPU decoding. LANGDECODE057 p95 failure remains OPEN; no training/GPU/TEST.
