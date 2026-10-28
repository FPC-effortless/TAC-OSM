# CONTEXT026 implementation and mathematical scope

UNTRAINED native TAC candidate; parent LANGTRAIN025 source2b99b2c0 and exact
parent model SHA in FROZEN_PLAN.json. No prior source/checkpoint overwritten.
Default500122 trainable/deployed parameters vs499978 parent (+144,0.0288%).
Additional3x48 lag weights are on the actual native read/write gradient path,
not padding. Full training compute/SSM parity has NOT been measured.

Typed new path: byte input[B,T] -> shared lexical embedding[B,T,48] -> causal
3-lag weighted input[B,T,48] -> tanh/native_key48→8 -> epsilon-normalized
query[B,T,8]. Native write key is shifted query[t-1], first native admission
false. Values[B,T,8] still derive from current observed byte+clock. Native
8x8 memory uses original rank-one delta form, native retention capped.995;
current query reads its postwrite matrix. All target bytes remain loss-only.
Other original16x8 aligned heads,8two-dimensional wave modes,6x16 concepts,
fast/slow readout, weighted concept pooling and FFN remain source-explicit in
context_transition.py. Nonzero start is rejected; no carried-state API or
constant-memory deployment measured.520 inherited summary floats exclude any
future context history/cache and scan working tensors.

Equations: l_t=tanh(E[x_t]); u_t=tanh(l_t + sum_{j=1..3,t>=j}
 tanh(w_j)⊙l_(t-j)); q_t=normalize(W_k u_t+b_k).
Native k_write_t=q_(t-1); P_t=alpha_t(I-beta_t*k_write*k_write^T),
Q_t=beta_t*v_t*k_write^T; M_t=M_(t-1)P_t+Q_t on admission, otherwise identity.
r_t=M_t q_t. First native M0=0. Native alpha<=199/200 and beta∈[0,1],
||k||<=1,||v||<=sqrt8 give conditional exact-real norm envelope
||M_next||<=rho||M||+sqrt8 and invariant||M||<=200sqrt8 from zero.
This loose envelope does not prove useful retention, recall, gradient survival,
full-model boundedness or IEEE arithmetic safety. Aligned bank gates unchanged.
Write is not multiplied by1-alpha; erase/write share the original scalar beta.
This is NOT GatedDeltaNet2 channelwise editing, new complex discretization or
sparse/monosemantic routing. Published primitives are reusable prior art;
existing model already has scans, rotations and repaired weighted pooling.

Independent PNDS Python formulates the exact changed input-to-key and serial
native update; Lean defines typed key/matrix/update/read and proves chronology,
nonwrite identity, conditional envelopes and an interior-gate oblique-erase
amplification counterexample. Numerical correspondence uses actual source
embedding/projection/gate values inFP64 and1e-12 tolerance. Parent whole-graph
formulation/refinement, autograd/IEEE and source-time carried-state proofs are
OPEN; six local kernel claims must not be promoted to end-to-end verification.
A dedicated reciprocal study record pins source; global historical17-entry
math archive remains partial. No mathematical completion or training exemption.

Actual source diagnostics use14 authenticated original FIT streams, first64
bytes per stream, fixed seed26026, future changes from32 onward. Prefix keys and
predictions invariant; shifted-key identity and native norm checked. Candidate
random initialization and synthetic suffix edits are mechanism controls, not
trained scores or real alternative outcomes. DEV NLL remains NOT_RUN.

Separate original-trained evidence: preserve five original LANGTRAIN025 model,
optimizer and RNG files from pinned artifact12846836297, plus exact d25d61a8...
receipt. A bot commits ONLY these public-source newly trained originals onto
this draft branch; separate read-only checkout then replays original source/
4096-case scores. Original private128-byte archives are never accessed.
Git persistence and same-owner CI replay are internal, not independent reward/
safety qualification or checkpoint promotion. Retain the0/5 P3 failure.
