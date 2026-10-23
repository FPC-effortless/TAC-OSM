# TAC-OSM: a reusable differentiable state program

Status: concrete notation for the existing trained ME080 architecture and an
execution-only implementation. **Not a new trained intelligence result.** The
native model has 99,816 learned parameters, fast/slow states and an 8×8 matrix.
Original ME080 models, source, scores, gates and numerical exceptions are immutable.

## 1. One mathematical object for the model and its execution

Write the predictive program as

\[
\boxed{\quad
\mathcal P_\theta = D_\theta\circ
\operatorname{Scan}(\Phi_\theta)\circ E_\theta\circ\Psi,
\qquad
(\widehat P_t,S_t)=\mathcal T_\theta(o_t,u_t,S_{t-1}).
\quad}
\]

Here \(o_t\in\mathbb R^{14}\) is the verified current observation and \(u_t\) is
an already-known command, not a future chosen action. \(\Psi\) constructs causal
features, \(E\) encodes them, \(\Phi\) advances native state, and \(D\) predicts
coordinate distributions at horizons \(h\in\{1,4,16\}\).
`Scan` means an ordinary sequential fold here. It does **not** assert an efficient
associative tensor scan for TAC's nonlinear transition.
The displayed composition uses tuple-carrying stages: the encoder stage keeps
\(\phi_t\) alongside \(e_t\), and the fold emits per-event readout values as
well as the terminal cache. More explicitly,

\[
\phi_t=\Psi(o_t,u_t,S_{t-1}),\quad e_t=E_\theta(\phi_t),\quad
(S_t,z_t)=\Phi_\theta(S_{t-1},e_t,\phi_t),\quad
\widehat P_{t,h}=D_{\theta,h}(\phi_t,e_t,z_t).
\]

The full transition also copies the verified physical observation/velocity into
the new cache. The readout \(z_t\) contains the **pre-write** matrix read; it is
emitted during transition, not assumed reconstructible from updated \(M_t\).

State and identity are explicit:

\[
S_t=(f_t,s_t,M_t,o_t,v_t,b_t),\qquad
\nu=(\operatorname{hash}(\theta),\operatorname{hash}(\text{preprocessing})),
\qquad \kappa=(\nu,\text{entity},\text{sequence}).
\]

Fast and slow vectors each have 32 coordinates; \(M_t\in\mathbb R^{8\times8}\).
The last observation, observed velocity and first-observation flag supply causal
physical context. This is 157 float32 values, or 628 bytes of **state payload**.
The inference handler additionally stores the latest forecast and retry inputs:
its full tensor payload is larger (1,592 bytes/entity for this implementation),
plus Python/tensor/container overhead. Model weights and activations are separate.

## 2. Shared causal feature graph

Let \(\mu_x,\sigma_x,\mu_u,\sigma_u,d\) denote frozen FIT-only statistics.
All observations below are already standardized using those fixed statistics.
For the initial observation, velocity and acceleration are zero; otherwise

\[
v_t=o_t-o_{t-1},\quad a_t=v_t-v_{t-1},\quad
c_t=\frac{u_t\odot\sigma_u+\mu_u-\mu_x}{\sigma_x}-o_t,
\]
\[
\phi_t=\Psi(o_t,u_t,S_{t-1})=
[o_t;c_t;\operatorname{asinh}(v_t/d);
\operatorname{asinh}(a_t/d);1/15]\in\mathbb R^{57},
\qquad e_t=E_\theta(\phi_t)\in\mathbb R^{64}.
\]

The nominal 15 Hz spacing is not a general irregular-time model. No future
observation, takeover label or future command enters these features.

Express shared computation as explicit let-bindings:

\[
\textbf{let }(k_t^0,w_t,\rho_t^0,\beta_t^0)=A_\theta e_t+b_A,
\quad p_t=W_o e_t+b_f,
\quad k_t=\frac{k_t^0}{\sqrt{\|k_t^0\|^2+10^{-6}}}.
\]

Encoder, address and observed fast-state projection can run over the entire
causal chunk in batch. Their values are shared, not recomputed by every consumer.
This is explicit program restructuring; PyTorch/JAX tracing does not automatically
guarantee common-subexpression elimination or fusion of arbitrary Python code.

## 3. Native fast, slow and associative dynamics

Read the previous matrix before writing the current observation:

\[
r_t=M_{t-1}k_t,
\quad [c_t^f;g_t^f]=p_t+W_{fs}[f_{t-1};s_{t-1}],
\]
\[
f_t=(1-\sigma(g_t^f))\odot f_{t-1}
       +\sigma(g_t^f)\odot\tanh(c_t^f),
\]
\[
[i_t;g_t^s]=W_s[f_t-s_{t-1};s_{t-1}]+b_s,
\quad
s_t=(1-\sigma(g_t^s))\odot s_{t-1}
       +\sigma(g_t^s)\odot\tanh(s_{t-1}+\tanh(i_t)).
\]

For associative storage,

\[
\alpha_t=\sigma(\rho_t^0),\quad\beta_t=\sigma(\beta_t^0),\quad
\bar M_t=\alpha_tM_{t-1},
\]
\[
M_t=\Pi_4\left(\bar M_t+
\beta_t(\tanh(w_t)-\bar M_tk_t)k_t^\top\right),
\quad \Pi_4(Z)=\frac{Z}{\max(1,\|Z\|_F/4)}.
\]

Thus zero-start fast/slow coordinates stay within [−1,1] and the matrix norm
is at most four. These are stability constraints, not guarantees of useful recall.
The learned recurrent representation is \(z_t=W_z[f_t;s_t;r_t]+b_z\).
Gated delta updates are established prior art, not a TAC novelty claim.

## 4. Predictive distribution and differentiable objective

The decoder uses a SwiGLU map of \([e_t;z_t;\phi_{t,14:56}]\in\mathbb R^{170}\).
For each coordinate and horizon it predicts

\[
p_\theta(\delta_{t,h,j}\mid S_t,e_t)
=(1-\pi_{t,h,j})\operatorname{Laplace}(0,b^0_{t,h,j})+
\pi_{t,h,j}\operatorname{Laplace}(m_{t,h,j},b^1_{t,h,j}).
\]

Scales are positive; the learned motion centre is bounded and combines observed
velocity, a FIT-derived servo term and a learned residual. Medians and 5/95%
quantiles invert the **same** CDF. A separate head predicts the factual event
that mean physical displacement exceeds 0.025. Its probability is not identical
to latent mixture membership. The marginals do not specify joint trajectory
or cross-coordinate dependence.

For genuine available future targets, the existing objective is

\[
\mathcal L(\theta)=\sum_i\omega_i
\left[\operatorname{mean}_{t,h,j\,\mathrm{valid}}
(-\log p_\theta(y_{i,t+h,j}\mid H_{i,t})+\log\sigma_{x,j})
+0.2\operatorname{mean}_{t,h\,\mathrm{valid}}
\operatorname{BCE}(q_{i,t,h},\mathrm{motion}_{i,t,h})\right].
\]

The notation includes the physical-unit Jacobian and registered sampling
weights. The execution change preserves this objective and its derivatives;
it does not change loss weights, data exposure or update any learned parameter.

## 5. Scheduling the tensor DAG

The decoder has no dependence on other decoded time positions. Therefore

\[
\operatorname{Stack}_t D_\theta(e_t,z_t,\phi_t)
=\operatorname{Unflatten}\left[
D_\theta(\operatorname{Flatten}_{B,T}(e,z,\phi))\right].
\]

This is equality in real arithmetic, with floating-point agreement tested
separately. The new implementation replaces T small decoder calls with one
larger batched call. Arithmetic remains comparable; dispatch/layout efficiency
can improve. If only the last forecast is requested, compute
\(D_\theta(e_T,z_T,\phi_T)\) once and skip the other T−1 decoder evaluations.
The encoder and native recurrence still consume the entire observed prefix.

For uncertainty, share float64 conversion, component scales, weights and bounds
across the three requested quantiles. Forty bisection iterations process all
three probability levels together, instead of three separate 40-iteration
Python loops. This is vectorized tensor execution, **not** a fused GPU kernel.

The all-prefix path remains differentiable. Float64 reference tests compare
all parameter gradients before any optimizer exists. Inference uses `no_grad`;
its detached cache must never substitute for a training computation graph.

## 6. Effect notation and the concrete runtime handler

Separate the pure transition from ordered state effects:

\[
\operatorname{run}_{\mathcal H_\nu}
[\operatorname{advance}(e,n,o,u)]
=\begin{cases}
\operatorname{clone}(\widehat P_{e,n}),&\text{exact latest-event retry},\\
\mathcal T_\theta(o,u,S_{e,n-1}),&n=\operatorname{last}(e)+1,\\
\operatorname{reject},&\text{conflicting retry, gap or stale version}.
\end{cases}
\]

`EntityInferenceCache` implements this idea with ordinary Python/PyTorch,
not an algebraic-effects language or a claim of category-theory innovation.
It is caller-scoped, single-threaded, bounded by an explicit entity capacity,
and returns defensive copies. Entity state cannot bleed through a global cache.
The first sequence index is zero. Historical corrections require explicit reset
and replay of the verified prefix; capacity exhaustion never silently evicts state.

The handler checks parameter/buffer identity, dtype, device and PyTorch mutation
versions and rejects normal model changes. Recreate it with new model and
preprocessing revisions after updates. Unsupported `.data` mutation, monkey
patching or concurrent unsynchronized calls are outside this research contract.
No model cache can infer whether an external source silently revised history;
the caller must supply trustworthy sequence/version metadata.

## 7. Complexity and limits

For fixed state/model dimension, incremental TAC updates cost O(1) per event
and O(T) over T events, with bounded state payload. Replaying the entire growing
prefix after each event instead costs O(T²) total. This does not mean recurrent
state preserves all historical information or that an eight-frame-trained
model is accurate on indefinitely carried state.

KV caching also avoids recomputing old key/value projections, but ordinary dense
attention still reads a growing cache: attention work for one new token is O(t)
and total generation attention is O(T²), not universally linear or exponential
without caching. SSM/linear-scan properties do not automatically transfer to
nonlinear TAC. Function composition is associative abstractly, but representing
those composed nonlinear functions compactly is the hard part.

## Prior art and research boundary

Bishop (1994), [Mixture Density Networks](https://publications.aston.ac.uk/id/eprint/373/);
[Gated Delta Networks](https://arxiv.org/abs/2412.06464);
[JAX composable transformations](https://github.com/jax-ml/jax);
[FlashAttention](https://arxiv.org/abs/2205.14135) for IO-aware execution;
[Mamba-2/SSD](https://proceedings.mlr.press/v235/dao24a.html) for structured scan
rigor; and [R2D2](https://willdabney.com/publication/r2d2/) for recurrent state
staleness. These motivate engineering choices; no papers' performance results
are transferred to TAC. No Triton/MLIR/compiler fusion, new fit, new prediction
gain, independent-source success, policy or customer benefit is established here.

The next **predictive capability** work still needs a separate prospective
training registration addressing ME080's real moving-state/four-frame and quiet
precision failures. Faster execution alone is not greater intelligence.
