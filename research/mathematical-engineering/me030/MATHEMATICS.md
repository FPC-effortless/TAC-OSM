# ME-030 mathematical engineering

Let h be the existing 64-coordinate split-RMS fast/slow state and x the 26
common causal features, including current item embedding and duration.

The ME028/029 head has logit

\[
 \ell_0(h,x)=\operatorname{logit}(p_{32})+w^\top h+
 v^\top\tanh(Ax+a)+b.
\]

For fixed prior and representation, its mixed logit derivative with respect to
h and x is zero. This restricts conditional readout expressiveness: state
sensitivity cannot depend on the offered item. The final sigmoid still introduces
probability interactions, and shared encoders are trained jointly. This is not a
statistical independence claim or evidence that state lacks information.

ME030 uses

\[
 \ell_1(h,x)=\operatorname{logit}(p_{32})+w^\top h+
 \sum_{k=1}^{12}v_k\tanh(B_kh+A_kx+a_k)+b.
\]

Writing \(z_k=B_kh+A_kx+a_k\),

\[
 \frac{\partial\ell_1}{\partial h_i}=w_i+
 \sum_k v_k\operatorname{sech}^2(z_k)B_{ki},\qquad
 \frac{\partial^2\ell_1}{\partial h_i\partial x_j}=
 -2\sum_k v_k\tanh(z_k)\operatorname{sech}^2(z_k)B_{ki}A_{kj}.
\]

The tests compare the mixed derivative to double-precision autograd. The additive
head is the subspace B=0; greater expressiveness does not guarantee better
finite-budget optimization. Extra 64x12=768 weights all participate in scoring.
Native totals: 135377 -> 136145 (+0.567%); pooled total 136681. Maximum gap is
1304/135377=0.9632%. Nonembedding counts 7377/8145/8681 disclose allocation changes.
Joint native and pooled heads have equal expressiveness conditional on the latent;
their encoder families and optimization costs differ.

For fixed finite weights,
\(\|\nabla_h\ell_1\|_2\leq\|w\|_2+\|B\|_2\|v\|_2\).
This is a parameter-dependent readout bound. B is not spectrally constrained;
there is no uniform training-time or coupled recurrent Jacobian guarantee.
Original convex gate/tanh coordinate bounds and RMS epsilon .01 stay unchanged.
No associative scan is claimed for nonlinear native recurrence.

Native recurrent state remains 64 float32 coordinates, 256 bytes, plus separate
prior/observable bookkeeping. Joint readout adds a 12x64 matrix-vector product
per forecast without increasing recurrent-state size or per-event recurrence
arithmetic. These are full 128-event-window fits, not continuous-stream qualification.
The pooled LATENT is order-free; its shared observable EWMAs/prior retain order.

For Bayes-optimal binary squared-loss predictors,
\(R^*(C)-R^*(C,H)=E[(E[Y|C,H]-E[Y|C])^2]\).
Finite fitted families add approximation and optimization errors. Their score
difference cannot identify that ideal quantity or conditional mutual information.
FIT-mean state substitution also changes the input distribution: diagnostic only.
