# ME034 mathematical specification and limits

Let u_t=[E_R(item_t),metadata_t], with E_R learned solely on representation
users R and immutable during downstream S training. Frozen coordinates do not
make the distribution of observations stationary or freeze the learned encoder.

The native equations are unchanged:

```
q = tanh(A u + T f + bq)
g = sigmoid(Gu u + Gf f + bg)
f_next = (1-g)*f + g*q
v = sigmoid(Vf f_next + Vs s + bv)
r = tanh(Sf f_next + Ss s + bs)
s_next = (1-v)*s + v*r
```

Fast and slow each have32 coordinates. Initial f=s=0; initial gate biases
log(1/7) and log(1/23) imply1/8 and1/24 before other logits. These are initial
gate biases, not proven effective memory half-lives. Convex gated updates bound
state coordinates within[-1,1], but do not guarantee useful long-range gradients.

Writing D(a) for a diagonal matrix, the exact fast-state Jacobian is

```
Jf = I-D(g) + D(g) D(1-q²) T + D(q-f) D(g*(1-g)) Gf.
B  = D(v) D(1-r²) Sf + D(r-s) D(v*(1-v)) Vf.
C  = I-D(v) + D(v) D(1-r²) Ss + D(r-s) D(v*(1-v)) Vs.
J_coupled = [[Jf, 0], [B Jf, C]].
```

Freezing E_R removes its downstream optimization drift, not the gate derivatives
or coupling term. It does not establish spectral contraction, a cause of earlier
failures, or a remedy for all optimization problems. The intervention is tested
through actual future-outcome loss.

At each observed prefix k=32/64/96/128, RMS-condition each32-vector separately:
R(x)=x/sqrt(mean(x²)+0.01). Its Jacobian is
I/a - xx'/[32 a³], a=sqrt(mean(x²)+0.01). Its operator norm is at most10;
the concatenated normalized64-vector has Euclidean norm at most8. RMS remains
a learned-readout conditioning choice, not an expansion of stored information.

The centered readout remains

```
logit = prior32_logit + common_R(C)
        + w'h + v'[tanh(Bh+AC+a)-tanh(AC+a)].
```

The correction is identically zero at h=0; output/direct weights initialize to
zero. Memory gradients therefore start only after those output coefficients
change. This is checked rather than called a gradient failure at step1. A fixed
nonzero h can still implement a context function, so a useful correction alone
does not establish newly extracted historical information.

For an extra representation h=g(C), I(Y;h|C)=0 exactly in the ideal probabilistic
model. It can nevertheless improve finite fitting, conditioning or inductive
bias. A recurrent state h=f(H) of raw history H need not be determined by the
compressed common vector C; I(Y;f(H)|C) need not vanish. Finite-model held-out
score differences are not estimates or proofs of conditional mutual information.

The GRU control uses trainable tanh(Linear11->24), standard GRUCell24->36,
and fixed64x36 Q with Q'Q=I. Its36D recurrent state is mapped injectively into
the existing64D RMS/readout interface. Q is a buffer with no learned capacity;
all6984 learned body coefficients affect the updates. Native body has6976.
The common correction adds1260 active coefficients to both:8244 versus8236.
Their B8 measured forward/backward operator subtotals are42,484,007 versus
42,652,991 (0.3962% gap). This excludes unsupported operator work and Adam,
so full FLOP matching is not established by these figures.

EWMA feature k has numerator sum_i gamma^(k-1-i)y_i and denominator
sum_i gamma^(k-1-i), gamma=2^(-1/half_life), over observed responses only.
No current response enters the feature. The tree and EWMA MLP are deliberately
unmatched practical controls, not capacity-equivalent architecture challengers.

The final primary requires a Brier advantage larger than0.001 and positive
AUC effect against common-head adaptation. Two97.5% user-clustered bootstrap
intervals use the Bonferroni union bound for at least95% joint nominal coverage,
conditional on these nine fitted models per arm. Bootstrap intervals remain
approximate finite-sample intervals; they do not cover training-seed uncertainty
or hidden undocumented prior data access. Failure closes this recipe's positive
memory claim, not the mathematical possibility of recurrence on other tasks.
