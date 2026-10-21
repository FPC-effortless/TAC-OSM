# Next native intervention — proposal only, not fitted in ME027

ME027 finds fast learning from a nonlinear observable-feature MLP, followed by
worse performance under excessive fixed CPU exposure. Native state reliance
is inconsistent. This points to testing a small nonlinear observable readout
before adding another recurrent gate. It does not prove that a readout change
will work, that hidden state is useless, or that a larger MLP is optimally tuned.

The current recurrent readout is additive and affine in the stored duration
feature d=log1p(raw_video_duration)/12, followed by FIT standardization. Its
second logit derivative with respect to this stored feature d is zero. It
already has log-affine curvature with respect to raw duration; the claim is
not that raw-duration curvature is absent.
It cannot learn additional curvature in the stored duration feature through the
historical state, because the offered duration is not yet in that causal state.
The feature MLP can represent this curvature and observable-feature interactions.
This is a representational difference, not an empirically established causal
explanation of its observed learning curves.

A falsifiable next native candidate retains all fast/slow equations and uses

```
logit(p) = logit(prior32) + w_h^T phi(h) + v^T tanh(A x+a) + b
```

with the exact common 26-vector and 12 observable hidden units. Its head has
64 + 26*12 + 12 + 12 + 1 = 401 trainables versus the present 91. Total parameters
would be 135377, only 310 more (0.230%). There is no new recurrent state, learned
memory gate or SSM conversion. Zero final v, w_h and b recover the initial
analytic prior. Initialization of A and every optimizer group must be frozen
in that future registration, not tuned after looking at development scores.

Writing z=A x+a and d for the stored, pre-standardization log-duration scalar,
and holding other inputs fixed,

```
d logit(p)/dd = sum_j v_j sech^2(z_j) A_jd / duration_scale
d^2 logit(p)/dd^2 = -2 sum_j v_j tanh(z_j) sech^2(z_j)
                          (A_jd/duration_scale)^2
```

The second derivative can be nonzero. Native recurrent bounds are unchanged;
these derivatives do not imply bounded whole-model sensitivity since A and
embeddings remain unconstrained. Mean-feature replacement still preserves the
FIT-average linear latent logit. Its OOD interpretation remains limited.

The next primary should isolate this readout intervention against the unchanged
native equations with matched source, exposure, tuning privileges and measured
cost. Both resource lanes and an identical validation-selection policy need
registration. Do not force a rapid learner into thousands of unnecessary
updates and market the resulting score gap as useful state information.
Any early-stopping or learning-rate schedule must apply under equally declared
privileges; final qualification still needs genuinely untouched observations.
No future study is fitted or pre-approved by this proposal. All ME027 scores
and the failed state gate remain unchanged.
