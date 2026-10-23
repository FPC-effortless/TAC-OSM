# ME-060 native predictive dynamics

Inputs at an observed transition are x_previous, executed a_previous and the
newly arrived x_current. Innovation is tanh((x_current - x_previous -
servo(x_previous,a_previous))/delta_scale). Zero physical innovation is zero;
there is no bias-only learned physical-feedback update. The encoder also sees
current x, executed a and known control authority.

Native fast f and slow s each have 32 coordinates. Fast updates are convex
interpolation with a tanh proposal; slow uses a residual proposal and at most
one-quarter of the learned interpolation gate. From initial coordinates in
[-1,1], both stay in that interval. This bounds coordinates, not the Jacobian
or the useful information lifetime.

Associative M has 8x8 entries. For unit key k and bounded value v:

    retained = rho * M
    M_star = retained + beta * (v - retained @ k) outer k
    M_next = M_star / max(1, ||M_star||_F)

rho is learned in [.95,.9999], beta in [0,1]. The final Frobenius projection
guarantees ||M_next||_F <=1. A forecast query depends on known physical context,
action-prefix displacement and horizon. Reading M @ query leaves M unchanged.

The fixed FIT servo integrates supplied action prefixes, yielding b_h. A shared
SwiGLU decoder sees current state, b_h-x, h, fast/slow and associative read and
emits three tanh residual experts r_(h,k), four softmax weights pi and a positive
Laplace scale. The zero residual is the fourth mixture component:

    mean_h = x + (1-pi_(h,0))*(b_h-x)
               + h*delta_scale*sum_(k=1..3) pi_(h,k)*r_(h,k)

Thus |learned_residual_h| <= h*delta_scale coordinatewise. Persistence and the
servo are direct prediction paths, and every neural expert can train from
factual future errors. Means are not recursively treated as new observations;
future target data cannot update the prediction-time state. Later action
prefixes cannot influence earlier horizon predictions.

scale_h = (softplus(u_h)+.001)*max(h*delta_scale,.0001).
Positive scale is a numerical guarantee. Calibration and useful uncertainty
are empirical questions. No Gaussian, causal-control or thought claim follows
from these equations. Both comparator architectures use the same decoder
family, objectives, priors and authentic source information.
