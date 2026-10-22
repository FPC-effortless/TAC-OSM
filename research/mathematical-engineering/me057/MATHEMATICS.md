# ME-057 fused predictive anchor

ME-057 retains TAC's 32-dimensional fast state, 32-dimensional slow state and
8-dimensional relevance trace.  At a history boundary it constructs

\[
z=[u_t;f_t;s_{t,1:25};m_t\odot q_t]\in\mathbb R^{81}
\]

and evaluates one joint projection

\[
[b_t;p_t]=W_a z+c_a\in\mathbb R^{17+73}.
\]

The first 17 coordinates produce the relevance write address, residual and
strength.  The remaining 73 produce the trace value/write strength and the
slow-state innovation/gate.  One map replaces the two serial ME-056 maps.

The terminal persistent belief is also paired with the final 32 authentic
history transitions.  The anchor predicts bounded physical delta, action
error, takeover onset and transition statistics.  This makes dense training a
prediction-preserving retention objective without storing every intermediate
belief. The same objective directly trains the physical-feedback key, value
and joint correction maps against bounded physical deltas and takeover onset.
The observation encoder remains trained end to end by the multistep
rollout; its input to this auxiliary objective is detached to avoid counting a
second redundant encoder-gradient path. Initial physical state is injected
deterministically into the fast coordinates; the former unreachable learned
initializer weights expand the active predictive anchor instead.

The persistent memory and slow-state updates remain bounded convex updates.
No mechanism here establishes policy value, causal intervention effects or
autonomous control.
