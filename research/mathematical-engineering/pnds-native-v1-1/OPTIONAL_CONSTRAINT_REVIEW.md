# TAC-OSM v1.2 constrained-decision mathematics and native source verification

This is a **prospective untrained optional mathematical adapter** for civilian planar single-integrator navigation; it is not an existing MILP-based TAC-OSM model, a defense planner, or a certified controller. Mathematical authority: https://github.com/FPC-effortless/pnds-lean/pull/7. Source authority: https://github.com/FPC-effortless/TAC-OSM/pull/389. The unchanged native TAC-OSM remains the predictive fast/slow/associative observational state model.

## Conditional planar safety mathematics

For position p, obstacle o, r=p-o, radius R, h=||r||²-R², measured obstacle velocity w_hat and a *genuinely supported* bound ||w_true-w_hat||<=epsilon, choose a proposed velocity u satisfying

  2r·u >= 2r·w_hat - gamma*h + 2||r||*epsilon.

If h>=0, gamma>0, gamma*dt<=1, and both robot and obstacle obey exactly constant velocities throughout this interval, Cauchy-Schwarz and exact planar translation give:

  h(r+s(u-w_true)) = (1-gamma*s)h(r)
                +s*(2r·(u-w_true)+gamma*h(r))
                +s²||u-w_true||² >= 0

for all s in [0,dt]. The corresponding conditional sampled and throughout-interval planar implications and scalar half-line QP/KKT facts are proved in PNDS/ConstraintAdapter.lean. None of this is a proof for accelerated systems, delays, unmodeled contacts, inaccurate bounds or deployed autonomy.

For n=2r and b=2r·w_hat-gamma*h+2||r||*epsilon, the *unbounded* strictly convex Euclidean halfspace projection is

  u_star = u_des + max(0,b-n·u_des)n/||n||².

The optional src/tac_osm/variants/planar_constraint_adapter.py implements this formula in differentiable Torch with explicit failure checks. Gradients are mathematically piecewise smooth away from active-set boundaries; this is not general OptNet implicit-KKT differentiation and no learned TAC parameters were retrained. In particular the model **does not implement actuator bounds**, so its output is never treated as a deployable physically admissible command.

The PNDS bounded Python reference additionally handles a componentwise 2D actuator box. It enumerates the convex region's candidate edges and detects infeasibility. On solver or feasibility failure, there is *no* automatic zero-action safety guarantee: at contact, a stationary robot may still be struck by an approaching moving obstacle. Fail closed with a separate certified supervisor for actual control; the research implementation is not that supervisor.

## Preserve the identity and evidence of the original model

The provided description of Tactical Optimization and Constraint-Aware Operational State Models is a different system from the actual Task–Action–Counterfactual Observational State Model. Diffusion sampling, hypergraph Laplacians, Riemannian metrics, Mamba backbones and Transformer planners are *separate unverified proposals*, not silently imported into core TAC-OSM or treated as mathematical guarantees. A finite energy penalty does not guarantee a feasible trajectory, a spatial cost map is not necessarily a positive-definite metric, and a nonlinear native recurrence does not inherit the affine SSM parallel-scan theorem.

The genuine native graph remains event→encoder→old associative read→fast→slow→retention/write gates→role-masked Frobenius projection→final read/decoder. The accompanying original-source replay script independently reconstructs its full forward including all intermediate memory/gates, checks every learned-parameter gradient and checks nonlinear state JVPs against centered finite differences. This is *mechanism verification*, not original trained checkpoint performance or a Lean whole-model proof.

The exact file SHA-pinned cross-repository optional mathematical component index is separately mirrored in PNDS and TAC-OSM. It is **not** added to the historical 17 neural-variant register because it has not been fitted and is not a new native predictive architecture. Original failures, training evidence and real data remain unchanged.

Sources: OptNet https://proceedings.mlr.press/v70/amos17a.html ; CBF-QP https://doi.org/10.1109/TAC.2016.2638961 ; HGNN https://doi.org/10.1609/aaai.v33i01.33013558 ; mathematically scoped 2026 diffusion safety https://proceedings.mlr.press/v306/hoshino26a.html .
