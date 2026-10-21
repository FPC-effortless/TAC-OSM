# ME044: multiscale predictive innovation architecture

The engineering aim is a stable, effective additional predictive pathway. It combines established orthogonal forecast bases, residual objectives and native TAC dynamics; this document makes no literature-wide novelty claim.

## Shared observed information

X is a complete96-observation multivariate window. Its causal window mean mu and scale sigma produce normalized channel history Z and shared across-channel history. The original AR plus context forecaster gives F_cheap(X). Its128-dimensional latent l and16 physical/window statistics c are available to all matched systems, as are the same twelve observed eight-sample patches. The periodic prior is24hours, expressed as144 ten-minute samples or24 hourly samples from public source cadence; no future calendar/outcome values are used.

## A fixed orthogonal horizon representation

Construct eight linearly interpolated hat functions over horizon h and sine/cosine functions at harmonics1,2,4,8 of the daily period. Thin QR in float64 gives Q; canonicalize column signs using the diagonal of R. Store B=sqrt(h)*Q^T as a fixed float32 checkpoint buffer. Thus, to rounding precision:

    B B^T / h = I_16.

For an observed training residual r=(Y-F_cheap(X))/sigma, its least-squares coefficients are a_star=r B^T/h. In exact arithmetic:

    ||r-a B||^2/h = ||r||^2/h - ||a_star||^2 + ||a-a_star||^2.

This provides a direct predictive target in a compact, multiscale span. It does not prove that this span contains all useful temporal information, nor that a native recurrence learns those coefficients better than a simpler encoder. The spline and seasonal components span both coarse shapes and recurrent patterns; orthogonalization changes coordinates without changing their joint function space.

## Native state and bounded decoder

Initialize fast32 and slow32 from .1*tanh of cheap latent coordinates. Initialize the8x8 associative matrix to zero. The original ME038 nonlinear step processes six observed tokens u_j=.25*p_(2j)+.75*p_(2j+1). Its normalized-key error-corrected write remains:

    M_j = rho*M_(j-1) + beta*(v-rho*M_(j-1)*k)*k^T.

Recall modulates the original fast/slow dynamics; the final64-vector is RMS-conditioned. Concatenate state, detached l, and detached c into208 features. A208->16->16 MLP predicts a. A separate208->1 sigmoid predicts trust q. The new physical-scale innovation is:

    delta = sigma * .5*q*tanh((a B)/.5)
    Yhat = F_cheap(X) + delta.

The per-coordinate bound is |delta|<=.5*sigma. It bounds perturbation, not prediction error or physical safety. The new branch has no forced horizon720 decay. Both its output coefficients and correction are zero at initialization. Inference always executes this branch; it does not retain ME043's hard compute router. State is reset per window; no constant-time stream update is claimed.

Matched context and GRU systems replace only the state encoder with active same-capacity alternatives and receive the same basis, residual objective, detached context, readout and observed history. They are competitive complete forecasters, not a requirement to attribute every individual mechanism before development continues.

## Decoupled gradients, joint predictive training

On every update, train the original cheap predictor on its original MSE, AR-anchor and correction losses. Every fourth update, also train the new expert on the actual full-forecast residual MSE, projected coefficient loss and bounded innovation regularization. Stop gradients through F_cheap and l in that expert objective. Clip cheap and expert gradients separately.

Therefore the expert loss has zero derivative with respect to cheap parameters. With identical initialized cheap coefficients, same batches, optimizer settings and update counts, cheap-path optimization follows the original context predictor exactly. This property is verified numerically and checked against actual fixed-exposure model checkpoints. It does not guarantee that adding the innovation improves held-out forecasts. The complete system trains both pathways, but does not backpropagate the full expert loss through the cheap encoder.

Cheap-path training runs every update, while native dynamics/readout learn every fourth update. Expert optimizer clocks, actual CPU, full inference latency and all negative outcomes are measured. Fixed exposure and fixed CPU answer different engineering questions; no fully matched superiority claim follows from either alone.
