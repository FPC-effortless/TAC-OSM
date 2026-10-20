# ME024 — bidirectional timescale dynamics and conditioned state readout

ME023 supports causal prefix training as a development direction. Its
action-query advantage failed; that512-parameter head is not carried into this
experiment. Every current cell has the original135,050trainables and a linear
readout after its declared state feature map. No architecture rival is refit.

## Coupling fast innovation to slow context

The original fast update cannot read slow state. Test a single tied feedback:

```
q = tanh(E u + W(z + alpha s)), alpha in {0, 0.25}
g = sigmoid(G[u;z])
z' = (1-g) z + g q
p = tanh(S[z';s]), v = sigmoid(V[z';s])
s' = (1-v) s + v p
```

There are no new learned matrices or added states. Sharing W ties the feedback
coordinates to the established native32-dimensional latent coordinates. This
is a hypothesis, not proof those coordinates already align optimally.
Starting in[-1,1], both states remain coordinatewise in[-1,1] by convexity of
their gated updates. Bounded state does not imply contraction or good memory.

Let independent local partials be A=dz'/dz, D=dz'/ds, B=ds'/dz' and C=ds'/ds
holding updated z' fixed. The **full** state Jacobian is

```
J = [[A, D], [B A, C + B D]]
D = diag(g) diag(1-q^2) alpha W
A = diag(1-g) + diag(g) diag(1-q^2) W
    + diag(q-z) diag(g(1-g)) G_z
B = diag(v) diag(1-p^2) S_z + diag(p-s) diag(v(1-v)) V_z
C = diag(1-v) + diag(v) diag(1-p^2) S_s
    + diag(p-s) diag(v(1-v)) V_s
```

Factorization `J=[[I,0],[B,I]] [[A,D],[0,C]]` gives `det J=det A det C` at
the evaluated partials. A, B and C themselves depend on the changed trajectory;
this is **not** a claim that changing alpha preserves determinant, eigenvalues
or stability. For alpha0, D=0 recovers the original triangular audit. For
alpha.25, genuine slow-to-fast sensitivity must be included in every derivative
and chronological Jacobian product. Tests compare all4096entries to autograd.

## Split RMS conditioning

For each32-coordinate state block x, optionally read

```
phi(x) = x / r, r = sqrt(mean(x^2)+epsilon), epsilon=0.01
dphi/dx = I/r - x x^T/(32 r^3)
```

Tangential eigenvalues are1/r and the radial eigenvalue is epsilon/r^3.
Thus `||dphi/dx||2 <= 1/sqrt(epsilon)=10` and `||phi(x)||2 <= sqrt(32)`.
At zero the derivative is10I, finite. The block-diagonal fast/slow feature
map has the same derivative bound. It adds no learned gains or biases. This
conditions the readout, not the recurrent dynamics or a whole-model Lipschitz
constant. It may discard useful magnitude information; empirical benefit is
not assumed.

The readout remains linear in phi(h), normalized current duration and current
item embedding, plus the unchanged analytic causal prior32. FIT-only duration
normalization and shared causal prediction at depths32/64/96/128 inherit ME023.
All four cells have the same objective and supervised target exposure.

## A cleaner feature-replacement diagnostic

Replace varying phi(h) by its mean over that model's first1024FIT states,
computed after training, without changing action, duration, prior or intercept.
Because the final head is linear, this preserves its FIT-average latent logit
level. Averaging raw states before nonlinear RMS would not preserve that level
(Jensen's inequality); that is deliberately avoided. The intervention is still
OOD and does not identify causal memory benefit or out-of-sample utility.

## Execution and analysis

The incremental transition retains64float32state coordinates (256bytes).
The caller still maintains the prior32FIFO and entity ownership. Full-window
encoding materializes O(T) inputs; streaming avoids replay, not training BPTT.
No associative scan or linear-SSM conversion is claimed. Fixed feedback adds
only elementwise arithmetic to the native transition; RMS adds readout work.

Post-fit diagnostics use exactly8original FIT windows per model, copied to
float64, with the analytical full64x64Jacobian and actual chronological product
P_t=J_t...J_1 at depths8/32/64/96/128. Singular spectra are diagnostics, not a
proxy success gate. Local expansion, better gradient throughput, or a changed
product norm cannot on its own establish useful predictive memory.
