# ME105 explicit graph formulation completion

TAC original training source: `573f81cb9f2f421d141ddb722525ecce25972edc`.
PNDS source-linked formulation: `52d65e76d871658a5e45ccc7255bdf165d8e66dc`.

`PNDS/SensorGraph105.lean` explicitly defines all learned affine tensors, input
masking and encoder, station lookup, source/evidence chronological state,
fast/slow proposals, native rank-one memory, pre-event decoder read, inclusive
belief read, reconstruction and variance, source-gap protection and bounded
median innovation. New-history and carried-state entry points are typed.
CDF/quantile/NLL equations expose a named NormalLaw boundary whose Gaussian
identities remain OPEN; no arbitrary function is proved normal. The trainable
shape arithmetic theorem is100,349, matching the live model.

Together with ProtectedSensor there are26 limited theorems; see the actual
PNDS axiom transcript and ME105_VERIFICATION.json for count/status. Modules
compile; audited axioms are only propext,Classical.choice,Quot.sound. This closes
the earlier opaque-parent *definition* gap, not whole-network verification.
Python source/serial and fitted numerical checks remain separate. Full source
Lean refinement, exact gradient and IEEE correspondence, Gaussian measure
proofs and empirical coverage are OPEN. Six-family independent capability
qualification is also OPEN.
