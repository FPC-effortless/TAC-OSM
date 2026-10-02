# TACOSM-PLM-INTEGRATED-E2E-004

E2E-003 showed that correct structural addressing did not change held-out
capability. E2E-004 therefore fixes addressing and tests the representation to
execution-content interface.

The typed_learned arm stores the supervised 12-bit representation output and
executes the requested Boolean operator directly over those typed bits.

The typed_oracle arm uses the true observed 12-bit payload and is a downstream
execution ceiling, not a learned result.

A positive typed_learned result is limited to this supervised synthetic
mechanism and would not establish semantic abstraction, learned semantic
addressing, natural multimodal competence, or scaling.
