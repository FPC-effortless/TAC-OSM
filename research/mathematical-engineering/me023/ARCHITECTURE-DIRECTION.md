# Mathematical architecture development

The long-term objective is a learned persistent predictive system with useful
state, explicit relevance and executable decisions. Progress is measured by
what each mechanism adds under real data and resource constraints.

The current engineering sequence is:

1. **Condition the numerical interface.** ME022 established that the native
   model could learn far more effectively after input/readout conditioning.
   Its calibration regression and negative state-readout evidence remain.
2. **Make memory queryable and train it predictively.** ME023 implements a
   bounded action-conditioned bilinear readout and shared causal prefix losses.
   The code has a useful representational guarantee, explicit gradient bounds
   and constant-size inference state. Empirical usefulness is a separate gate.
3. **Develop state dynamics only against a demonstrated blocker.** Candidate
   research includes horizon-aware timescales, learned slow-to-fast feedback,
   and depth with normalized residual computation. Each needs its own exact
   coupled Jacobian, boundedness/conditioning analysis, activation accounting,
   ablations and real-data registration. These are proposals, not implemented
   capabilities. Strict contraction everywhere is not automatically desirable
   when distant predictive information must survive.
4. **Engineer execution and calibration with the learned model fixed.** Measure
   exact projection/kernel transformations, chunk behavior, latency, memory,
   approximation error and proper out-of-sample calibration. Nonlinear TAC
   cannot inherit an associative parallel scan merely because SSMs use one.
5. **Connect state and action predictions to the broader system.** CDL routing,
   CASM execution and verified consequences require their own source-specific
   evidence. Query-conditioned neural readout alone is not CDL teacher
   distillation, symbolic program discovery or reliable autonomous control.

SSMs motivate explicit state dynamics, conditioning and execution analysis.
LLMs motivate shared representations and dense causal prediction objectives.
Matching those models' scale, breadth or empirical capability would require
much more evidence than a small recurrent research experiment. This roadmap
does not rename TAC as either architecture or claim that capability has arrived.
