# COLLECT016 — connected credit is present; retrieval was not learned

Nine fixed-final 512-update fits tested full 384-byte BPTT, state carried with
128-byte detachment, and a separately trained current-value-only phase lane.
All three seeds of every arm failed the registered primary source-copy assay:
0% unrestricted first-byte, exact-four-byte, complete-map and absent-first-byte
accuracy. Restricted nine-character first-byte accuracy was 12.5%, the balanced
constant-class baseline. The unrestricted first prediction was a space.

Ordinary 8,192-target DEV NLL means (seed SD): full 3.48940 (0.02019), detached
3.49378 (0.01093), current-only 3.48564 (0.01416). These are early internal
learnability results, not converged language models or an architecture ranking.
Compatible initializers/FFN slices and every paired 512-step input batch match.
Current-only still has native fast/slow/rotary/SSA history and reallocates active
FFN capacity; it is not history-free or a same-shape intervention. Full FLOPs
and hardware parity are open.

Actual first-answer-only activation credit reaches earlier chunks in the full
condition, with first/last gradient-L1 ratios 0.13771, 0.14109 and 0.12289; the
declared detached condition gives exactly zero earlier activation credit.
Passing causal, owner, source, replay, numerical and surrogate-gradient audits
does not imply useful retrieval.

The pre-results supplemental cases also fail local copy. Post-null checks keep
the original primary unchanged and test both key families at the exact TRAIN
answer position 380, plus the final sixteen actual TRAIN episodes. All scores
remain zero. Every TRAIN answer had the same position; this is a confound for
generalization, but not the sole explanation of the observed failure.

First-layer phase addresses are constructed before contextual native outputs
are mixed, so the final query-space address is context-free. Later layers and
query-driven writes are contextual; whole-model query blindness is not proved.
The inherited normalized preconditioned write is not exact RLS. No tested
capacity ceiling, exact global-attention replacement or quality/compute advantage
follows from these fits.

COLLECT017, added after these nulls and registered before its own fits, also
fails local and distant copy in all three seeds using native causal attention.
This identifies the current recipe as an unsuccessful learnability test, not an
architecture discriminator. A future positive control should establish local
first-byte binding and delimiter/position handling, then increase delay. Context
composition for addresses/routers and decoupled current-value/admission paths
are untested candidates described in CANDIDATE_REPAIRS.md.

The original fit source is 2209e1328e3b42db4dd48cd702a21a55ffad24af, original
run https://github.com/FPC-effortless/TAC-OSM/actions/runs/41544326038. The
supplemental/audit/address protocols were published before inspecting full-fit
scores at 28ad49d1a29007ea3affa6b601e8fa2d92c963b6. Position/frequency checks
are explicitly post-null. SUMMARY_PUBLIC.json retains every seed; original
workflow ZIPs, full snapshots and raw read-only evidence accompany the report.

4,608 full updates +24 smoke updates; 36 full snapshots +6 smoke snapshots;
3,538,944 full-fit input bytes presented, including generated framing; 18 finite
512-byte generations at temperatures 0.8/1.2; zero GPU minutes. All failed
metadata/Lean attempts and their repairs are retained. Internal gates keep
architecture-superiority eligibility false. Full Lean/source/IEEE/AD proof,
independent qualification, protected TEST and promotion remain open. Draft PR
#700 is not merged.
