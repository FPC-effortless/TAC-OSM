# COLLECT016 — multi-window credit and supervised source retrieval

Internal ~100K-parameter research under Woyengi Intelligence Labs. No promotion,
global-attention replacement or resource-qualified architecture winner.

Three freshly trained conditions, three paired seeds, fixed 512 updates each:
connected 384-byte BPTT; state carried with detach at 128; current-value-only phase
lane with other native memories retained. Every step includes one original TRAIN
window and one constructed delayed copy episode with verbatim TRAIN quotations.
Full details and gates are frozen in EXPERIMENT_PROTOCOL.json and
EVALUATION_PROTOCOL.json before full fits. The original 768-update plan and all
24 smoke updates remain under results/prefreeze_v1 and original smoke folders.

Runtime imports pinned COLLECT011/012/014 source. All models stay inside the
95K–105K active/deployed band, within 1% between arms. No inactive padding; the
current-only lane reallocates removed address capacity to active FFNs. Equal
example/update exposure is enforced; full training FLOPs are not qualified.

Before fit: `python -m collect016.register`, exact source hash verification,
seven full-source prefit suites, and source-controlled CPU planning. Then frozen
LAUNCH_REQUEST.json triggers only the dedicated bounded CPU workflow. Each fit
retains initialization, 128/256/512-step model/AdamW/RNG checkpoints, original
training journal, source quote offsets, exact replay and unselected final model.

The tests cover actual cross-boundary gradient flow versus detach, all parameter
activity, owner/future isolation, chunk/state correspondence and whole-graph
FP32/FP64 numerical/surrogate-gradient references. Partial Lean definitions are
not a full source proof; mathematics/STATUS.json records OPEN release obligations.

RESEARCH_NOTES.md records the user's sufficient-statistic/biological proposals,
primary-source checks, corrections and untested engineering candidates.
