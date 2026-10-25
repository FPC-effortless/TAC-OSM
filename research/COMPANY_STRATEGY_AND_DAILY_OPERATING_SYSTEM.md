# Company strategy and daily operating system
**Adopted:** 2026-10-25. **Status:** binding research/product prioritization guidance, not an experimental result, deployment claim, or authorization to merge or spend money. **Authority:** [Unified Master Research Plan](MASTER_RESEARCH_PLAN.md), all existing research/evidence rules, and original immutable experimental artifacts take precedence over this strategy where technical gates differ.

## Mission and positioning
Build an open-source AI research and engineering company focused on **reliable, efficient, stateful predictive intelligence**: systems that maintain useful changing-world representations, forecast consequential outcomes, propose bounded decisions and verify what actually happened. Pursue scientifically distinctive TAC-OSM/PNDS model capabilities and commercially useful tools **in parallel**, not by pretending unqualified research is a product. Initial market hypothesis: dependable stateful agents and operational decision support; validate buyers, jobs, use cases, willingness to pay, data rights, deployment and outcomes rather than asserting traction.

### Three simultaneous company layers
1. **Research laboratory:** TAC-OSM six research families, shared parallel/streaming computational graph, PNDS full-input-to-output mathematical engineering, and real-data capability research.
2. **Open developer ecosystem:** Forge dataset creation/cleaning/provenance, TrainLab reproducible experiments and comparison receipts, Veritas actual-state/outcome agent qualification, SDKs, model evaluations, source/licensing and first usable docs. **Forge and TrainLab are proposed product identities unless independently verified implemented.** PNDS remains the mathematical/proof authority.
3. **Commercial systems:** Veritas agent qualification and/or advisory operational forecasting as competing **initial product hypotheses to validate**, Woyengi authorized state/execution orchestration, managed integrations and eventually supported predictive model inference. Neither is established paid product-market fit. Distinguish research, product readiness, commercial evidence; do not use one as proxy for another.

The practical product recommendation is to validate **Veritas state-integrity/agent-qualification workflows early** because they can be valuable without a new TAC-OSM capability. Preserve the older **Enterprise advisory workflow-risk/time-forecasting wedge** as an equally explicit market hypothesis until real customer conversations, attainable data and measured pilot economics determine which is strongest. No invented independent customers, pilots, revenue or traction.

## Model portfolio: exactly six active TAC-OSM research families
Do not create a seventh active model family, rename historical artifacts, or imply six qualified trained checkpoints. Reuse a common mathematical core where supported; each lane has distinct real data, labels and gates. The [existing six-lane binding section](MASTER_RESEARCH_PLAN.md#binding-six-regime-tac-osm-model-family-and-evidence-roadmap-2026-10-23) and most recent architecture/evidence decisions control details, including the prospective 95k–105k parameter policy and fair-comparison requirements.
| Family | Research role | Product opportunity (hypothesis) |
| --- | --- | --- |
| System 1 | Fast observations, calibrated immediate event/risk/readout and permission-bounded action proposals | Real-time workflow/agent routing |
| Physical | Genuine robot/physical action-conditioned dynamics and measured state transitions; strongest persistence/velocity/servo references | Industrial prediction, robotics advisory |
| Enterprise | Real entity-linked process events, status changes, multiple-horizon likelihood/time forecasts and operator decision costs | Operational advisory / agent state evaluation |
| Temporal | As-of/bitemporal state reconstruction and irregular/time-dependent forecasting, with source/knowledge time protected | Auditability, event intelligence |
| Edge-Lite | Real hardware export, quantization, memory/energy and numerical behavior of qualified models | Private and offline deployment |
| Counterfactual | Action alternatives and consequences, separated from unidentifiable observational causal effects | Verified scenario analysis, guarded planning |

**Research integration:** shared encode → source-authorized structured fast/slow/entity/associative state → mathematically specified parallelizable transition and incremental causal update → specialist prediction/uncertainty/decision heads → independent verification/correction, keeping CED design where source-backed. Boundaries: TAC-OSM is predictive/advisory; PNDS formalizes; Woyengi owns authoritative facts/action permissions; Veritas evaluates. Avoid making all recurrent/nonlinear operations falsely "parallel" or turning design formulas into whole-model Lean proof claims.

**Specialized research heads, not extra flagship models:** memory-usefulness estimation, source-causal entity/event retrieval, anomaly scoring, event/time forecasts, decision ranking, calibrated uncertainty, factual correction and safety/consistency checks. Retrieving a useful memory is not the same as a verified downstream gain. EWA001/002 utility weighting harmed the matched similarity control; EWA002 bounded hashed lookup missed the scored-candidate budget. These negatives must constrain any new proposal, not be erased by rhetoric.

## Non-negotiable experimental operating rules
- **Capability-first:** choose a meaningful missing end-to-end ability and train/test the full integrated model where legitimate data exist. Do not delay substantive capability work for default small ablations or attribution matrices. Use controls when necessary to answer a falsifiable question, and do matched architecture comparisons *before claiming an architecture winner*.
- **Authentic data:** genuine observed temporally ordered records, source rights, exact online-available inputs, identities, actions and verified outcomes, time splits, truthful censoring and complete denominators. Simulation/synthetic tasks are diagnostics, not real-world qualification. Avoid opening or repurposing protected TEST.
- **Matched comparisons:** when claiming a winner, equal parameter counts within active gates, equal data/splits/exposure/training compute and fair tuning/evaluation; mismatches explicitly disqualify head-to-head superiority. CPU wall time is separately measured efficiency, **not** compulsory equal training-time wall clock. Historical checkpoints and gate definitions cannot be rewritten.
- **Provenance and replay:** pre-register hypothesis/metrics/baselines/thresholds/seeds; preserve original model/optimizer/RNG/checkpoints/scores/source/license/config/receipts in Git; independent clean checkout and GitHub replay. Publish real failures. Separate software PASS, mathematical reference/proof PASS, authentic developmental improvement, independent untouched test, deployment qualification, and customer ROI.
- **Math:** PIN exact learned tensor sizes, every input/output operator, recurrence/scan associativity assumptions, update order and computational/activation cost; specify Python reference and Lean obligations in PNDS, compare actual PyTorch gradients/FP32/FP64 against references, state proof limits. No unsupported whole-model proof or general-world-model claim.
- **Operational boundaries:** read-only/shadow-mode advice until permissions, security, monitoring, abstention and measured benefit are established. No automatic merges, customer actions, business expenditure or deployment without authority; license model weights, SDK, datasets and any external components carefully.

## How to decide the NEXT task (daily priority engine)
Each day, read the **latest** plan, original recent GitHub PR/results/CI, current evidence inventory, outstanding blockers and any user-directed work. Prefer the following order, adjusting for real findings and project milestones:
1. **P0 / hard blocker or safety/evidence integrity:** failed verification, missing artifacts, leaked/test contamination, correctness/math problem, source access or qualification problem. Preserve negative evidence.
2. **P1 / one substantive integrated capability experiment:** an authentic outcome-backed gap with a complete preregistered fit+score+replay path. Focus on strongest model capability first, including early/difficult failures and reliable uncertainty.
3. **P1 / one user-facing open-source artifact:** small end-to-end value from Veritas/Forge/TrainLab/Woyengi SDK/documentation/tests, independently usable without TAC-OSM breakthrough. Do not invent implemented scope.
4. **P2 / evidence-based market validation:** document a buyer hypothesis, real interview/feedback, a verifiable problem/benchmark/pilot proposal, license/security blocker and commercial value hypothesis. Avoid speculative outreach as completed.
5. **P2 / environment efficiency and scientific synthesis:** state/compute/energy profiling, research comparison, architectural ideas registry, new article review only when tied to a live measurable capability gap.

Do **not** treat all six families as six mandatory concurrent workstreams. One or two active capability lanes plus a product lane is enough; protect historical work and maintain the others as research tracks. Early research distribution can emphasize System 1/Enterprise/Temporal, but actual current research and evidence-supported priorities (e.g. Physical/Temporal developments) override an arbitrary fixed percentage allocation. Edge-Lite is a hardware qualification/deployment specialization, and Counterfactual remains causal-evidence-gated.

## Daily operating cadence — flexible blocks, not assumed calendar appointments
**Morning (~8 AM Africa/Lagos, flexible scheduled briefing):**
- Review latest repository evidence and external blockers; distinguish merged/main, canonical-plan-only, draft/unmerged, verified, in-progress, and proposed.
- Select **three outcomes for today**, each with: why now, exact source branch/files, next executable task, measurable artifact, and an acceptance gate. Prefer **one research**, **one useful open-source/product**, and **one founder/business/integration** outcome; shift all three to a verified research blocker when necessary.
- Write a *stop-doing* choice: which attractive ideas should not consume today's work. Never invent a completion estimate.

**First protected work block — deep research:** input-to-output math and actual capability building/training/independent checking. Plan fits and time-split source gates before scoring; write source-backed results. Do not claim success from a proposed experiment.

**Second block — product/developer adoption:** ship/verify a smallest independently valuable tool improvement (e.g. Veritas state mutation receipts, replay QA, Forge split/provenance audit, TrainLab comparable run manifest, Woyengi guarded adapter) without overstating it.

**Third block — commercial learning and engineering hygiene:** customer problem validation if evidence is available; review licenses, documentation, model packaging and operational risks; record decisions and blockers. No invented conversations.

**End-of-day evidence ledger:** report DONE / IN PROGRESS / BLOCKED; exact commits/PRs/CI/results if available; what got better, worse and unknown; money/customer evidence (if any); tomorrow's next falsifiable task. Update the canonical master research plan and linked ideas registry **only when authoritative findings or policy changes warrant it**, preserving append-only history. No unattended merges.

**Weekly strategy review:** review trained capability gates, developer adoption evidence, customer discoveries, actual runtime and cost metrics, open-source quality, research backlog, and legal/data/IP restrictions. Choose one primary scientific bottleneck and one primary adoption bottleneck for the following week. Reallocate focus from evidence, not hype; never lower historical gates to report progress.

## Required daily briefing output format
**Research state** — fresh verified results, six-lane status (only if changed), open safety/math/evidence blockers, exact source references.
**Today's Top 3** — each priority states named project, realistic task, expected artifact, metric, pass/fail criterion and repository branch.
**Execution sequence** — what can proceed in parallel and what depends on an external source, GitHub CI, new data or user approval.
**Company progress** — Forge/TrainLab/Veritas/Woyengi advancement, actual external user evidence and financial hypotheses vs facts.
**Closeout checklist** — tests, checkpoint/archive, independent replay, notes/PR draft, failed hypotheses, next-day handoff.
When repositories cannot be accessed, say so; produce a constrained plan from the last verified evidence, do not invent up-to-the-minute state.

## Baseline at adoption (2026-10-25; frozen historical note)
- **ME105 Temporal:** fifteen actual fitted model variants recovered primary PM2.5 coverage, but **full five-seed qualification FAILED**; calibration/point-sharpness/other-channel tradeoffs remain. Draft PR406/PNDS PR18; next research hypothesis is separating point vs uncertainty gradient roles and channel-specific calibration, with no protected 2016–2017 source use.
- **EWA002 Physical:** genuine independent ALOHA candy task: similarity-only residual cache offered only marginal improvement over strong velocity reference; verified-usefulness reranking regressed and hash scoring demand exceeded gate. **Full gate FAILED 0/5**; PR405 draft. Follow-up would require a *new* untouched real source and preregistered stronger physical references, not a fitted-on-exposed rerun.
- Prior research and six families remain preserved. Neither a robot policy, autonomous driving, whole-model formal proof, customer-ready model nor paid deployment is qualified by these results.
- First practical founder hypothesis to test: Veritas state-correctness and action-verification workflow for developers; compare with the earlier Enterprise operator-risk/time-advisory wedge using authentic customer problem evidence. Forge and TrainLab are proposed developer-platform capabilities pending implementation verification.

## Scope and update policy
This strategy **guides** priorities; it does not automatically modify separate repositories or claim that a scheduled assistant can directly run multi-hour experiments. Daily briefings may advise actions; commits, training, merges, deployments and external spending require the actual connected permissions, source facts and user authorization. Link future results rather than rewriting this adoption baseline.
