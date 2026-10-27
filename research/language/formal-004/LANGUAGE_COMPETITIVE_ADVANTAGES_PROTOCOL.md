# TAC-OSM language and memory: competitive-evidence and product qualification gates

Research review: 2026-10-28. This is a **test plan**, not a claim of superiority, product readiness or revenue. It belongs to the **System-1/Language specialization within the existing six consolidated TAC families**, not a new seventh model.

## What competing research already demonstrates

- **Mamba-3 (ICLR 2026)**: explicitly targets hardware-efficient recurrence, state tracking and retrieval. Reports comparable perplexity to Mamba-2 at half the state size in its experiments and downstream results at **1.5B** parameters. https://proceedings.iclr.cc/paper_files/paper/2026/hash/8abd2043b71a074278d5f687947bff9c-Abstract-Conference.html
- **RWKV-7 (2025)**: mathematically designed for constant-inference-memory state and parallelizable training, with public 0.19B–2.9B pretrained-model lineage. https://arxiv.org/abs/2503.14456
- **LongMemEval (ICLR 2025)**: real assistant memory evaluation spanning information extraction, cross-session and temporal reasoning, knowledge updates and abstention. https://github.com/xiaowu0162/LongMemEval
- **LongMemEval-V2 (2026)**: agentic historical memory with multimodal web trajectories, experience-based questions and retrieval latency. https://github.com/xiaowu0162/LongMemEval-V2
- **NoLiMa (ICML 2025)**: long-context retrieval where question and stored material have weak literal lexical overlap. https://github.com/adobe-research/NoLiMa
- **LongBench v2 (ACL 2025)**: 503 complex real long-context tasks spanning documentation, dialogue, codebases and multi-document reasoning. https://aclanthology.org/2025.acl-long.183/

**Implication:** Parallel scan, fixed recurrent state, or attention-free updates are **not** standalone competitive advantages. The business thesis requires *demonstrably higher useful source-memory retrieval per measured end-to-end cost*, not a larger context-length figure.

## Two deliberately separated evidence tracks

### Track A — Native TAC language research

Status after MLAG001: real Vim manual 100,680-parameter character model; next-character DEV NLL ~2.827 and tested historical delays 64/96 within 128-character windows. Five paired seeds × two source gate modes; no independent TEST, model-to-model fairness beyond the objective ablation or billion-scale benchmark yet. The dual-horizon objective substantially improves the trained 96-character query, but the method is not qualitatively competitive with production LLMs. All source math/Lean claims remain scoped.

Gate A1 (reproducible real-text character baseline): source and checkpoint CI replay must be available, new independent real whole-document TEST, fixed corpus provenance/license, frozen objectives/thresholds before TEST. Compare equally sized GRU, Transformer, Mamba/RWKV-like implementations if feasible under exactly recorded 95k–105k active learned-parameter band, within 1% parameter gap, equal examples and optimizer-update budgets. Separate **parameter parity** from **FLOP parity**, and report both. Strong unigram/bigram and no-memory controls mandatory.

Gate A2 (genuine retained information): register history horizons 64/96/128/256/512 and up, source-ordinal/owner-disjoint retrieval, right-key vs wrong-key (same-target intersection), zero-memory lesion, eventual fact supersession and chronological state reconstruction. Use authentic source identities; constructed tasks may diagnose internal operators only. Measure historical correctness, NLL, query identity, state interference and forgetting. A simple FIT prior is not historical memory.

Gate A3 (language generation and economics): held-out cross-domain next-token bits/character, coherent generation and semantic/temporal instruction compliance; p50/p95 first-token/decode latency, actual GPU/CPU power, batch throughput, cost per **quality-matched** token, memory footprint including external records and source audit receipts. Parallel scan theoretical complexity is NOT a throughput claim.

Gate A4 (larger-scale external validation): only after trained semantics and ingestion are ready, adopt LongBench v2, NoLiMa, and conversational LongMemEval memory. A 100K-parameter character baseline cannot answer many of those tasks unaided, so mark **not applicable**, not fabricated low scores. Report scaling requirements and independent comparison budgets; never represent 100K architecture diagnostics as Mamba-3 1.5B/RWKV-7 multi-billion head-to-head.

### Track B — Memory Engine for existing agents (nearer-term commercial test)

Build one **source-authorized persistent memory component**, not a chatbot model replacement. The minimum open-source interface:

- `append_event(owner_id, source_id, event_id, ordinal, occurred_at, observed_at, payload, authorization)`: append immutable identity/provenance; reject cross-owner reuse; support monotonic chronology and independently auditable deletion/retention.
- `update_fact(owner_id, entity_id, version, supersedes, evidence_refs)`: preserve old source as historical truth while marking newer claims current; reject future-source and unverified authority.
- `query_memory(owner_id, entity_id, as_of, question, budget)`: return `evidence_refs`, validity interval, confidence/calibration (if measured), route, abstention and **full cost/latency receipt**. Never claim knowledge not supported by retrieved evidence.
- `export_replay(owner_id)`: deterministic source/event manifest, access log and replay; permission checks at each read and write, with external storage/accounting explicit.

First baseline should be **deterministic source/event index + recency**, then BM25 and embedding retrieval with identical allowed evidence and budgets. Compare TAC neural state only after it improves accuracy or total cost on a genuine authorized workload. Use LongMemEval, LongMemEval-V2 and NoLiMa-informed semantic mismatch protocols as they become compatible with this interface. Strongest near-term differentiators would be auditing, chronology, versioned knowledge updates and privacy-aware controlled forgetting, *if proven useful*.

### Product/revenue gates

No investor valuation or near-term customer traction follows from MLAG001. A hypothetical 50 customers × $20,000 annual contract = $1M ARR is **not a sales forecast**. Start with a narrow pilot: one authenticated owner-separated enterprise workflow with a measurable historical update/retrieval pain point; open-source adapter, JSON evidence receipts, documented access control and replay. Offer paid deployment/support **after** a real pilot measures improved outcome quality and cost versus off-the-shelf indexing/RAG.

Do not bundle a huge future language model, memory API, verifier and training infrastructure into one untestable product promise. Ship the narrow evidence-backed memory SDK first, preserve the native TAC language experiment as a high-risk longer-term research track.

## Required experiment reporting format

Every promotion report must include: source license/authenticity; date and hash of FIT/DEV/untouched TEST; architecture/optimizer/sample order/active learned params/FLOPs; downstream task and source-identity labels; negative baselines; full five-seed results and seed-block uncertainty; computer/device latency, energy, external storage; checkpoint and independently rerun CI link; uncertainty/abstention performance; failure cases. Any missing required item is an **OPEN gate**, not a success.
