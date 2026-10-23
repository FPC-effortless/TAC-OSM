# Compulsory TAC-OSM repository-owned checkpoints and experiment replay

**Effective: 2026-10-23. Status: binding for every new or continued trained model study, both local and externally executed (including GitHub Actions, hosted notebooks, remote accelerators and collaborator machines).** This rule applies to each TAC variant, each trained comparator, each seed, selected and rejected checkpoints, and negative/failed runs for which recoverable model states were produced. It supplements—not replaces—[REAL-WORLD-EVIDENCE-RULES.md](REAL-WORLD-EVIDENCE-RULES.md), [CAPABILITY-FIRST-RESEARCH-RESET-2026-10-22.md](CAPABILITY-FIRST-RESEARCH-RESET-2026-10-22.md) and the [MASTER_RESEARCH_PLAN.md](MASTER_RESEARCH_PLAN.md).

## Non-negotiable publication rule

**No run is "complete", independently reproducible, final-qualified, or eligible to support a new architecture/capability claim until its model checkpoints AND the exact materials needed to load, replay, audit, and recompute reported predictions/results are committed in a persistent, versioned location *inside this TAC-OSM GitHub repository*, retrievable from a clean checkout at a pinned commit.** It does not matter whether the training took place on a laptop, workstation, GitHub Actions, Kaggle, cloud GPU/TPU or another external system.

- A temporary GitHub Actions artifact, CI cache, notebook output, local `results/` or `runs/`, personal drive, unpublished tarball, expiring download URL, external repo, model registry or a SHA/hash **without the corresponding recoverable bytes in this repository** is insufficient. External copies may be supplementary only.
- Use committed version-controlled files under `research/replay-bundles/<study_id>/` (or an explicitly enumerated existing study directory), retaining exact git-tree identities. For weights larger than GitHub's practical per-file limit, use **repository-addressable, retrievable Git LFS objects** or byte-exact, indexed smaller shards in the same repository; a bare LFS pointer without retrievable object bytes **fails**. Never silently discard optimizer/checkpoint components to meet size limits.
- Pre-register a repository storage plan **before starting a new full training run**: exact bundle prefix, expected model/seed count, byte size/storage strategy, restore/replay procedure, privacy/licence constraints, GitHub upload and independent verification steps. Reserve storage and check repository permissions/quota beforehand. Do not start expensive work that cannot be preserved.
- **Before opening held-out labels when the protocol requires frozen checkpoints**, freeze and hash the final selected checkpoint for every arm/seed, with read-only immutable receipts. Publish the frozen weights and associated replay bundle to the repository as soon as permitted without revealing sealed labels or violating the holdout protocol. The final evidence report must reference the exact checkpoint commit and immutable identities.
- **All fitted arms and seeds** must have recoverable trained model weights and enough architecture metadata to recreate their module shapes, even when worse than a baseline. Retain last and best/selected state whenever different, checkpoint index/selection criteria, optimizer/scheduler state and RNG state if resumption or full training reconstruction is claimed. For interrupted runs, preserve the latest valid recoverable checkpoint and failure logs; mark unrecoverable interruptions explicitly as FAIL, never as completed fits.

## Minimum repository bundle

Each study's replay bundle must include, or point at **specific tracked paths in the same pinned TAC-OSM repository commit**:

1. **Model bytes:** original trained state for *each* arm and seed; weight format/version, loading map and exact SHA-256 of actual reconstructed bytes, not just archive labels. All distinct selected checkpoints and references required for the reported comparisons.
2. **Replay entry point:** runnable script/CLI, exact command and output locations to load original weights **without retraining**, recreate original predictions and recompute the exact metrics. Include expected numeric outputs, independent replay results and tolerance policy (bitwise if promised; platform-dependent tolerances declared beforehand otherwise).
3. **Execution contract:** pinned code commit, source/config files, architecture IDs and dimensions, Python/package lock or pinned dependencies, framework/OS/CPU/GPU information, environment variables **by name only** (no secret values), model load order, serialization versions and random seeds.
4. **Training/resumption provenance:** optimizer, scheduler, scaler, RNG and relevant sampler states, exact training order, steps and per-arm budget receipts, tuning/selection policy, checkpoints/failed attempts and original training logs. If full training replay is claimed, all necessary resumption state must be in-repo; otherwise explicitly state that only inference/evaluation replay is proven.
5. **Evaluation evidence:** preregistered protocol, exact dataset/source version/license, splits and entity/time IDs or immutable manifest hashes, preprocessing/normalization fitted on TRAIN only, action/missingness/censoring contract, outcome-scoring code, prediction arrays/metrics and measurement receipts. Raw sensitive/redistribution-restricted data must **not** be uploaded without rights.
6. **Data availability:** include redistributable tiny inputs, factual prediction fixtures or legal reproducible source acquisition scripts plus hashes and access terms. If true original evaluation data are not lawfully retrievable by an independent reviewer, label `FULL_EVALUATION_REPLAY=BLOCKED` even when weights and public test vectors can be replayed. A checksum and an inaccessible URL do **not** satisfy full reproducibility.
7. **Machine-readable manifest:** `REPLAY_MANIFEST.json` enumerating every original arm/seed/checkpoint, Git-tracked relative file locations, SHA-256 for actual bytes, code/environment/data dependencies, original and replay scores, artifact-completeness status and evidence of verification. Document all deviations as `FAIL` or `BLOCKED`, never silently drop them.
8. **Independent verification:** a clean checkout at a pinned commit (and `git lfs pull` if needed) must be able to resolve every claimed file, check its bytes/hash, load each frozen model and rerun predictions/score computations under the declared data access contract. Record replay command, runner/version, pass/fail, mismatch details and the verifying CI run/commit in the repo. Do not count an original producer machine reading its own workspace as an independent clean-clone audit.

## Mandatory gates and claims

| Stage | Required status | Scientific interpretation |
| --- | --- | --- |
| Before training | Storage plan, exact replay scope and file budget registered | `PLANNED`; never treat as model proof |
| After each fit, before experiment is declared finished | Recoverable exact checkpoint bytes for every applicable arm/seed in repository; manifest and hashes committed | Until published: `NOT_PRESERVED`; run cannot be called complete |
| After repository publication | Clean checkout proves file presence and hashes, and successfully reloads trained models without refitting | `MODEL_REPLAY_PASS` only if audited |
| Before comparative/final/result promotion | Independent scoring replay of original recorded observations/outputs and metrics, plus real-data source availability where needed | `FULL_EVALUATION_REPLAY_PASS` only if all dependencies are available; otherwise `BLOCKED` or `FAIL` |
| If any gate fails | Keep original failure, raw receipts and scientific limits; repair storage/serialization only without altering original training or opening/test-tuning | **No final qualification or capability/superiority promotion** |

**Do not mistake replay for scientific validity.** Passing this preservation rule never overrides real-data, causal identification, parameter/data/compute parity, heldout access order, benchmark integrity, statistical or safety requirements. Conversely, a successful Actions workflow is *not* checkpoint preservation or independent replay by itself.

### Storage, privacy and historical studies

- Version assets with checksum-indexed, independently reconstructable lossless shards or Git LFS rather than violating GitHub size limits. Keep original score/output files where licensing allows. If storage cannot be made available from the repository, stop publication/qualification and explicitly escalate the storage constraint; do not substitute a transient artifact.
- Do not commit secrets, access tokens, private customer records, protected data or unlicensed publisher archives. Redaction/encryption is acceptable only when an authorized reviewer can recover the **same** bytes using a documented permitted process; otherwise the corresponding full replay is `BLOCKED`.
- Prior archived studies are not retroactively relabeled as originally passing this rule. Audit them as `LEGACY_UNVERIFIED` or `LEGACY_PRESERVED` as justified; do not delete, fabricate or silently rewrite weights/results. Any **new fit, continuation, promotion or new result claim** after this effective date must meet the new rule, including older code paths such as `scripts/archive_trained_weights.py` that historically wrote only into gitignored `results/`.
- This is a **repository delivery rule**, not authorization to merge, deploy, publish proprietary data or spend on external storage. Preserve work in draft branches and verify the GitHub-hosted bytes before any final claims.

## Bundle handoff checklist

```text
Study ID / preregistration / source: ____
Original training host and GitHub branch+commit: ____
Trained arms/seeds/selected+failed checkpoints indexed: ____
Repository replay bundle path(s): ____
Per-file/blob SHA-256 and bytes available from clean checkout: PASS / FAIL
Model restore and original prediction reproduction: PASS / FAIL / BLOCKED
Original data availability / licence and scoring replay: PASS / FAIL / BLOCKED
Environment lock / command / original+replay scores: ____
Independent clean-checkout verification commit and run: ____
Frozen test access ordering, parameter/data/compute parity: PASS / FAIL / UNKNOWN
Final claim status: QUALIFIED / UNQUALIFIED / BLOCKED
```
