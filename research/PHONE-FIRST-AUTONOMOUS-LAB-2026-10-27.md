# TAC-OSM phone-first agent research laboratory

Updated 2026-10-27. Designed for research from an Android phone, using
only AI coding agents (when connected), repo instructions, GitHub and Kaggle.
It EXTENDS the already merged lightweight lab and signed reward-hacking gate.
No historical experiment, source cohort, checkpoint or fit was rewritten.

## Phone home screen

- [Lab progress Issue #472](https://github.com/FPC-effortless/TAC-OSM/issues/472):
  GitHub CPU planning and safe no-fit diagnostics.
- [Kaggle T4 x2 Issue #473](https://github.com/FPC-effortless/TAC-OSM/issues/473):
  protected remote GPU setup and available remote status.
- [Phone Research Actions](https://github.com/FPC-effortless/TAC-OSM/actions):
  manual run dispatch, automatic research triage and Kaggle progress checks.
- The existing PR feed is the work review queue. The existing research lab
  CLI and notebook remain available unchanged for local scalar metrics,
  checkpoint hashes, plots and evidence retention.

**You only need your phone** to make a GitHub Issue, inspect one updated
status comment, look at the Actions summary, review an agent PR, and approve
a quota-consuming GPU workflow when appropriate.

## Automation currently built

1. Three times per day, phone-research-control.yml reviews EVERY JSON
   proposal under research/phone_queue; runs deterministic compute routing;
   executes a bounded, source-only diagnostic suite for at most three
   eligible short no-fit GitHub jobs; updates a bot status comment for
   tracked GitHub CPU Issues and publishes JSON/Markdown evidence artifacts.
2. Kaggle monitor runs three times per day. With a repository-scoped token it
   checks remote status; without one it checks the last successful GitHub
   submission with read-only Actions access, and explicitly marks remote
   execution **unverified**, not absent. It does not submit or spend GPU quota.
   A completed notebook is NEVER scientific qualification.
3. AI agent instruction roles under .github/agents guide coding researchers
   and adversarial reviewers; CHARTER.md and AGENTS.md remain compulsory.
   A compatible coding agent must be enabled in GitHub to actually write
   new PRs without a chat session. The scheduler itself is NOT a hosted
   generative coding agent and cannot invent new experiments.
4. Per-study JSON proposals are reviewable, stable, hash-fingerprinted
   and protected from issue-text shell injection. The agent or human first
   runs python scripts/phone_lab.py plan on the proposal.
5. A justified GPU job chooses Kaggle NvidiaTeslaT4 exclusively, i.e.
   **two Tesla T4 GPUs**. Never silently substitute a CPU, one T4,
   P100/TPU/A100, or random available accelerator. The Kaggle runtime
   verifies TWO physical T4 names, CUDA operations on both devices,
   exact workload hash and a reviewed 2-GPU seed-shard or DDP strategy
   before any experiment step. Each of two seed workers receives a
   distinct physical GPU, with outputs and worker logs preserved.
6. GitHub Actions GPU dispatch has separate compute-planning and
   permissioned execution jobs. The protected environment named
   kaggle-research is intended to require operator approval.
   GPU submission is disabled until protected environment secret
   PHONE_LAB_GPU_EXECUTION_ENABLED is explicitly set to true.
   Actual Kaggle API token and username must also be provided.
7. Trained/evaluated real-world studies additionally need exact original
   source + frozen FAIR_COMPARISON manifest + signed independently verified
   source, outcome, reward and ACL preflight receipt and trusted public
   verifier key, before the Kaggle API push is even attempted.
   Postfit science and checkpoint promotion remain SEPARATELY gated.

## Compute selection before every run

| Situation | Venue | Reason |
| --- | --- | --- |
| CI, lint, source audit, small no-fit, concise CPU task | GitHub Actions | No GPU quota; easy replay/logs |
| CUDA model or genuinely long GPU-efficient fit/eval | Kaggle T4 x2 | Two physical T4s, no fallback |
| Long CPU job without measured reason for GPU | BLOCKED redesign | Do not burn hours or manufacture a GPU justification |
| Missing source authorization, equal-budget plan, signing receipt or quota | BLOCKED | No fabricated approval |
| Kaggle unavailable/quota exhausted/not two T4 devices | BLOCKED | Do not switch GPU class or assert benchmark comparable |

All GPU budgets have an explicit upper bound in the proposal (max 240
GPU minutes per approved submission, conservative policy). CPU time is
an efficiency metric, not a fairness parity requirement. Parameters,
original training data, training budget and evaluation conditions MUST
remain matched. A T4 x2 allocation is two separate 16GB GPUs, NOT a
shared automatic 32GB device. Your model must be DDP-aware or allow
independent seed workers to utilize both.

## First use on your phone

1. Open Issue #472 or write a new research issue via the repository.
2. Have your GitHub coding agent create/update a reviewed proposal JSON
   under research/phone_queue and provide source and hypothesis;
   the planner will refuse unsafe or incomplete fields.
3. In GitHub Actions, find Phone Research Lab automatic planning and
   progress; use Run workflow for an immediate no-fit review.
4. Open the Actions run summary (table of variant, stage, compute route
   and BLOCKED/READY) or Issues to see the current progress.
5. To test Kaggle after explicit approval, set up the environment described
   below, then open the Phone Research Lab approved Kaggle dual T4
   execution workflow; choose the reviewed Kaggle dual-T4 smoke proposal,
   type APPROVE_T4_X2, and approve the environment in GitHub.
6. Remote execution only begins if the runner has an operator-enabled
   Kaggle token and accepted quota. The first seed-sharded smoke job uses
   constructed CUDA tensor arithmetic only: no model fits, source data
   or capability claims. Its result shows whether BOTH T4 devices exist.

## Operator setup needed; currently NOT configured by this PR

On GitHub repository settings (phone browser Desktop Site if required):

- Create protected environment named kaggle-research and require a
  reviewer (must be configured by the repository owner; this code cannot
  provision repository environment rules).
- Add environment secret PHONE_LAB_GPU_EXECUTION_ENABLED with value true
  ONLY after reviewing the job limits and approving quota usage.
- Add secret KAGGLE_API_TOKEN from your Kaggle API settings. Prefer scoped
  token; never post token/key in chat, Issues or git files. A token ONLY in the
  protected `kaggle-research` environment is usable by approved submissions
  but NOT by unattended status checks. For automatic remote polling also set
  `KAGGLE_API_TOKEN` as a repository Actions secret (a wider credential scope);
  otherwise the monitor reports only accepted GitHub submission evidence and
  the user can examine the private Kaggle notebook manually. Do not weaken
  the submission approval or expose the token in logs.
- Add repository or environment variable KAGGLE_USERNAME.
- For *genuine trained models only*, separately configure a non-model
  independent trusted public key TAC_OSM_REWARD_VERIFIER_PUBLIC_KEY_HEX and
  privileged signed PREFIT receipt TAC_OSM_PREFIT_ATTESTATION_JSON from
  an external verifier. Do not give the private signing key to agents.
- On Kaggle, activate/verifiy your account and confirm an available T4 x2
  allocation and quota. Kaggle may queue/refuse exhausted GPU capacity;
  this software will not assume a grant.

GitHub executes the official Kaggle CLI with accelerator NvidiaTeslaT4.
Metadata also pins the same machine_shape, enable_gpu true,
is_private true and enable_internet false. This implementation has no
Kaggle API credentials, hence live GPU dispatch and quota usage have NOT
been tested in Kaggle. The source-only packaging/runtime contract is
testable in GitHub CI with explicit no-fit fixtures.

## Limits and next actual research integration

No autonomous research **training agent** is deployed just by adding an
agent prompt file. If your GitHub Copilot coding agent or Codex is
connected, assign a research Issue to it so that it can propose and
review PRs under the charter. Without that connection, the scheduler
still manages issues, compute decisions and safe automatic diagnostics.

The only registered Kaggle workload in this PR is a no-fit T4 x2 smoke
probe. No Kaggle real-world dataset, private evaluator, live signed
prefit evidence, Kaggle-trained model checkpoint, postfit independent
scorer or two-T4 utilization receipt has been provided. Actual
experimental workloads must be implemented as reviewed source-controlled
scripts, require original data and signed gate evidence, and preserve
exact weights plus replay materials for independent verification.

Do not turn on background paid API coding or GPU runs without explicit
user authorization. Notebook quota costs and GitHub Actions minutes
must be checked against platform limits.

## Read-only verification of the already-submitted dual-T4 smoke

A successful GitHub `kaggle kernels push` receipt is not proof that the
private Kaggle kernel executed, or that two workers succeeded. The protected
workflow `.github/workflows/phone-kaggle-smoke-audit.yml` is a **manual,
read-only** check that can use the existing `kaggle-research` environment's
Kaggle token without expanding secret scope to unrelated scheduled jobs.

On GitHub Actions select **Phone Research Lab • read-only Kaggle dual-T4
smoke evidence** → **Run workflow**, then approve the protected environment
if requested. This action does NOT submit any Kaggle kernel, change versions,
request accelerators or consume new Kaggle GPU execution quota. It downloads
only `phone-research-runtime.json`, `seed-101.json`, `seed-102.json`
from the existing saved kernel using Kaggle's supported `kernels output`
command. The offline auditor checks source SHA, frozen seed identifiers,
two distinct worker indexes, physical slot mapping, successful child exits,
T4 device names, arithmetic and the explicit no-fit/no-science flags.

If any output is missing or cannot be read (including private notebook
access-denied), the audit **FAILS** and archives only an incomplete status
report; do not attempt to resolve that by rerunning a GPU job. If the
receipts pass, status is **SELF_REPORTED_TWO_WORKER_RECEIPTS_CONSISTENT**,
not independent device attestation: Kaggle execution version provenance,
trusted independent device observability, and any model capability claim
remain distinct. The source-controlled report archive is nonsecret; raw
private Kaggle downloads are not uploaded to GitHub Actions artifacts.

## Useful source references

Official Kaggle CLI:
https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md
Metadata:
https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md
Kaggle token authentication:
https://github.com/Kaggle/kaggle-cli/blob/main/skills/references/auth.md

## Your existing ChatGPT Cloud Codex is the coding agent

Use [the Codex Cloud mobile handoff](CODEX-CLOUD-PHONE-HANDOFF-2026-10-27.md) for the prepared prompt and one-time published environment check. You do not need Claude Code, CrewAI or another agent runtime on your phone. GitHub Actions remains a deterministic diagnostics/monitoring system; it is not automatically capable of creating Codex Cloud tasks without a separate integration and approval.
