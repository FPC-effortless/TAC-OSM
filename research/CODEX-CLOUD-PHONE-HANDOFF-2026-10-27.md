# Cloud Codex from ChatGPT Android — the research agent for TAC-OSM

The operator already uses **ChatGPT on Android with Codex Cloud**. Do not
require Claude Code, CrewAI, Cursor, a local laptop or a separate coding-agent
subscription to get started. Cloud Codex is the human-instructed coding and
research agent; GitHub Actions is the automated check/issue-status runner;
Kaggle is the permissioned accelerator execution venue.

## One-time cloud environment requirement

Use an existing **published Codex Cloud environment** with GitHub repository
`FPC-effortless/TAC-OSM`, or connect this repository and publish an
environment from the Codex web/desktop interface once. Cloud tasks can
then start and continue from ChatGPT mobile. Do not confuse a Codex Cloud
environment with Remote Control of a powered-on desktop computer.

Codex automatically reads repository `AGENTS.md`; task prompts explicitly
require `CHARTER.md`, `CODEX.md` and the relevant scientific research rules.
GitHub issue status and Actions summaries remain the durable phone dashboard.

## How to start a real coding/research investigation on your phone

1. In ChatGPT, open your **Codex Cloud** task interface and select the
   published TAC-OSM environment for `FPC-effortless/TAC-OSM`.
2. Give Codex a bounded task or send the following template, including a
   specific GitHub Issue and the one next scientific question.
3. Review Codex's proposed hypothesis/source authorization/compute decision,
   test results, code diff, and draft PR. Continue the same cloud task if
   revisions are needed; avoid opening independent duplicate tasks.
4. Monitor GitHub **Issue #472** and the GitHub Actions progress summary.
   A separate **Issue #473** tracks Kaggle T4×2 readiness.
5. When a GPU is genuinely needed, give explicit approval through the
   protected `kaggle-research` environment; never paste Kaggle keys in Codex
   prompts, GitHub Issues or source files.

## Reusable Codex Cloud task prompt

```text
Repository: FPC-effortless/TAC-OSM
Track progress in GitHub Issue #472 (or a new linked scientific Issue).

Read AGENTS.md, CHARTER.md, CODEX.md,
research/REAL-WORLD-EVIDENCE-RULES.md,
research/PHONE-FIRST-AUTONOMOUS-LAB-2026-10-27.md,
research/REWARD-INDEPENDENT-RUNNER-GATE-2026-10-26.md,
and the existing experiment history before changing code.

Act as a researcher, engineer, and independent adversarial reviewer.
Choose one unresolved genuine real-world capability hypothesis.
Locate original evidence, preregister FAIR_COMPARISON constraints,
preserve ~100k real learned parameters and matched comparators, use equal
original data and training budgets, protect sealed TEST, and preserve
replayable checkpoints plus PNDS Lean/Python proof obligations.

BEFORE any training/evaluation, write a reviewed JSON protocol under
research/phone_queue and run scripts/phone_lab.py plan on that file.
Use GitHub Actions for bounded CPU/source/diagnostic checks; choose Kaggle
only if real CUDA/acceleration is justified. Kaggle must request
NvidiaTeslaT4 (T4 x2), verify both GPUs at runtime and use both;
no alternative accelerator or hidden CPU fallback.

Never spend GPU quota without my protected GitHub environment approval;
never bypass independently signed real source/reward/goal/access preflight;
never call no-fit synthetic diagnostic evidence a trained capability pass.

Commit small, reviewable changes on a branch, run tests, open a DRAFT PR,
and post phone-readable updates to the linked Issue:
completed / planned / blocked / next run / evidence / decisions required.
Do not merge unverified model research or invent historical results.
```

## Important distinction: what runs automatically

- GitHub Actions already performs **scheduled no-fit diagnostics, compute
  planning and progress reporting**. It does NOT independently initiate
  generative Codex Cloud reasoning/coding tasks just because an Issue exists.
- Codex Cloud runs tasks you start in its cloud interface; it can keep
  working when your laptop is off. If you want *scheduled generative
  research*, configure a separately authorized Codex/Work scheduled or
  event-driven workflow where the product and account allow it. Protect
  quotas, approvals and secrets; do not claim it exists without setup.
- Kaggle GPU work requires configured Kaggle secrets, available quota and
  protected environment approval. Selecting NvidiaTeslaT4 does not
  guarantee an available two-T4 accelerator at runtime.
- A finished PR, green Action or completed Kaggle notebook is **not**
  independent proof of model performance or reward integrity.

## Verify the user-visible flow

Visit [Issue #472](https://github.com/FPC-effortless/TAC-OSM/issues/472)
and [Actions](https://github.com/FPC-effortless/TAC-OSM/actions).
For Kaggle GPU prerequisites use
[Issue #473](https://github.com/FPC-effortless/TAC-OSM/issues/473).

Official guidance:
- https://help.openai.com/en/articles/20001545-using-codex-cloud
- https://help.openai.com/en/articles/11369540-using-codex-with-your-chatgpt-plan
- https://help.openai.com/en/articles/20001275-chatgpt-work-and-codex

## Operator-authorized daily research continuation

On 2026-10-27 the operator requested both immediate research and automatic
continuation. An enabled Codex/Work task named **Continue TAC-OSM research**
was created with a daily flexible schedule. This is external account state,
not a schedule provisioned by this repository; inspect the task in ChatGPT
to manage its timing or pause it. It requests implementation and draft PRs,
not merely a reminder, and requires reading current repository rules, Issue
#472 and open PRs on each run to avoid duplicate work. Creation was confirmed;
a successful future execution has not yet been observed.

Each run must preserve scientific gates and report actual tools/access
blockers. Independent evaluator credentials and GPU approval remain separate.
The first bounded investigation is documented in
[label-blind history controls](LABEL-BLIND-HISTORY-CONTROLS-2026-10-27.md).
