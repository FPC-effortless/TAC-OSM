# CODEX — phone-first autonomous research instructions

This is a short operator/agent index, not a replacement for AGENTS.md.

- Read CHARTER.md and AGENTS.md first.
- Keep GitHub Issues and Actions as the mobile status/compute interface.
- Before EVERY new training/evaluation, run the source-controlled compute
  planner on a reviewed research/phone_queue JSON manifest.
- Choose GitHub Actions for bounded no-fit and CPU experiments; choose
  Kaggle ONLY if a GPU is needed or justified with a recorded budget.
- Kaggle accelerator must be NvidiaTeslaT4, i.e. exactly two NVIDIA T4s;
  runtime must prove both GPUs exist and the reviewed code must use both.
  If unavailable, BLOCK rather than substitute P100, CPU, TPU, L4 or A100.
- Operator permission and signed independent original outcome/source/reward/
  evaluator/access attestation are NEVER replaced by an agent opinion.
- Protect the sealed TEST set. All primary performance claims need authentic
  source-qualified data, matched 100k parameter/data/training budgets,
  checkpoint replay and explicit qualification gates.
- Agents can prepare issue-scoped research PRs but must not quietly merge,
  spend quota, make provider accounts or claim completed trained results.
- Always leave a short phone-readable Issue update: completed, in progress,
  blocked, next action, exact run link, dataset and review caveats.

## Codex Cloud from ChatGPT mobile

If the operator already has ChatGPT Codex Cloud on Android, that is the preferred coding/research agent. Use its published `FPC-effortless/TAC-OSM` environment to start and continue tasks and review proposed branches/PRs. Follow the repo-scoped prompt and scientific controls in `research/CODEX-CLOUD-PHONE-HANDOFF-2026-10-27.md`. Do not require a second coding-agent framework. The GitHub Actions schedule runs the no-fit checks and status reports, **not** generative Codex Cloud tasks until separately authorized scheduling is configured.
