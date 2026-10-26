# Continuous no-fit architecture and anti-cheat research gates

**Effective:** 2026-10-26. **Purpose:** run engineering falsifiers
automatically on *every pull request* to `main` or `master`, and on
*every push* to those branches, with an optional manual dispatch.

## CI configuration

- `.github/workflows/research-anticheat.yml` — verifies nested/future
  information boundaries, source-contract poisoning, partition contamination,
  actual native causal-window fixtures, neural gradient failure detection, and
  other adversarial no-fit regression tests.
- `.github/workflows/research-observatory.yml` — runs live native
  architecture introspection, memory tomography, local autograd input
  dependencies, gradient-conflict and experiment-receipt falsifiers, plus
  batched-versus-single-entity isolation and local runtime instrumentation.
- Existing `.github/workflows/ci.yml` remains the complete Python test
  suite, including both suites and the provenance test-count checker.

Both specialized workflows run on **all PRs and all mainline pushes**
(no path filters), so unrelated refactors cannot skip the scientific
guardrails. No workflow starts training or opens publisher-protected TEST.
The runtime fixture check asserts *invariants*, not a maximum latency,
because runner resource sharing makes that comparison uninterpretable.

## Test results and human-readable evidence

Each GitHub Actions run includes a downloadable 14-day artifact:

| Workflow | Artifact | Contents |
|---|---|---|
| Anti-cheat | `anticheat-no-fit-ci` | JUnit test XML and structured JSON from the robot-window negative control |
| Observatory | `observatory-no-fit-ci` | JUnit test XML plus JSON architecture, memory, entity isolation, and runtime probe results |

The files deliberately identify the observations as
**constructed/untrained fixture evidence**. They're for auditing mechanism
validity, and must never be cited as real-world model accuracy.

Run locally:

    python -m pytest -q tests/test_research_ci_automation.py
    python -m pytest -q tests/test_research_anticheat.py tests/test_robot_anticheat.py tests/test_research_observatory.py
    python scripts/run_robot_anticheat.py --fixture
    python scripts/run_research_observatory.py --fixture --entity-isolation --profile

## Important limitations

Automatic GitHub Actions execution is not the same as configuring GitHub
*required status checks* on a protected branch. Branch rulesets must be
verified or configured separately by a repository administrator. Where
branch-protection API access is unavailable, do not claim the checks cannot
be bypassed by an authorized actor.

This CI does not interrogate all six trained variant checkpoints or their
real-source data lineage. It validates the diagnostic tooling, which must
then be integrated into each real experiment's source-authorized runner
with its own preregistered fail-closed gates, honest source receipts,
source-to-output graph lineage, and frozen TEST policy.

No inference about architecture superiority follows from CI success.
