# ARC-AGI-3 public-development baseline

**Status:** A deterministic, untrained, bounded-memory action-selection baseline with tamper-evident public action replay; it is not an adaptive world model and has **no official or actual game score** until run against a genuine permitted local ARC environment. Tests use made-up interface fixtures and are only software diagnostics.

## Official interface and setup

Official open [ARC-AGI toolkit](https://github.com/arcprize/ARC-AGI) exposes `Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=...)`, `make(game_id)`, `env.observation_space`, `env.action_space`, and `env.step(action, data=...)`. Use legitimately obtained **public development** game files only; no hidden games, evaluation API, credentials, guessed objectives or hardcoded per-game routes. Pin the toolkit version in your run's evidence.

1. Install the official toolkit in a separate Python 3.12 environment: `python -m pip install arc-agi==0.9.9`.
2. Put legitimately available public local environments in a dedicated directory (required by the CLI). No network fallback.
3. Run `python scripts/arc3_public_baseline.py --game YOUR_PUBLIC_ID --environments /path/to/public_games --out /tmp/arc3_run.json --max-actions 80`.
4. Run constructed unit tests with `python -m unittest discover -s tests -p 'test_arc3_public_baseline.py' -v`.

GitHub CI installs and checks this exact toolkit version and its action enum interface **without opening any public or hidden game**. This is an API compatibility check, not a performance result.\n\nThe agent selects non-RESET legal actions using bounded visible-frame state-action visit counts; complex-action coordinates default to the middle of the first observed grid. This is an intentionally weak **blind** baseline; it does not learn hidden rules, optimize scores, or generate planned trajectories. It stops on WIN or GAME_OVER or a hard action budget. No repeated game reset retries.

## Evidence and anti-cheat

Output JSON includes public observation SHA256s, action names, public post-action state, levels completed, hash-chained entries, a bounded-memory limit and explicit `official_score: NOT_EVALUATED`. The chain catches post-record changes **if the final head hash is trusted/preserved independently**; by itself it cannot establish genuine game provenance or evaluator identity. Replay hashes are not a substitute for full raw authorized frame capture or independent scorer records.

For qualification, store run code+commit, actual official toolkit version, public game/version/seed, full authorized (privacy-safe) observation/action recordings, run budgets, external signed evaluator score, human-normalized action efficiency, measured wall time, inference cost, memory, and honest failures. Only a separately authorized official held-out evaluation can support an official score. ARC benchmark performance is not evidence of real-world commercial utility or general AGI.

## Next actual capabilities

- Source-authorized object-state encoder and validated state transition learning.
- Online system/goal identification under unseen mechanics and budgets.
- Information-gain exploration and bounded lookahead planning.
- Legitimate state retention and replay across actions, no benchmark-name shortcut.
- Compare against deterministic, strong structured-memory, and LLM/recurrent agent controls with actual resource accounting. No seventh TAC model family and no unapproved neural fitting.
