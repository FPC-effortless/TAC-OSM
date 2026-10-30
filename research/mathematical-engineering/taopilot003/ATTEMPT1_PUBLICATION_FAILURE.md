# TAOPILOT003 — preserved attempt-1 publication failure (2026-10-30)

**Evidence level: real FIT execution occurred, but original checkpoint preservation FAIL. This is NOT a completed study or qualified optimizer result.**

Original GitHub Actions run: https://github.com/FPC-effortless/TAC-OSM/actions/runs/41011047100
Job: `123916758843`.

- Licensed authentic CPython FIT original-source / actual model registration preflight: **PASS**.
- All five original 32-update per-arm true next-byte training loops actually executed, all five in-job cold-model/optimizer verifications completed, all five checkpoint hashes verified in that job's transient filesystem.
- AdamW ephemeral hash: `96a41a50f7bbdb701fddc36f85a6c1193a2f5e8b654f789a644d3b3e1cf3ea90`
- Muon ephemeral hash: `c75fee2cdcbff2706359804d84a2251e0af405a26b6ec25ed0befeaee54c538d`
- AdaMuon ephemeral hash: `387bccb4f3f652db2b0a8933c8e477bba594eb55ac6e99a520f6fc201ec5ca5e`
- NorMuon ephemeral hash: `1e80abbffa8e5b5886f65391980e8d209890b9ac554a1d990640cc5783721c67`
- Pre-NS ephemeral hash: `3ec607feefcdb895b87befcd057aeb9bf2a6b490b4d61985ca238284a5350070`
- **FAIL: no Git-backed trained weights.** Final `git push` was rejected as non-fast-forward because companion independent-replay workflow code had been added after checkout. Transient original bytes are not accessible; no old hash is claimed as retrievable. No DEV was scored.
- Result: `REPO_REPLAY=FAIL`, `ORIGINAL_CHECKPOINT_PRESERVATION=FAIL`, `QUALIFIED_PREDICTIVE_GAIN=NOT_MEASURED`. Do not infer a winner, treat these as replayable original model files, or replace/fabricate original bytes.
- Corrective fix: branch-safe `git fetch; git rebase; git push` in the FIT workflow. A subsequent entire five-arm run is a **new attempt**, not recovery of the irretrievably ephemeral first originals. Record both attempts.
