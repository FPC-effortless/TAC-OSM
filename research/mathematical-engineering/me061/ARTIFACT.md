# Original ME060/ME061 trained-model archive

Local deliverable:
`/workspace/artifacts/TAC-OSM-ME060-ME061-2026-10-23.tar.gz`

- Size:15,046,491 bytes.
- SHA256:`38c399c9e40ccbfb4c1e64bc24ef7293097fd93932429129b84062c58ef29cb8`.
-20 original fitted checkpoints with full optimizer states and learning traces.
-36 payloads verified byte-for-byte against the internal SHA256 manifest.
- Detailed original forecast-error scores, final resource manifests, source
  hashes, calibration summaries and different-task results included.
- Raw publisher trajectories, secrets and external account data excluded.

This is a workspace artifact, not a GitHub Actions training artifact. Repository
CI validates software/registration contracts; it did not independently train
these models. Research weights stay out of git under repository artifact policy.

ME060 model source commit:ca259e59c832cbc7096d29d343493ef34915d4b2.
ME061 model source commit:9d7b93e9. Both protocols and the complete failed
attempt ledger remain in draft PRs #339 and #340. All models froze before
source CAL/DEV scoring. Target tasks were separately registered before their
trajectories opened, with no target fitting or preprocessing adaptation.

Restore in a fresh folder, check the outer checksum and every `MANIFEST.json`
entry, then use the current research branch's replay helpers:

    PYTHONPATH=src:. python scripts/me061_replay_transfer.py \
      --source-cache /path/to/private/publisher-cache --output /restored/me061

This downloads only the pinned source files needed for CAL/DEV replay if absent.
For the frozen target results:

    PYTHONPATH=src:. python scripts/me061_replay_transfer.py --transfer \
      --source-cache /path/to/private/target-cache --output /restored/me061

Keep downloaded publisher parquets outside artifact payloads. Replay of an
already evaluated target is reproducibility evidence, not a new untouched
test. Use Python3.12.14, PyTorch2.14.1+cpu, NumPy2.3.5 and PyArrow21.0.0;
the original runner used one CPU process thread. See `RESULTS.md` for gate
failures and scientific limitations.
