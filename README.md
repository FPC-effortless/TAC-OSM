# TAC-OSM audited development evidence — 17 October 2026

Six studies, three seeds each; 96 newly fitted configurations plus 27 original DEV-044A checkpoints. **Development evidence, not sealed confirmation or deployment approval.**

Binary release upload returned GitHub HTTP 401. These complete archives are preserved on a dedicated evidence branch through the working Git connection. Main and research source branches contain no evidence binaries.

## Verify and reproduce

1. Run `sha256sum -c SHA256SUMS` in this directory.
2. Extract `continuation-evidence.tar.gz`. Its `FILE-SHA256.json` binds all 168 internal evidence/source/log files.
3. Unzip each `DEV044A-original-seedNNNN.zip` into `original-dev044/NNNN`. These preserve the original GitHub artifact bytes.
4. Clone frozen source with `git clone research-source.bundle source`.
5. Use `source-refs.json` and each report's `source` commit to inspect its frozen recipe. The source bundle contains every training and additional-audit commit.
6. Read the archive's `docs/DEV-044-048-RESULTS.md` and reproduction guide before interpreting numbers.

All six separate replay audits and the 1,253-test GitHub Actions gate passed. Checkpoints, raw predictions/actions/outcomes, calibration records, protocol/source hashes, runtime metadata and CI logs are retained. Audits reproduce inference and physical effects, not optimizer trajectories. Shared-machine timings are descriptive.

DEV-044A froze internal query projections with writer dynamics. DEV-044B0 isolates those groups; lower costs still fail the reliability guard in some seeds. Neural improvements frequently add failures. DEV-048 selected exact public history on development data in every seed. No integrated architectural superiority is established.

Cancelled local checks are retained and explicitly superseded in runtime-and-provenance.json. Earlier DEV-015–043 evidence and main are unchanged. Research PRs #203–207 remain draft and unmerged. Matched baseline tuning/compute, larger confirmation, learned retention, original CASM/CDL parity, GEPA reflection and external-domain validation remain unpassed gates.
