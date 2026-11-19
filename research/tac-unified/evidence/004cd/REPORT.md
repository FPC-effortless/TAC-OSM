# TAC-UNIFIED-004C/D — hosted constructed diagnostics

Qualification: **NOT_QUALIFIED**. Five seeds 201–205, pinned Veritas TrainLab and Research Lab OS commit eec439492407b0aa26922c23b5ae40d95289f4eb. No optimizer steps, GPU use, authentic evaluation, neural routing or native CSL integration. No external independent replication.

- Preregistered protocol: research/tac-unified/004cd/PROTOCOL.md (initial commit a86993272fd2d18a6a2203cd4bd4af5dfbd229c1).
- Failed checksum-transcription preflight: https://github.com/FPC-effortless/TAC-OSM/actions/runs/44585329558; preserved in 44585329558-1. Corrected before any diagnostic ran; no scientific changes.
- Successful pinned run: https://github.com/FPC-effortless/TAC-OSM/actions/runs/44585518497 at 7d0760491c4aed1e8724bdee8c3eaee11367c6ec.
- Full unpacked receipts, raw outputs, per-seed results and Research Lab memory/database: 44585518497-1 (SHA256SUMS).
- Original Actions artifact: https://github.com/FPC-effortless/TAC-OSM/actions/runs/44585518497/artifacts/14561147205, SHA256 44933f10cd89a6dfa65e9738b74979b23d512fe3cacaaa0093c7821db00ebd78.
- All five file-hash verification receipts valid, five Research Lab evidence import pins. Compulsory repository gate succeeded: https://github.com/FPC-effortless/TAC-OSM/actions/runs/44585529861. This is not scientific qualification.

## C — targeted discovery

Mean across seeds: full 262208 intervention rows, 14.8638888 ms; targeted 9280 intervention rows, 1.6059306 ms. 96.46% fewer interventions, 28.26x intervention-count reduction, 9.26x mean discovery latency speedup. Identical recovered graph and alternate table in 5/5 supported seeds. Joint planner using the targeted graph solved 320/320 goals (64 per seed). Both read-only and cycle challenges rejected in all five targeted runs.

Critical negative: dormant hidden extra dependency falsely accepted by targeted discovery in 5/5, whereas exhaustive discovery rejected in all five. The witness screen does not certify graph completeness. Interventions are simulator rows, not physical acquisitions; timing is a descriptive CPU measurement, not an isolated hardware benchmark.

## D — diversified perception

640 chains per regime, 1024 actions per chain. Final-state correct counts:

|Regime|Repeat primary|Fixed diverse|Adaptive diverse|Forced secondary|Forced mean reads|Forced mean ms|Fixed diverse mean ms|
|---|---:|---:|---:|---:|---:|---:|---:|
|exact8|3|640|640|640|2.41255|104.3124|136.0823|
|variable789|8|639|639|639|2.42307|105.4868|135.2883|
|heavy|8|568|567|567|2.64179|114.0301|137.5838|
|shared_all|9|9|9|9|2.92824|124.6226|136.1528|

All 15 narrow preregistered gates across five seeds passed; severe and shared corruption remain unresolved. Exact-action-chain results (forced): 640, 639, 548, 3 respectively. Forced acquisition did not improve final-state correctness over the unforced diversified adaptive arm in this cohort. Both used nearly identical observation counts. Distinct independent views, not merely rereading or forced acquisition, explain the main gain. Secondary corruption is simulated; no real sensor independence, calibration or distribution-free guarantee. Timing excludes generation/acquisition and includes decoding and simulated execution.

## Next discriminating work

Online residual-triggered invalidation of provisional graph, sparse periodic exploration, targeted refinement and bounded fallback on novel dependencies; evaluate hidden dependency discovery delay and false acceptance, execution cost and replanning overhead. Independent evidence source selection under heavy/shared corruption. Native CSL and neural routing only after authentic source and signed prefit, matched parity and reciprocal PNDS Lean/Python obligations. Preserve all negative results. No merge or promotion.
