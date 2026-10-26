# ME111 original artifact contract

Prospective record: exactly five original new models, seeds111001–111005,1536updates. Each original full model/optimizer/RNG at512,1024and1536(final.pt), complete training history, live budget receipts, fixedFITmonitor arrays at0/512/1024/1536. Before REPORT, allfive original fits and receipts must be committed and pushed. No optimizer resume/retries/overwrite or favorable checkpoint selection.

FullME110parents are retained; ME111source pin0e712a798ce943b9f25c306bec1250e0b1e29031; later reciprocal/doc commits do not alter training source. Complete45tensor source hashes and actual plan hashes inFAIR_COMPARISON.json. Primary source UCI501 hash d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8. Report original parent lineage and additional new updates, not a compute-matched superiority comparison.

After original final scoring, independent replay commands (no optimizer):

    PYTHONPATH=src:. python scripts/me111_conditional_training.py replay
    PYTHONPATH=src:. python scripts/me111_replay_audit.py --replay
    PYTHONPATH=src:. python scripts/me111_summary.py --replay

Raw forecast replay contract relative1e-5/absolute1e-6. Source-reference graph/gradient contracts FP32relative2e-5/absolute2e-6 and FP64relative1e-9/absolute1e-10, fixed before fitting. Any violation is FAIL, not permission to change thresholds. Later numeric rows2016–2017 are not parsed/scored. Original exposed2015 is retrospective, not qualification on an untouched cohort.

Current record is pre-training; no model fit or empirical gate pass is claimed by this document. PNDSdraftPR37 supplies Python law/mask/routing reference and limited Lean claims; full source/neural/IEEE refinement and statistical calibration remainOPEN. TACdraftPR442; no merge/deployment.
