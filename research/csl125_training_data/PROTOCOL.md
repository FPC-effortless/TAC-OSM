# CSL125 R4 training-data and tokenizer preparation

Parent: PR721 at 087eee4b9d571c201fd24e62f559a13ba15b8648. Continue the same <=125M tool caller; no architecture change or merge.

## Pre-execution scope
Owner says Proceed to preparing reproducible tool-calling training. This increment acquires ONLY publisher mllmTeam/DroidCall README.md, annotated_api.jsonl and DroidCall_train.jsonl after recording an immutable Hub revision, public-access state and license declaration. NEVER fetch DroidCall_test.jsonl or the preformatted code_short mirror. The publisher declares synthetic provenance; public availability is NOT real-world outcome evidence or independent scientific qualification. Do not use source labels to choose the inference tool catalogue. Preserve exact file hashes and quarantine unsupported contracts/labels instead of repairing gold answers silently.

Preparation: profile source schema, duplicates and conflicting labels, create deterministic normalized-request-group FIT/DEV splits from official TRAIN only, fit byte-level BPE on FIT only, compare token counts at candidate capacities, test UTF-8, exact JSON and prompt/target masking. Keep the untouched publisher TEST unopened. Register all exclusions with source line hashes, not vague drops. Do not claim unseen-tool generalization from an in-catalogue grouped split. Tools in the approved catalog are visible public inputs, not selected by gold labels.

Budget: bounded GitHub CPU, no GPU allocation. Source audit and tokenizer preprocessing only; no neural optimizer steps and no checkpoint promotion. A deterministic BPE vocabulary fit is a preprocessing transform, not a trained neural model. All corpus rows remain synthetic labels, not observed executions.

## Admission boundary
The existing repository requires source/access-separated evaluator/ACL evidence, current declared-target-v2 FAIR_COMPARISON preflight, actual matched baseline counts and approved compute BEFORE model fitting. This continuation must neither fabricate signatures nor treat passing software CI as permission. Record remaining blockers explicitly. The removed 100K ceiling is not applied. Full model hard cap remains125M. Use existing opt-in FP64-memory repair as the future reference; CUDA precision/cost and 2xT4 allocation remain unqualified.

## Output
Source receipt, data audit, duplicate/conflict quarantine ledger, FIT/DEV manifests, preprocessing/tokenizer artifacts, integration tests and honest model-training admission status. No external tools executed; every learned-confidence head remains uncalibrated.
